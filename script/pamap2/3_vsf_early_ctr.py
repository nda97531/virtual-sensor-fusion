"""
Labeled sensors: chest + hand + ankle; acceleromter + magnetometer
Unlabeled sensors: chest + hand + ankle; acceleromter + magnetometer + gyroscope + orientation
- Classification:
    [late fusion acc+mag of all sensor positions]
    [acc] of every position
    [mag] of every position
    [early fusion gyro of all sensor positions]
    [early fusion ori of all sensor positions]
- Contrast: same as classification
"""

import itertools
import os
from collections import defaultdict
from copy import deepcopy
from glob import glob
import time
import numpy as np
import torch as tr
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vahar_datasets_formatter.vahar.datasets.pamap2 import Pamap2NpyWindow, Pamap2Const
from vsf.data_generator import augmentation
from vsf.data_generator.classification_data_gen import FusionDataset, BalancedFusionDataset
from vsf.data_generator.unlabelled_data_gen import UnlabelledFusionDataset
from vsf.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from vsf.networks.backbone_resnet1d import ResNet1D
from vsf.networks.complete_model import VsfModel
from vsf.networks.vsf_distributor import VsfDistributor
from vsf.loss_functions.contrastive_loss import MultiviewInfoNCELoss
from vsf.flow.vsf_flow import VsfE2eFlow


def load_cls_data(parquet_dir: str, window_size_sec=5.12, step_size_sec=2.56) -> dict:
    """
    Load all the Pamap2 dataset into a dataframe

    Args:
        parquet_dir: path to processed parquet folder
        window_size_sec: window size in second for sliding window
        step_size_sec: step size in second for sliding window

    Returns:
        a 3-level dict:
            dict[train/valid/test][submodal name][label index] = windows array shape [n, ..., channel]
    """
    pamap2_dataset = Pamap2NpyWindow(
        only_protocol=True,
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        exclude_labels=[0], # excludes 'other' class label
        no_transition=True,
        modal_cols={
            Pamap2Const.INERTIA_MODAL: {
                'chest_acc': ['chest_acc_x(m/s^2)', 'chest_acc_y(m/s^2)', 'chest_acc_z(m/s^2)'],
                'hand_acc': ['hand_acc_x(m/s^2)', 'hand_acc_y(m/s^2)', 'hand_acc_z(m/s^2)'],
                'ankle_acc': ['ankle_acc_x(m/s^2)', 'ankle_acc_y(m/s^2)', 'ankle_acc_z(m/s^2)'],

                'chest_mag': ['chest_mag_x(uT)', 'chest_mag_y(uT)', 'chest_mag_z(uT)'],
                'hand_mag': ['hand_mag_x(uT)', 'hand_mag_y(uT)', 'hand_mag_z(uT)'],
                'ankle_mag': ['ankle_mag_x(uT)', 'ankle_mag_y(uT)', 'ankle_mag_z(uT)'],

                'gyro': ['chest_gyro_x(rad/s)', 'chest_gyro_y(rad/s)', 'chest_gyro_z(rad/s)',
                         'hand_gyro_x(rad/s)', 'hand_gyro_y(rad/s)', 'hand_gyro_z(rad/s)',
                         'ankle_gyro_x(rad/s)', 'ankle_gyro_y(rad/s)', 'ankle_gyro_z(rad/s)'],

                'ori': ['chest_orientation_x(invalid)', 'chest_orientation_y(invalid)', 'chest_orientation_z(invalid)', 'chest_orientation_w(invalid)',
                        'hand_orientation_x(invalid)', 'hand_orientation_y(invalid)', 'hand_orientation_z(invalid)', 'hand_orientation_w(invalid)',
                        'ankle_orientation_x(invalid)', 'ankle_orientation_y(invalid)', 'ankle_orientation_z(invalid)', 'ankle_orientation_w(invalid)'],
            }
        }
    )
    df = pamap2_dataset.run()
    list_sub_modal = list(itertools.chain.from_iterable(list(sub_dict) for sub_dict in pamap2_dataset.modal_cols.values()))

    # split TRAIN, TEST
    test_set_idx = df['subject'].isin([5, 6])
    test_set = df.loc[test_set_idx]
    train_set_idx = ~test_set_idx
    train_set = df.loc[train_set_idx]

    def concat_data_in_df(df):
        # concat sessions in cells into an array
        # dict key: modal (including label); value: array shape [num window, widow size, channel]
        modal_dict = {}
        print('data shape:')
        for col in df.columns:
            if col != 'subject':
                col_data = df[col].tolist()
                modal_dict[col] = np.concatenate(col_data)
                print(col, modal_dict[col].shape)

        # construct dict with classes are keys
        # key lvl 1: modal; key lvl 2: label; value: array shape [num window, widow size, channel]
        label_list = np.unique(modal_dict['label'])

        # dict[modal][label index] = window array
        class_dict = defaultdict(dict)
        for label_idx, label_val in enumerate(label_list):
            idx = modal_dict['label'] == label_val
            class_dict['chest_acc'][label_idx] = modal_dict['chest_acc'][idx]
            class_dict['hand_acc'][label_idx] = modal_dict['hand_acc'][idx]
            class_dict['ankle_acc'][label_idx] = modal_dict['ankle_acc'][idx]

            class_dict['chest_mag'][label_idx] = modal_dict['chest_mag'][idx]
            class_dict['hand_mag'][label_idx] = modal_dict['hand_mag'][idx]
            class_dict['ankle_mag'][label_idx] = modal_dict['ankle_mag'][idx]

            class_dict['gyro'][label_idx] = modal_dict['gyro'][idx]

            class_dict['ori'][label_idx] = modal_dict['ori'][idx]

        class_dict = dict(class_dict)

        assert list(class_dict.keys()) == list_sub_modal, 'Mismatched submodal list'
        return class_dict

    results = {
        'train': concat_data_in_df(train_set),
        'test': concat_data_in_df(test_set)
    }
    return results


def load_unlabelled_data(parquet_dir: str, window_size_sec=5.12, step_size_sec=2.56) -> dict:
    """
    Load Pamap2 as an unlabeled dataset

    Args:
        parquet_dir: path to processed parquet folder
        window_size_sec: window size in second for sliding window
        step_size_sec: step size in second for sliding window

    Returns:
        a 2-level dict:
            dict[train/valid/test][submodal name] = windows array shape [n, ..., channel]
    """
    npy_dataset = Pamap2NpyWindow(
        only_protocol=False,
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        modal_cols={
            Pamap2Const.INERTIA_MODAL: {
                'chest_acc': ['chest_acc_x(m/s^2)', 'chest_acc_y(m/s^2)', 'chest_acc_z(m/s^2)'],
                'hand_acc': ['hand_acc_x(m/s^2)', 'hand_acc_y(m/s^2)', 'hand_acc_z(m/s^2)'],
                'ankle_acc': ['ankle_acc_x(m/s^2)', 'ankle_acc_y(m/s^2)', 'ankle_acc_z(m/s^2)'],

                'chest_mag': ['chest_mag_x(uT)', 'chest_mag_y(uT)', 'chest_mag_z(uT)'],
                'hand_mag': ['hand_mag_x(uT)', 'hand_mag_y(uT)', 'hand_mag_z(uT)'],
                'ankle_mag': ['ankle_mag_x(uT)', 'ankle_mag_y(uT)', 'ankle_mag_z(uT)'],

                'gyro': ['chest_gyro_x(rad/s)', 'chest_gyro_y(rad/s)', 'chest_gyro_z(rad/s)',
                         'hand_gyro_x(rad/s)', 'hand_gyro_y(rad/s)', 'hand_gyro_z(rad/s)',
                         'ankle_gyro_x(rad/s)', 'ankle_gyro_y(rad/s)', 'ankle_gyro_z(rad/s)'],

                'ori': ['chest_orientation_x(invalid)', 'chest_orientation_y(invalid)', 'chest_orientation_z(invalid)', 'chest_orientation_w(invalid)',
                        'hand_orientation_x(invalid)', 'hand_orientation_y(invalid)', 'hand_orientation_z(invalid)', 'hand_orientation_w(invalid)',
                        'ankle_orientation_x(invalid)', 'ankle_orientation_y(invalid)', 'ankle_orientation_z(invalid)', 'ankle_orientation_w(invalid)'],
            }
        }
    )
    df = npy_dataset.run()

    test_set_idx = df['subject'].isin([5, 6])
    test_set = df.loc[test_set_idx]
    train_set_idx = ~test_set_idx
    train_set = df.loc[train_set_idx]

    def concat_data_in_df(df):
        # concat sessions in cells into an array
        # dict key: modal (including label); value: array shape [num window, widow size, channel]
        modal_dict = {}
        for col in df.columns:
            if col not in {'subject', 'label'}:
                col_data = df[col].tolist()
                modal_dict[col] = np.concatenate(col_data)
        return modal_dict

    three_dicts = {
        'train': concat_data_in_df(train_set),
        'test': concat_data_in_df(test_set)
    }
    return three_dicts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda:0')

    parser.add_argument('--name', '-n', required=True,
                        help='name of the experiment to create a folder to save weights')

    parser.add_argument('--data-folder', '-data',
                        default='/home/ducanh/parquet_datasets/Pamap2/',
                        help='path to parquet data folder')

    parser.add_argument('--output-folder', '-o', default='./log/pamap2',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    NUM_REPEAT = 1
    MAX_EPOCH = 300
    MIN_EPOCH = 0
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOP_PATIENCE = 30
    LR_SCHEDULER_PATIENCE = 15
    TRAIN_BATCH_SIZE = 32

    # load data
    three_class_dicts = load_cls_data(parquet_dir=args.data_folder)
    train_cls_dict = three_class_dicts['train']
    test_cls_dict = three_class_dicts['test']
    del three_class_dicts

    three_unlabelled_dicts = load_unlabelled_data(parquet_dir=args.data_folder)
    train_unlabelled_dict = three_unlabelled_dicts['train']
    test_unlabelled_dict = three_unlabelled_dicts['test']
    del three_unlabelled_dicts

    first_modal = list(train_cls_dict.keys())[0]
    assert train_cls_dict[first_modal][0].shape[1:] == train_unlabelled_dict[first_modal].shape[1:]

    test_scores = []
    model_paths = []
    # train 3 times
    for _ in range(NUM_REPEAT):
        # create model
        backbone = tr.nn.ModuleDict({
            'chest_acc': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=4, stride=4),
            'hand_acc': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=4, stride=4),
            'ankle_acc': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=4, stride=4),

            'chest_mag': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=4, stride=4),
            'hand_mag': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=4, stride=4),
            'ankle_mag': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=4, stride=4),

            'gyro': ResNet1D(in_channels=9, base_filters=128, kernel_size=9, n_block=4, stride=4),

            'ori': ResNet1D(in_channels=12, base_filters=128, kernel_size=9, n_block=4, stride=4),
        })

        num_cls = len(train_cls_dict[list(train_cls_dict.keys())[0]])
        head = VsfDistributor(
            input_dims={'imu_cls': 128, 'imu_ctr': 128} | {modal: 128 for modal in backbone.keys()},  # affect contrast loss order
            num_classes={'imu_cls': num_cls} | {modal: num_cls for modal in backbone.keys()},  # affect class logit order
            contrastive_loss_func=MultiviewInfoNCELoss(),
            cls_dropout=0.5
        )
        model = VsfModel(
            backbones=backbone, distributor_head=head,
            connect_feature_dims={'imu_cls': [768, 128], 'imu_ctr': [768, 128]},
            cls_fusion_modals=['chest_acc+hand_acc+ankle_acc+chest_mag+hand_mag+ankle_mag'],
            ctr_fusion_modals=['chest_acc+hand_acc+ankle_acc+chest_mag+hand_mag+ankle_mag'],
            fusion_name_alias={'chest_acc+hand_acc+ankle_acc+chest_mag+hand_mag+ankle_mag': 'imu'}
        )

        # create folder to save result
        save_folder = f'{args.output_folder}/{args.name}'
        last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
        last_run = max(last_run) + 1 if len(last_run) > 0 else 0
        save_folder = f'{save_folder}/run_{last_run}'

        # create training config
        optimizer = tr.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model_file_path = f'{save_folder}/model.pth'
        flow = VsfE2eFlow(
            model=model,
            optimizer=optimizer,
            device=args.device,
            callbacks=[
                ModelCheckpoint(MAX_EPOCH, model_file_path, smaller_better=False),
                EarlyStop(EARLY_STOP_PATIENCE, smaller_better=False),
                ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=LR_SCHEDULER_PATIENCE, verbose=True)
            ],
            callback_criterion='f1_imu_cls',
            name=args.name
        )

        # train and valid
        augmenter = {
            key: augmentation.ComposeAugmenters([
                augmentation.TimeWarp(),
                augmentation.MagnitudeWarp()
            ])
            for key in [
                'chest_acc', 'hand_acc', 'ankle_acc',
                'chest_mag', 'hand_mag', 'ankle_mag',
                'gyro', 'ori'
            ]
        }
        train_set_cls = BalancedFusionDataset(deepcopy(train_cls_dict), augmenters=augmenter)
        valid_set_cls = FusionDataset(deepcopy(test_cls_dict))

        timewarp_seed = int(time.time())
        augmenter = {
            key: augmentation.ComposeAugmenters([
                augmentation.TimeWarp(sigma=0.3, random_seed=timewarp_seed),
                augmentation.MagnitudeWarp(sigma=0.25)
            ])
            for key in [
                'chest_acc', 'hand_acc', 'ankle_acc',
                'chest_mag', 'hand_mag', 'ankle_mag',
                'gyro', 'ori'
            ]
        }
        train_set_unlabelled = UnlabelledFusionDataset(deepcopy(train_unlabelled_dict), augmenters=augmenter)
        valid_set_unlabelled = UnlabelledFusionDataset(deepcopy(test_unlabelled_dict))

        train_loader = {
            'cls': DataLoader(train_set_cls, batch_size=TRAIN_BATCH_SIZE // 2, shuffle=True),
            'contrast': DataLoader(train_set_unlabelled, batch_size=TRAIN_BATCH_SIZE // 2, shuffle=True)
        }
        valid_loader = {
            'cls': DataLoader(valid_set_cls, batch_size=64, shuffle=False),
            'contrast': DataLoader(valid_set_unlabelled, batch_size=64, shuffle=False),
        }
        train_log, valid_log = flow.run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            max_epochs=MAX_EPOCH, min_epochs=MIN_EPOCH
        )

        # test
        test_set = FusionDataset(deepcopy(test_cls_dict))
        test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
        test_score = flow.run_test_epoch(test_loader, model_state_dict=tr.load(model_file_path))
        test_scores.append(test_score)
        model_paths.append(model_file_path)

        # save log
        train_log.to_csv(f'{save_folder}/train.csv', index=False)
        valid_log.to_csv(f'{save_folder}/valid.csv', index=False)
        test_score.to_csv(f'{save_folder}/test.csv', index=True)
        logger.info("Done!")

    print(f'Mean test score of {NUM_REPEAT} runs:')
    print(*model_paths, sep='\n')
    print(sum(test_scores) / len(test_scores))
