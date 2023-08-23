"""
Exp 1.4
"""

import itertools
import os
from collections import defaultdict
from copy import deepcopy
from glob import glob

import numpy as np
import torch as tr
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vsf.data_generator.augmentation import Rotation3D
from vsf.data_generator.classification_data_gen import FusionDataset, BalancedFusionDataset

from vsf.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from vsf.flow.vsf_flow import VSFFlow
from vsf.networks.backbone_tcn import TCN
from vsf.networks.complete_model import VsfModel
from vsf.networks.vsf_distributor import VsfDistributor
from vsf.public_datasets.sfu_imu_dataset import SFUNpyWindow, SFUConst
from vsf.loss_functions.contrastive_loss import CocoaLoss


def load_data(parquet_dir: str, window_size_sec=4, step_size_sec=2, min_step_size_sec=0.5,
              max_short_window=5) -> dict:
    """
    Load all the UP-Fall dataset into a dataframe

    Args:
        parquet_dir: path to processed parquet folder
        window_size_sec: window size in second for sliding window
        step_size_sec: step size in second for sliding window
        min_step_size_sec: min step size in second for shifting window
        max_short_window: number of window for each session of short activities

    Returns:
        a 3-level dict:
            dict[train/valid/test][submodal name][label index] = windows array shape [n, ..., channel]
    """
    npy_loader = SFUNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        min_step_size_sec=min_step_size_sec,
        max_short_window=max_short_window,
        modal_cols={
            SFUConst.MODAL: {
                f'{pos}_acc': [f'{pos}_acc_{axis}(m/s^2)' for axis in ['x', 'y', 'z']]
                for pos in SFUConst.SENSOR_POSITIONS
            }
        }
    )
    df = npy_loader.run()
    list_sub_modal = list(itertools.chain.from_iterable(list(sub_dict) for sub_dict in npy_loader.modal_cols.values()))

    # split TRAIN, VALID, TEST
    assert np.unique(df['subject']).tolist() == list(range(1, 11))
    train_subject = {2, 4, 6, 8}
    test_subject = {1, 3, 5, 7}
    valid_subject = {9, 10}

    # split TRAIN, VALID, TEST
    # assert np.unique(df['subject']).tolist() == list(range(1, 11))
    # train_subject = {2, 4, 6, 8}
    # test_subject = {1, 3, 5, 7}
    # valid_subject = {9, 10}

    # train_set_idx = df['subject'].isin(train_subject)
    # valid_set_idx = df['subject'].isin(valid_subject)
    # assert not (train_set_idx & valid_set_idx).any(), 'Train and valid overlapped'
    # test_set_idx = ~(train_set_idx | valid_set_idx)
    randomizer = np.random.default_rng(100)
    random_idx = randomizer.permutation(len(df))
    assert len(random_idx) == 600
    train_set_idx = random_idx[:250]
    valid_set_idx = random_idx[250:350]
    test_set_idx = random_idx[350:600]

    train_set = df.iloc[train_set_idx]
    valid_set = df.iloc[valid_set_idx]
    test_set = df.iloc[test_set_idx]

    def concat_data_in_df(df):
        # concat sessions in cells into an array
        modal_dict = {}
        for col in df.columns:
            if col != 'subject':
                col_data = df[col].tolist()
                modal_dict[col] = np.concatenate(col_data)

        class_dict = defaultdict(dict)
        for sub_modal in list_sub_modal:
            for label in [0, 1]:
                idx = modal_dict['label'] == label
                class_dict[sub_modal][label] = modal_dict[sub_modal][idx]

        assert list(class_dict.keys()) == list_sub_modal, 'Mismatched submodal list'
        return class_dict

    three_dicts = {
        'train': concat_data_in_df(train_set),
        'valid': concat_data_in_df(valid_set),
        'test': concat_data_in_df(test_set)
    }
    return three_dicts


if __name__ == '__main__':
    tr.autograd.set_detect_anomaly(True)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda:0')

    parser.add_argument('--name', '-n', default='exp1_4',
                        help='name of the experiment to create a folder to save weights')

    parser.add_argument('--data-folder', '-data',
                        default='/mnt/data_partition/UCD/UCD04 - Virtual sensor fusion/processed_parquet/SFU-IMU',
                        help='path to parquet data folder')

    parser.add_argument('--output-folder', '-o', default='./log/sfu',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    NUM_REPEAT = 1
    NUM_EPOCH = 300
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-3
    EARLY_STOP_PATIENCE = 30
    LR_SCHEDULER_PATIENCE = 15
    TRAIN_BATCH_SIZE = 16

    # load data
    three_dicts = load_data(parquet_dir=args.data_folder)
    train_dict = three_dicts['train']
    valid_dict = three_dicts['valid']
    test_dict = three_dicts['test']
    del three_dicts

    test_scores = []
    model_paths = []
    # train 3 times
    for _ in range(NUM_REPEAT):
        # create model
        backbone = nn.ModuleDict({
            f'{pos}_acc': TCN(
                input_shape=(200, 3),
                how_flatten='spatial attention gap',
                n_tcn_channels=(64,) * 5 + (128,) * 2,
                tcn_drop_rate=0.5,
                use_spatial_dropout=False,
                conv_norm='batch',
                attention_conv_norm=''
            )
            for pos in SFUConst.SENSOR_POSITIONS
        })
        num_class = len(train_dict[list(train_dict.keys())[0]])
        head = VsfDistributor(
            input_dims={modal: 128 for modal in backbone.keys()},
            num_classes={
                submodal: num_class
                for submodal in backbone.keys()
            },
            contrastive_loss_func=CocoaLoss(temp=0.1, norm2_eps=1e-8),
            contrast_feature_dim=None,
            main_modal=None
        )
        model = VsfModel(
            backbones=backbone,
            distributor_head=head,
            dropout=0.5
        )

        # create folder to save result
        save_folder = f'{args.output_folder}/{args.name}'
        last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
        last_run = max(last_run) + 1 if len(last_run) > 0 else 0
        save_folder = f'{save_folder}/run_{last_run}'

        # create training config
        cls_optimizer = tr.optim.SGD(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9
        )
        contrast_optimizer = None  # tr.optim.SGD(
        #     model.backbones['wrist_acc'].parameters(),
        #     lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9
        # )

        model_file_path = f'{save_folder}/model.pth'
        flow = VSFFlow(
            model=model, cls_optimizer=cls_optimizer, contrast_optimizer=contrast_optimizer,
            device=args.device,
            callbacks=[
                ModelCheckpoint(NUM_EPOCH, model_file_path, smaller_better=True),
                EarlyStop(EARLY_STOP_PATIENCE, smaller_better=True),
                ReduceLROnPlateau(optimizer=cls_optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, verbose=True),
                # ReduceLROnPlateau(optimizer=contrast_optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE,
                #                   verbose=True)
            ],
            callback_criterion='cls loss'
        )

        # train and valid
        augmenter = {
            submodal: Rotation3D(angle_range=180, p=1, random_seed=None)
            for submodal in backbone.keys()
        }
        logger.info('Constructing training set object')
        train_set = BalancedFusionDataset(deepcopy(train_dict), augmenters=augmenter)
        logger.info('Constructing validation set object')
        valid_set = FusionDataset(deepcopy(valid_dict))
        train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
        train_log, valid_log = flow.run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=NUM_EPOCH
        )

        # test
        logger.info('Constructing test set object')
        test_set = FusionDataset(deepcopy(test_dict))
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
