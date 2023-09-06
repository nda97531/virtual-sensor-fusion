"""
Multi-task: classification of all labels (11 classes of UP-Fall) +
    VSF contrastive (UP-Fall)
Sensors: waist accelerometer, skeleton
"""

import itertools
import os
from collections import defaultdict
from copy import deepcopy
from glob import glob
import pandas as pd
import numpy as np
import torch as tr
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vsf.data_generator.augmentation import Rotation3D, HorizontalFlip
from vsf.data_generator.classification_data_gen import FusionDataset, BalancedFusionDataset
from vsf.data_generator.unlabelled_data_gen import UnlabelledFusionDataset
from vsf.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from vsf.flow.vsf_flow import VsfE2eFlow
from vsf.networks.backbone_tcn import TCN
from vsf.networks.complete_model import VsfModel
from vsf.networks.vsf_distributor import VsfDistributor
from vsf.loss_functions.contrastive_loss import CMCLoss
from vahar_datasets_formatter.vahar.datasets.upfall_dataset import UPFallNpyWindow, UPFallConst
from vsf.loss_functions.classification_loss import AutoCrossEntropyLoss


def split_3_sets(df: pd.DataFrame) -> tuple:
    # split TRAIN, VALID, TEST
    # subject 5, 10, 15 for validation
    valid_set_idx = df['subject'] % 5 == 0
    valid_set = df.loc[valid_set_idx]
    # odd subjects as train set
    train_set_idx = (df['subject'] % 2 != 0) & (~valid_set_idx)
    train_set = df.loc[train_set_idx]
    # 1/3 subjects as test set
    test_set_idx = ~(train_set_idx | valid_set_idx)
    test_set = df.loc[test_set_idx]

    return train_set, valid_set, test_set


def load_class_data(parquet_dir: str, window_size_sec=4, step_size_sec=2, min_step_size_sec=0.5,
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
    upfall = UPFallNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        min_step_size_sec=min_step_size_sec,
        max_short_window=max_short_window,
        modal_cols={
            UPFallConst.MODAL_INERTIA: {
                'waist': ['belt_acc_x(m/s^2)', 'belt_acc_y(m/s^2)', 'belt_acc_z(m/s^2)'],
                'wrist': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)'],
            },
            UPFallConst.MODAL_SKELETON: {
                'ske': list(itertools.chain.from_iterable(
                    [f'x_{joint}', f'y_{joint}'] for joint in
                    ['Neck', 'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle']
                ))
            }
        }
    )
    df = upfall.run()
    list_sub_modal = list(itertools.chain.from_iterable(list(sub_dict) for sub_dict in upfall.modal_cols.values()))

    train_set, valid_set, test_set = split_3_sets(df)

    def concat_data_in_df(df):
        # concat sessions in cells into an array
        modal_dict = {}
        for col in df.columns:
            if col not in {'subject', 'trial'}:
                col_data = df[col].tolist()
                modal_dict[col] = np.concatenate(col_data)

        # construct dict with classes are keys
        label_list = np.unique(modal_dict['label'])
        # dict[modal][label index] = window array
        class_dict = defaultdict(dict)
        for label_idx, label_val in enumerate(label_list):
            idx = modal_dict['label'] == label_val
            class_dict['waist'][label_idx] = modal_dict['waist'][idx]
            class_dict['wrist'][label_idx] = modal_dict['wrist'][idx]
            class_dict['ske'][label_idx] = modal_dict['ske'][idx]
        class_dict = dict(class_dict)

        assert list(class_dict.keys()) == list_sub_modal, 'Mismatched submodal list'
        return class_dict

    results = {
        'train': concat_data_in_df(train_set),
        'valid': concat_data_in_df(valid_set),
        'test': concat_data_in_df(test_set)
    }
    return results


def load_unlabelled_data(parquet_dir: str, window_size_sec=4, step_size_sec=1) -> dict:
    """
    Load all the CMDFall dataset into a dict

    Args:
        parquet_dir: path to processed parquet folder
        window_size_sec: window size in second for sliding window
        step_size_sec: step size in second for sliding window

    Returns:
        a 2-level dict:
            dict[train/valid/test][submodal name] = windows array shape [n, ..., channel]
    """
    npy_dataset = UPFallNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        modal_cols={
            UPFallConst.MODAL_INERTIA: {
                'waist': ['belt_acc_x(m/s^2)', 'belt_acc_y(m/s^2)', 'belt_acc_z(m/s^2)'],
                'wrist': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)'],
            },
            UPFallConst.MODAL_SKELETON: {
                'ske': list(itertools.chain.from_iterable(
                    [f'x_{joint}', f'y_{joint}'] for joint in
                    ['Neck', 'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle']
                ))
            }
        }
    )
    df = npy_dataset.run(shift_short_activity=False)

    train_set, valid_set, _ = split_3_sets(df)

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
        'valid': concat_data_in_df(valid_set)
    }
    return three_dicts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda:0')

    parser.add_argument('--name', '-n', required=True,
                        help='name of the experiment to create a folder to save weights')

    parser.add_argument('--class-data-folder', '-lbl',
                        default='/home/ducanh/parquet_datasets/UP-Fall/',
                        help='path to parquet data folder - classification task')

    parser.add_argument('--unlabelled-data-folder', '-ulb',
                        default='/home/ducanh/parquet_datasets/UP-Fall/',
                        help='path to parquet data folder - contrastive learning task')

    parser.add_argument('--output-folder', '-o', default='./log/upfall',
                        help='path to save training logs and model weights')

    args = parser.parse_args()

    NUM_REPEAT = 3
    NUM_EPOCH = 300
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0
    EARLY_STOP_PATIENCE = 30
    LR_SCHEDULER_PATIENCE = 15
    TRAIN_BATCH_SIZE = 32

    # load data
    three_class_dicts = load_class_data(parquet_dir=args.class_data_folder)
    train_cls_dict = three_class_dicts['train']
    valid_cls_dict = three_class_dicts['valid']
    test_cls_dict = three_class_dicts['test']
    del three_class_dicts

    three_unlabelled_dicts = load_unlabelled_data(parquet_dir=args.unlabelled_data_folder)
    train_unlabelled_dict = three_unlabelled_dicts['train']
    valid_unlabelled_dict = three_unlabelled_dicts['valid']
    del three_unlabelled_dicts
    assert train_cls_dict['waist'][0].shape[1:] == train_unlabelled_dict['waist'].shape[1:]

    test_scores = []
    model_paths = []
    # train 3 times
    for _ in range(NUM_REPEAT):
        # create model
        backbone = tr.nn.ModuleDict({
            'waist': TCN(
                input_shape=train_unlabelled_dict['waist'].shape[1:],
                how_flatten='spatial attention gap',
                n_tcn_channels=(64,) * 5 + (128,) * 2,
                tcn_drop_rate=0.5,
                use_spatial_dropout=True,
                conv_norm='batch',
                attention_conv_norm=''
            ),
            'wrist': TCN(
                input_shape=train_unlabelled_dict['wrist'].shape[1:],
                how_flatten='spatial attention gap',
                n_tcn_channels=(64,) * 5 + (128,) * 2,
                tcn_drop_rate=0.5,
                use_spatial_dropout=True,
                conv_norm='batch',
                attention_conv_norm=''
            ),
            'ske': TCN(
                input_shape=train_unlabelled_dict['ske'].shape[1:],
                how_flatten='spatial attention gap',
                n_tcn_channels=(64,) * 4 + (128,) * 2,
                tcn_drop_rate=0.5,
                use_spatial_dropout=True,
                conv_norm='batch',
                attention_conv_norm=''
            )
        })

        head = VsfDistributor(
            input_dims={modal: 128 for modal in backbone.keys()},  # affect contrast loss order
            num_classes={  # affect class logit order
                'waist': len(train_cls_dict[list(train_cls_dict.keys())[0]]),
                'wrist': len(train_cls_dict[list(train_cls_dict.keys())[0]]),
                'ske': len(train_cls_dict[list(train_cls_dict.keys())[0]])
            },
            contrastive_loss_func=CMCLoss(temp=0.1),
            cls_dropout=0.5,
            contrast_feature_dim=None,
        )
        model = VsfModel(
            backbones=backbone,
            distributor_head=head
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
            cls_loss_fn=AutoCrossEntropyLoss(confidence_loss_weight=1),
            callbacks=[
                ModelCheckpoint(NUM_EPOCH, model_file_path, smaller_better=False),
                EarlyStop(EARLY_STOP_PATIENCE, smaller_better=False),
                ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=LR_SCHEDULER_PATIENCE, verbose=True)
            ],
            callback_criterion='f1_waist'
        )

        # train and valid
        augmenter = {
            'waist': Rotation3D(angle_range=30),
            'wrist': Rotation3D(angle_range=30)
        }
        train_set_cls = BalancedFusionDataset(deepcopy(train_cls_dict), augmenters=augmenter)
        valid_set_cls = FusionDataset(deepcopy(valid_cls_dict))

        augmenter = {
            'waist': Rotation3D(angle_range=180),
            'wrist': Rotation3D(angle_range=180),
            'ske': HorizontalFlip()
        }
        train_set_unlabelled = UnlabelledFusionDataset(deepcopy(train_unlabelled_dict), augmenters=augmenter)
        valid_set_unlabelled = UnlabelledFusionDataset(deepcopy(valid_unlabelled_dict))

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
            num_epochs=NUM_EPOCH
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
