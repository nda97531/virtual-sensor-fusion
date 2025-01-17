"""
Sensor fusion: waist acc + wrist acc + skeleton
Only classification.
"""

import itertools
import os
from collections import defaultdict
from copy import deepcopy
from glob import glob

import numpy as np
import torch as tr
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vahar_datasets_formatter.vahar.datasets.cmdfall import CMDFallNpyWindow, CMDFallConst
from vsf.data_generator.augmentation import Rotation3D
from vsf.data_generator.classification_data_gen import FusionDataset, BalancedFusionDataset
from vsf.flow.single_task_flow import SingleTaskFlow
from vsf.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from vsf.networks.backbone_resnet1d import ResNet1D
from vsf.networks.classifier import BasicClassifier
from vsf.networks.complete_model import FusionClsModel


def load_data(parquet_dir: str, window_size_sec=4, step_size_sec=0.4,
              min_step_size_sec=None, max_short_window=None) -> dict:
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
    npy_dataset = CMDFallNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        min_step_size_sec=min_step_size_sec,
        max_short_window=max_short_window,
        modal_cols={
            CMDFallConst.MODAL_INERTIA: {
                'waist': ['waist_acc_x(m/s^2)', 'waist_acc_y(m/s^2)', 'waist_acc_z(m/s^2)'],
                'wrist': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)']
            },
            CMDFallConst.MODAL_SKELETON: {
                'ske': [c.format(kinect_id=3) for c in CMDFallConst.SELECTED_SKELETON_COLS]
            }
        }
    )
    df = npy_dataset.run()
    list_sub_modal = list(itertools.chain.from_iterable(list(sub_dict) for sub_dict in npy_dataset.modal_cols.values()))

    # split TRAIN, VALID, TEST
    train_set_idx = df['subject'] % 2 != 0
    train_set = df.loc[train_set_idx]

    valid_set_idx = df['subject'] % 10 == 0
    valid_set = df.loc[valid_set_idx]

    test_set_idx = ~(train_set_idx | valid_set_idx)
    test_set = df.loc[test_set_idx]

    def concat_data_in_df(df):
        # concat sessions in cells into an array
        # dict key: modal (including label); value: array shape [num window, widow size, channel]
        modal_dict = {}
        for col in df.columns:
            if col != 'subject':
                col_data = df[col].tolist()
                modal_dict[col] = np.concatenate(col_data)

        # construct dict with classes are keys
        # key lvl 1: modal; key lvl 2: label; value: array shape [num window, widow size, channel]
        label_list = np.unique(modal_dict['label'])
        # dict[modal][label index] = window array
        class_dict = defaultdict(dict)
        for label_idx, label_val in enumerate(label_list):
            # don't count 'unknown' class (class index is 0)
            if label_idx != 0:
                idx = modal_dict['label'] == label_val
                class_dict['waist'][label_idx - 1] = modal_dict['waist'][idx]
                class_dict['wrist'][label_idx - 1] = modal_dict['wrist'][idx]
                class_dict['ske'][label_idx - 1] = modal_dict['ske'][idx]
        class_dict = dict(class_dict)

        assert list(class_dict.keys()) == list_sub_modal, 'Mismatched submodal list'
        return class_dict

    three_dicts = {
        'train': concat_data_in_df(train_set),
        'valid': concat_data_in_df(valid_set),
        'test': concat_data_in_df(test_set)
    }
    return three_dicts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda:0')

    parser.add_argument('--name', '-n', default='exp1_2',
                        help='name of the experiment to create a folder to save weights')

    parser.add_argument('--data-folder', '-data',
                        default='/home/ducanh/parquet_datasets/CMDFall/',
                        help='path to parquet data folder')

    parser.add_argument('--output-folder', '-o', default='./log/cmdfall',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    NUM_REPEAT = 3
    MAX_EPOCH = 300
    MIN_EPOCH = 40
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    EARLY_STOP_PATIENCE = 30
    LR_SCHEDULER_PATIENCE = 15
    TRAIN_BATCH_SIZE = 32

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
        backbone = tr.nn.ModuleDict({
            'waist': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=6, stride=4),
            'wrist': ResNet1D(in_channels=3, base_filters=128, kernel_size=9, n_block=6, stride=4),
            'ske': ResNet1D(in_channels=30, base_filters=128, kernel_size=9, n_block=6, stride=4)
        })
        classifier = BasicClassifier(
            n_features_in=768,
            n_classes_out=len(train_dict[list(train_dict.keys())[0]])
        )
        model = FusionClsModel(backbones=backbone, classifier=classifier)

        # create folder to save result
        save_folder = f'{args.output_folder}/{args.name}'
        last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
        last_run = max(last_run) + 1 if len(last_run) > 0 else 0
        save_folder = f'{save_folder}/run_{last_run}'

        # create training config
        optimizer = tr.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model_file_path = f'{save_folder}/model.pth'
        flow = SingleTaskFlow(
            model=model, optimizer=optimizer,
            device=args.device,
            callbacks=[
                ModelCheckpoint(MAX_EPOCH, model_file_path, smaller_better=False),
                EarlyStop(EARLY_STOP_PATIENCE, smaller_better=False),
                ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=LR_SCHEDULER_PATIENCE, verbose=True)
            ],
            callback_criterion='f1'
        )

        # train and valid
        augmenter = {
            'waist': Rotation3D(angle_range=30),
            'wrist': Rotation3D(angle_range=30)
        }
        train_set = BalancedFusionDataset(deepcopy(train_dict), augmenters=augmenter)
        valid_set = FusionDataset(deepcopy(valid_dict))
        train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
        train_log, valid_log = flow.run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            max_epochs=MAX_EPOCH, min_epochs=MIN_EPOCH
        )

        # test
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
