"""
Exp 1.1
Single task: classification of all labels
Single sensor: rankle accelerometer
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

from vsf.data_generator.augmentation import Rotation3D
from vsf.data_generator.classification_data_gen import BasicDataset, BalancedDataset
from vsf.flow.single_task_flow import SingleTaskFlow
from vsf.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from vsf.networks.backbone_tcn import TCN
from vsf.networks.classifier import BasicClassifier
from vsf.networks.complete_model import BasicClsModel
from vsf.public_datasets.sfu_imu_dataset import SFUNpyWindow, SFUConst


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
                'rankle_acc': [f'rankle_acc_{axis}(m/s^2)' for axis in ['x', 'y', 'z']]
            }
        }
    )
    df = npy_loader.run()
    list_sub_modal = list(itertools.chain.from_iterable(list(sub_dict) for sub_dict in npy_loader.modal_cols.values()))

    # split TRAIN, VALID, TEST
    assert np.unique(df['subject']).tolist() == list(range(1, 11))
    train_subject = {1}
    test_subject = {3, 5, 7, 9, 10}
    valid_subject = {2, 4, 6, 8}

    train_set_idx = df['subject'].isin(train_subject)
    valid_set_idx = df['subject'].isin(valid_subject)
    assert not (train_set_idx & valid_set_idx).any(), 'Train and valid overlapped'
    test_set_idx = ~(train_set_idx | valid_set_idx)

    # randomizer = np.random.default_rng(99)
    # random_idx = randomizer.permutation(len(df))
    # assert len(random_idx) == 600
    # train_set_idx = random_idx[:250]
    # valid_set_idx = random_idx[250:350]
    # test_set_idx = random_idx[350:600]

    train_set = df.loc[train_set_idx]
    valid_set = df.loc[valid_set_idx]
    test_set = df.loc[test_set_idx]

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', required=True)

    parser.add_argument('--name', '-n', default='exp1_1',
                        help='name of the experiment to create a folder to save weights')

    parser.add_argument('--data-folder', '-data',
                        default='/mnt/data_drive/projects/UCD04 - Virtual sensor fusion/processed_parquet/SFU-IMU',
                        help='path to parquet data folder')

    parser.add_argument('--output-folder', '-o', default='./log/sfu',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    NUM_REPEAT = 3
    NUM_EPOCH = 300
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-2
    EARLY_STOP_PATIENCE = 30
    LR_SCHEDULER_PATIENCE = 15
    TRAIN_BATCH_SIZE = 16

    # load data
    three_dicts = load_data(parquet_dir=args.data_folder)
    train_dict = three_dicts['train']['rankle_acc']
    valid_dict = three_dicts['valid']['rankle_acc']
    test_dict = three_dicts['test']['rankle_acc']
    del three_dicts

    test_scores = []
    model_paths = []
    # train 3 times
    for _ in range(NUM_REPEAT):
        # create model
        backbone = TCN(
            input_shape=(200, 3),
            how_flatten='spatial attention gap',
            n_tcn_channels=(64,) * 5 + (128,) * 2,
            tcn_drop_rate=0.5,
            use_spatial_dropout=False,
            conv_norm='batch',
            attention_conv_norm=''
        )
        classifier = BasicClassifier(
            n_features_in=128,
            n_classes_out=len(train_dict)
        )
        model = BasicClsModel(backbone=backbone, classifier=classifier, dropout=0.5)

        # create folder to save result
        save_folder = f'{args.output_folder}/{args.name}'
        last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
        last_run = max(last_run) + 1 if len(last_run) > 0 else 0
        save_folder = f'{save_folder}/run_{last_run}'

        # create training config
        loss_fn = 'classification_auto'
        optimizer = tr.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        model_file_path = f'{save_folder}/model.pth'
        flow = SingleTaskFlow(
            model=model, loss_fn=loss_fn, optimizer=optimizer,
            device=args.device,
            callbacks=[
                ModelCheckpoint(NUM_EPOCH, model_file_path),
                EarlyStop(EARLY_STOP_PATIENCE),
                ReduceLROnPlateau(optimizer=optimizer, mode='min', patience=LR_SCHEDULER_PATIENCE, verbose=True)
            ]
        )

        # train and valid
        augmenter = Rotation3D(angle_range=180, p=1, random_seed=None)
        train_set = BalancedDataset(deepcopy(train_dict), augmenter=augmenter)
        valid_set = BasicDataset(deepcopy(valid_dict))
        train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
        train_log, valid_log = flow.run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=NUM_EPOCH
        )

        # test
        test_set = BasicDataset(deepcopy(test_dict))
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