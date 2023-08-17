"""
Exp 1.2
Single task: classification of all labels + VSF contrastive
Sensor fusion: wrist accelerometer + skeleton
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

from vsf.data_generator.augmentation import Rotation3D, HorizontalFlip
from vsf.data_generator.classification_data_gen import FusionDataset, BalancedFusionDataset

from vsf.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from vsf.flow.vsf_flow import VSFFlow
from vsf.networks.backbone_tcn import TCN
from vsf.networks.complete_model import VsfModel
from vsf.networks.vsf_distributor import VsfDistributor
from vsf.public_datasets.up_fall_dataset import UPFallNpyWindow, UPFallConst
from vsf.networks.contrastive_loss import CMCLoss, CocoaLoss, Cocoa2Loss
from vsf.networks.classifier import BasicClassifier
from vsf.networks.complete_model import BasicClsModel


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
    upfall = UPFallNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        min_step_size_sec=min_step_size_sec,
        max_short_window=max_short_window,
        modal_cols={
            UPFallConst.MODAL_INERTIA: {
                'wrist_acc': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)'],
            },
            UPFallConst.MODAL_SKELETON: {
                'skeleton': list(itertools.chain.from_iterable(
                    [f'x_{joint}', f'y_{joint}'] for joint in
                    ['Neck', 'RElbow', 'LElbow', 'RWrist', 'LWrist', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle']
                ))
            }
        }
    )
    df = upfall.run()
    list_sub_modal = list(itertools.chain.from_iterable(list(sub_dict) for sub_dict in upfall.modal_cols.values()))

    # split TRAIN, VALID, TEST
    # 1/3 subjects as test set
    test_set_idx = df['subject'] % 3 == 0
    test_set = df.loc[test_set_idx]
    # trial 3 of 2/3 subjects as valid set
    valid_set_idx = (~test_set_idx) & (df['trial'] == 3)
    valid_set = df.loc[valid_set_idx]
    # trial 1 and 2 of 2/3 subjects as train set
    train_set_idx = ~(test_set_idx | valid_set_idx)
    train_set = df.loc[train_set_idx]

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
            class_dict['wrist_acc'][label_idx] = modal_dict['wrist_acc'][idx]
            class_dict['skeleton'][label_idx] = modal_dict['skeleton'][idx]
        class_dict = dict(class_dict)

        assert list(class_dict.keys()) == list_sub_modal, 'Mismatched submodal list'
        return class_dict

    three_dicts = {
        'train': concat_data_in_df(train_set),
        'valid': concat_data_in_df(valid_set),
        'test': concat_data_in_df(test_set)
    }
    return three_dicts


def get_trained_ske(ske_weight_path: str) -> nn.Module:
    backbone = TCN(
        input_shape=(80, 18),
        how_flatten='spatial attention gap',
        n_tcn_channels=(64,) * 4 + (128,) * 2,
        tcn_drop_rate=0.5,
        use_spatial_dropout=False,
        conv_norm='batch',
        attention_conv_norm=''
    )
    classifier = BasicClassifier(
        n_features_in=128,
        n_classes_out=11
    )
    model = BasicClsModel(backbone=backbone, classifier=classifier, dropout=0.5)

    model.load_state_dict(tr.load(ske_weight_path))
    backbone.requires_grad_(False)

    return backbone


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', default='cuda:0')

    parser.add_argument('--name', '-n', default='exp1_4_freeze',
                        help='name of the experiment to create a folder to save weights')

    parser.add_argument('--data-folder', '-data',
                        default='/mnt/data_partition/UCD/UCD04 - Virtual sensor fusion/processed_parquet/UP-Fall',
                        help='path to parquet data folder')

    parser.add_argument('--output-folder', '-o', default='./log/upfall',
                        help='path to save training logs and model weights')

    parser.add_argument('--ske-weight', '-w',
                        default='/mnt/data_partition/UCD/UCD04 - Virtual sensor fusion/virtual-sensor-fusion/log/model.pth',
                        help='path to pretrained skeleton model weight file')

    args = parser.parse_args()

    NUM_REPEAT = 3
    NUM_EPOCH = 300
    LEARNING_RATE = 1e-2
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
            'wrist_acc': TCN(
                input_shape=(200, 3),
                how_flatten='spatial attention gap',
                n_tcn_channels=(64,) * 5 + (128,) * 2,
                tcn_drop_rate=0.5,
                use_spatial_dropout=False,
                conv_norm='batch',
                attention_conv_norm=''
            ),
            'skeleton': get_trained_ske(args.ske_weight)
        })

        head = VsfDistributor(
            input_dims={modal: 128 for modal in backbone.keys()},
            num_classes={
                'wrist_acc': len(train_dict[list(train_dict.keys())[0]])
            },
            contrastive_loss_func=CocoaLoss(temp=0.1),
            contrast_feature_dim=None,
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
        optimizer = tr.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=0.9)
        model_file_path = f'{save_folder}/model.pth'
        flow = VSFFlow(
            model=model,
            cls_optimizer=optimizer, contrast_optimizer=None,
            device=args.device,
            callbacks=[
                ModelCheckpoint(NUM_EPOCH, model_file_path, smaller_better=False),
                EarlyStop(EARLY_STOP_PATIENCE, smaller_better=False),
                ReduceLROnPlateau(optimizer=optimizer, mode='max', patience=LR_SCHEDULER_PATIENCE, verbose=True)
            ],
            callback_criterion='f1_wrist_acc'
        )

        # train and valid
        augmenter = {
            'wrist_acc': Rotation3D(angle_range=180, p=1, random_seed=None),
            'skeleton': HorizontalFlip(p=0.5, random_seed=None)
        }
        train_set = BalancedFusionDataset(deepcopy(train_dict), augmenters=augmenter)
        valid_set = FusionDataset(deepcopy(valid_dict))
        train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)
        train_log, valid_log = flow.run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=NUM_EPOCH
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
