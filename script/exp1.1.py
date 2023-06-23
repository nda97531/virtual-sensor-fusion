import os
from typing import Tuple, Dict
import torch
from copy import deepcopy
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from public_datasets.up_fall_dataset import UPFallNpyWindow, UPFallConst
from vsf.data_generator.augment import Augmenter, ComposeAugmenters, TimeWarp, Rotation3D
from vsf.data_generator.classification_data_gen import ClassificationDataset, BalancedClassificationDataset
from vsf.flow.single_task_flow import SingleTaskFlow
from vsf.flow.torch_callbacks import ModelCheckpoint, EarlyStop
from vsf.networks.complete_model import CompleteModel
from vsf.networks.backbone_tcn import TCN
from vsf.networks.classifier import FCClassifier


def load_data(parquet_dir: str, window_size_sec=4, step_size_sec=2, min_step_size_sec=0.5,
              max_short_window=5) -> Tuple[Dict, ...]:
    """
    Load all the UP-Fall dataset into a dataframe

    Args:
        parquet_dir: path to processed parquet folder
        window_size_sec: window size in second for sliding window
        step_size_sec: step size in second for sliding window
        min_step_size_sec: min step size in second for shifting window
        max_short_window: number of window for each session of short activities

    Returns:
        a tuple of 3 dicts, each one:
            - key: label (int)
            - value: data array shape [n, ..., channel] or label array shape []
    """
    upfall = UPFallNpyWindow(
        parquet_root_dir=parquet_dir,
        window_size_sec=window_size_sec,
        step_size_sec=step_size_sec,
        min_step_size_sec=min_step_size_sec,
        max_short_window=max_short_window,
        modal_cols={
            UPFallConst.MODAL_INERTIA: {
                'wrist_acc': ['wrist_acc_x(m/s^2)', 'wrist_acc_y(m/s^2)', 'wrist_acc_z(m/s^2)']
            }
        }
    )
    df = upfall.run()

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
        class_dict = {}
        for label_idx, label_val in enumerate(label_list):
            idx = modal_dict['label'] == label_val
            class_dict[label_idx] = modal_dict['wrist_acc'][idx]

        return class_dict

    test_set = concat_data_in_df(test_set)
    valid_set = concat_data_in_df(valid_set)
    train_set = concat_data_in_df(train_set)
    return train_set, valid_set, test_set


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, required=False)

    parser.add_argument('--name', '-n', default='exp1.1',
                        help='name of the experiment to create a folder to save weights')

    parser.add_argument('--data-folder', '-data',
                        default='/mnt/data_drive/projects/UCD04 - Virtual sensor fusion/processed_parquet/UP-Fall',
                        help='path to parquet data folder')

    parser.add_argument('--output-folder', '-o', default='./log',
                        help='path to save training logs and model weights')
    args = parser.parse_args()

    # load data
    train_dict, valid_dict, test_dict = load_data(parquet_dir=args.data_folder)

    # train 3 times
    for _ in range(3):
        # create data loaders
        augmenter = Rotation3D(angle_range=180, p=1)

        train_set = BalancedClassificationDataset(deepcopy(train_dict), augmenter=augmenter)
        valid_set = ClassificationDataset(deepcopy(valid_dict))
        train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=64, shuffle=False)

        # create model
        backbone = TCN(
            input_shape=(200, 3),
            how_flatten='spatial attention gap',
            n_tcn_channels=(64,) * 5 + (128,) * 2,
            tcn_drop_rate=0.5,
            use_spatial_dropout=False,
            conv_norm='batch',
            attention_conv_norm='batch'
        )
        classifier = FCClassifier(
            n_features=128,
            n_classes=1
        )
        model = CompleteModel(backbone=backbone, classifier=classifier, dropout=0.5)

        # create folder to save result
        save_folder = f'{args.output_folder}/{args.name}'
        last_run = [int(exp_no.split(os.sep)[-1].split('_')[-1]) for exp_no in glob(f'{save_folder}/run_*')]
        last_run = max(last_run) + 1 if len(last_run) > 0 else 0
        save_folder = f'{save_folder}/run_{last_run}'

        # create training config
        loss_fn = 'classification_auto'
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        num_epochs = 40

        flow = SingleTaskFlow(
            model=model, loss_fn=loss_fn, optimizer=optimizer,
            device=args.device,
            callbacks=[
                ModelCheckpoint(num_epochs, f'{save_folder}/single_task.pth'),
                # EarlyStop(10)
            ]
        )

        train_log, valid_log = flow.run(
            train_loader=train_loader,
            valid_loader=valid_loader,
            num_epochs=num_epochs
        )

        train_log.to_csv(f'{save_folder}/train.csv', index=False)
        valid_log.to_csv(f'{save_folder}/valid.csv', index=False)

        print("Done!")
