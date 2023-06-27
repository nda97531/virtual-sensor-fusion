import numpy as np
import torch as tr
from loguru import logger
from torch.utils.data import Dataset

from vsf.data_generator.augmentation import Augmenter


def check_label_key(label_data_dict: dict) -> list:
    """
    Check if label values are valid

    Args:
        label_data_dict: a dict, key: label (int), value: data array shape [n, ..., channel]

    Returns:
        list of sorted label keys
    """
    labels = sorted(list(label_data_dict.keys()))
    sorted_idx = list(range(len(label_data_dict)))
    assert labels == sorted_idx, f'Invalid label values: {labels}'
    return labels


class BasicDataset(Dataset):
    def __init__(self, label_data_dict: dict, augmenter: Augmenter = None, float_precision: str = 'float32'):
        """
        Basic data generator for classification task

        Args:
            label_data_dict: key: label (int), value: data array shape [n, ..., channel]
            augmenter: Augmenter object
            float_precision: convert data array into this data type, default is 'float32'
        """
        check_label_key(label_data_dict)
        if augmenter is None:
            logger.info('No Augmenter provided')

        logger.info('Label distribution:\n' + '\n'.join(f'{k}: {len(v)}' for k, v in label_data_dict.items()))

        self.augmenter = augmenter
        self.float_precision = float_precision

        self.data = []
        self.label = []
        for label, data in label_data_dict.items():
            self.data.append(data)
            self.label.append([label] * len(data))

        self.data = np.concatenate(self.data)
        self.label = np.concatenate(self.label)

        self.label = self.label.astype(np.int64)

    def __getitem__(self, index):
        data = self.data[index]
        if self.augmenter is not None:
            data = self.augmenter.run(data)
        label = self.label[index]
        return data.astype(self.float_precision), label

    def __len__(self) -> int:
        return len(self.label)


class BalancedDataset(Dataset):
    def __init__(self, label_data_dict: dict, augmenter: Augmenter = None, shuffle: bool = True,
                 float_precision: str = 'float32'):
        """
        Balanced data generator for classification task.
        It resamples so that all classes are sampled with the same probability.

        Args:
            label_data_dict: key: label (int), value: data array shape [n, ..., channel]
            augmenter: Augmenter object
            shuffle: shuffle data after each epoch
            float_precision: convert data array into this data type, default is 'float32'
        """
        check_label_key(label_data_dict)
        if augmenter is None:
            logger.info('No Augmenter provided')
        logger.info('Label distribution:\n' + '\n'.join(f'{k}: {len(v)}' for k, v in label_data_dict.items()))

        self.shuffle = shuffle
        self.augmenter = augmenter
        self.float_precision = float_precision

        # key: label (int); value: data [n, ..., channel]
        self.label_data_dict = label_data_dict
        # key: label; value: index of the last called instance
        self.label_pick_idx = {}

        for cls, arr in self.label_data_dict.items():
            self.label_pick_idx[cls] = 0

        # calculate dataset size
        self.dataset_len = sum(len(arr) for arr in self.label_data_dict.values())
        self.mean_class_len = self.dataset_len / len(self.label_data_dict)

    def __getitem__(self, index):
        label = int(index // self.mean_class_len)
        data = self.label_data_dict[label][self.label_pick_idx[label]]
        if self.augmenter is not None:
            data = self.augmenter.run(data)

        # update index
        self.label_pick_idx[label] += 1
        # if reach epoch end of this class
        if self.label_pick_idx[label] == len(self.label_data_dict[label]):
            self.label_pick_idx[label] = 0
            self._shuffle_class_index(label)

        return data.astype(self.float_precision), label

    def _shuffle_class_index(self, cls: int):
        if self.shuffle:
            self.label_data_dict[cls] = self.label_data_dict[cls][
                tr.randperm(len(self.label_data_dict[cls]))
            ]

    def __len__(self) -> int:
        return self.dataset_len


class FusionDataset(Dataset):
    def __init__(self, label_data_dict: dict, augmenters: dict = None, float_precision: str = 'float32'):
        """
        Basic data generator for classification task from multiple input streams

        Args:
            label_data_dict: 2-level dict: dict[modal name][label index] = windows array shape [n, ..., channel];
                window order of all modals are the same (for fusion purpose).
            augmenters: dict of Augmenter objects: dict[modal name] = Augmenter object
            float_precision: convert data array into this data type, default is 'float32'
        """
        self.validate_params(label_data_dict, augmenters)

        self.augmenters = augmenters
        self.float_precision = float_precision

        # concatenate data/label of all labels into `self.data` and `self.label`
        self.data = {}
        for modal_name, modal_data in label_data_dict.items():
            self.data[modal_name] = np.concatenate(list(label_data_dict[modal_name].values()))
        self.label = np.concatenate(
            [[label] * len(data) for label, data in label_data_dict[modal_name].items()],
            dtype=np.int64
        )

    @staticmethod
    def validate_params(label_data_dict: dict, augmenter: dict = None):
        if augmenter is None:
            logger.info('No Augmenter provided')

        # validate params
        for modal_name, modal_dict in label_data_dict.items():
            # validate class label values (start from 0 to n-1)
            labels = check_label_key(modal_dict)

            # validate modal names of data and augmenter dicts
            if (augmenter is not None) and (modal_name not in augmenter):
                logger.info(f'No Augmenter provided for {modal_name}')

        # check label count of all modals
        count_check = None
        for modal_name in label_data_dict.keys():
            modal_count = {lbl: len(label_data_dict[modal_name][lbl]) for lbl in labels}
            if count_check is None:
                count_check = modal_count
            else:
                assert count_check == modal_count, 'All modals must have the same data and label'

        first_modal = list(label_data_dict.keys())[0]
        logger.info('Label distribution:\n' +
                    '\n'.join(f'{k}: {len(v)}' for k, v in label_data_dict[first_modal].items()))

    def __getitem__(self, index):
        # get data
        data = {modal: arr[index] for modal, arr in self.data.items()}
        if self.augmenters is not None:
            for modal, window in data.items():
                if self.augmenters[modal] is not None:
                    data[modal] = self.augmenters[modal].run(window)
        # get label
        label = self.label[index]
        # convert dtype
        data = {k: v.astype(self.float_precision) for k, v in data.items()}
        return data, label

    def __len__(self) -> int:
        return len(self.label)


class BalancedFusionDataset(Dataset):
    def __init__(self, label_data_dict: dict, augmenters: dict = None, shuffle: bool = True,
                 float_precision: str = 'float32'):
        """
        Balanced data generator for classification task from multiple input streams.
        It resamples so that all classes are sampled with the same probability.

        Args:
            label_data_dict: 2-level dict: dict[modal name][label index] = windows array shape [n, ..., channel];
                window order of all modals are the same (for fusion purpose).
            augmenters: dict of Augmenter objects: dict[modal name] = Augmenter object
            shuffle: shuffle data after each epoch
            float_precision: convert data array into this data type, default is 'float32'
        """
        FusionDataset.validate_params(label_data_dict, augmenters)

        self.augmenters = augmenters
        self.shuffle = shuffle
        self.float_precision = float_precision

        # dict[modal][label] = windows array shape [N, ..., channel]
        self.label_data_dict = label_data_dict
        # dict[label] = index of the last called instance (0 -> N-1)
        self.label_pick_idx = {}

        self.first_modal = list(label_data_dict.keys())[0]
        for cls, arr in self.label_data_dict[self.first_modal].items():
            self.label_pick_idx[cls] = 0

        # calculate dataset size
        self.dataset_len = sum(len(arr) for arr in self.label_data_dict[self.first_modal].values())
        self.mean_class_len = self.dataset_len / len(self.label_data_dict[self.first_modal])

    def __getitem__(self, index):
        # get label and data
        label = int(index // self.mean_class_len)
        data = {modal: modal_data[label][self.label_pick_idx[label]]
                for modal, modal_data in self.label_data_dict.items()}
        # augment
        if self.augmenters is not None:
            for modal, window in data.items():
                if self.augmenters[modal] is not None:
                    data[modal] = self.augmenters[modal].run(window)
        # update class pick index
        self.label_pick_idx[label] += 1
        # if reach epoch end of this class
        if self.label_pick_idx[label] == len(self.label_data_dict[self.first_modal][label]):
            self.label_pick_idx[label] = 0
            self._shuffle_class_index(label)
        # convert dtype
        data = {k: v.astype(self.float_precision) for k, v in data.items()}
        return data, label

    def _shuffle_class_index(self, cls: int):
        if self.shuffle:
            new_idx = tr.randperm(len(self.label_data_dict[self.first_modal][cls]))
            for modal in self.label_data_dict.keys():
                self.label_data_dict[modal][cls] = self.label_data_dict[modal][cls][new_idx]

    def __len__(self) -> int:
        return self.dataset_len


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = BalancedDataset({
        0: np.arange(1) + 0,
        1: np.arange(2) + 100,
        2: np.arange(5) + 200,
        3: np.arange(10) + 300,
    }, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for epoch in range(10):
        print(f'\nepoch {epoch}')
        iter_loader = iter(dataloader)
        len_loader = len(dataloader) - 1
        for batch in range(len_loader):
            _data, _label = next(iter_loader)
            print('batch: {}; data: {}; label: {}'.format(batch, _data, _label))
