from loguru import logger
from torch.utils.data import Dataset


class UnlabelledFusionDataset(Dataset):
    def __init__(self, data_dict: dict, augmenters: dict = None, float_precision: str = 'float32'):
        """
        Generator for unlabelled data from multiple input streams

        Args:
            data_dict: dict[modal name] = windows array shape [n, ..., channel];
                window order of all modals are the same (for fusion purpose).
            augmenters: dict of Augmenter objects: dict[modal name] = Augmenter object
            float_precision: convert data array into this data type, default is 'float32'
        """
        self.validate_params(data_dict, augmenters)
        logger.info(f'Total number of windows: {len(next(iter(data_dict.values())))}')

        self.augmenters = augmenters
        self.float_precision = float_precision
        self.data = data_dict

    @staticmethod
    def validate_params(label_data_dict: dict, augmenter: dict = None):
        if augmenter is None:
            logger.info('No Augmenter provided')

        # validate params
        for modal_name, modal_dict in label_data_dict.items():
            # validate modal names of data and augmenter dicts
            if (augmenter is not None) and ((modal_name not in augmenter) or (augmenter[modal_name] is None)):
                logger.info(f'No Augmenter provided for {modal_name}')
                augmenter[modal_name] = None

        # check label count of all modals
        count_check = None
        for modal_name in label_data_dict.keys():
            modal_count = len(label_data_dict[modal_name])
            if count_check is None:
                count_check = modal_count
            else:
                assert count_check == modal_count, 'All modals must have the same data size'

    def __getitem__(self, index):
        # get data
        data = {modal: arr[index] for modal, arr in self.data.items()}
        if self.augmenters is not None:
            for modal, windows in data.items():
                if self.augmenters[modal] is not None:
                    data[modal] = self.augmenters[modal].run(windows)
        # convert dtype
        data = {k: v.astype(self.float_precision) for k, v in data.items()}
        return data

    def __len__(self) -> int:
        return len(next(iter(self.data.values())))
