from typing import Union, Tuple
import torch as tr
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics

from vsf.flow.base_flow import BaseFlow
from vsf.flow.flow_functions import f1_score_from_prob


class MultiTaskFlow(BaseFlow):
    def _train_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        pass

    def _valid_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        pass

    def _test_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> any:
        pass
