from abc import ABC

import pandas as pd
import numpy as np
from typing import List, Union, Tuple
import torch as tr
from tqdm import tqdm
from torch.utils.data import DataLoader

from vsf.flow.flow_functions import auto_classification_loss, f1_score_from_prob
from vsf.flow.torch_callbacks import TorchCallback, CallbackAction


class BaseFlow(ABC):
    def __init__(self, model: tr.nn.Module,
                 optimizer: tr.optim.Optimizer,
                 device: str,
                 loss_fn: Union[tr.nn.Module, callable, list, str] = 'classification_auto',
                 callbacks: List[TorchCallback] = None):
        """
        Abstract base class for train/valid/test flow

        Args:
            model: model object
            optimizer: optimizer object
            device: hardware device to run, example: 'cpu', 'cuda:0', 'cuda:1'
            loss_fn: loss function
            callbacks: list of callback objects
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = auto_classification_loss if loss_fn == 'classification_auto' else loss_fn

        self.train_log = []
        self.valid_log = []

        self.callbacks = callbacks

    def train_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        """
        Run a training epoch. This function ensures that the model is switched to training mode

        Args:
            dataloader: DataLoader object or a list of objects
        """
        self.model = self.model.train()
        self._train_epoch(dataloader)

    def _train_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        """
        DO NOT call this method anywhere else but in the `train_loop` method.
        Run a training epoch.

        Args:
            dataloader: DataLoader object or a list of objects
        """
        raise NotImplementedError

    def valid_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        """
        Run a validation epoch. This function ensures that the model is switched to evaluation mode

        Args:
            dataloader: DataLoader object or a list of objects
        """
        self.model = self.model.eval()
        self._valid_epoch(dataloader)

    def _valid_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        """
        DO NOT call this method anywhere else but in the `valid_loop` method.
        Run a validation epoch.

        Args:
            dataloader: DataLoader object or a list of objects
        """
        raise NotImplementedError

    def run_callbacks(self, epoch: int) -> List[CallbackAction]:
        """
        Run all callback functions

        Args:
            epoch: epoch index

        Returns:

        """
        actions = [
            callback.on_epoch_end(epoch, self.model, self.train_log[-1]['loss'], self.valid_log[-1]['loss'])
            for callback in self.callbacks
        ]
        return actions

    def _test_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> any:
        """
        DO NOT call this method anywhere else but in the `run_test_epoch` method.
        Run a test epoch.

        Args:
            dataloader: DataLoader object or a list of objects

        Returns:
            any form of test score
        """
        raise NotImplementedError

    def run_test_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> any:
        """
        Run a test epoch. This function ensures that the model is switched to evaluation mode

        Args:
            dataloader: DataLoader object or a list of objects

        Returns:
            any form of test score
        """
        self.model = self.model.eval()
        return self._test_epoch(dataloader)

    def run(self, train_loader: Union[DataLoader, List[DataLoader]], valid_loader: Union[DataLoader, List[DataLoader]],
            num_epochs: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Train the model

        Args:
            train_loader: train data loader object
            valid_loader: valid data loader object
            num_epochs: maximum number of epochs

        Returns:
            2 pandas.DataFrame objects containing train and valid log
        """
        assert type(train_loader) is type(valid_loader), 'train_loader and valid_loader must be of the same type'
        if not isinstance(train_loader, DataLoader):
            assert len(train_loader) == len(valid_loader), \
                'number of tasks in train_loader and valid_loader must be the same'

        for epoch in range(1, num_epochs + 1):
            print(f"-----------------\nEpoch {epoch}")

            self.train_epoch(train_loader)
            self.valid_epoch(valid_loader)

            callback_actions = self.run_callbacks(epoch)
            if CallbackAction.STOP_TRAINING in callback_actions:
                break

        train_log = pd.DataFrame(self.train_log)
        valid_log = pd.DataFrame(self.valid_log)
        return train_log, valid_log
