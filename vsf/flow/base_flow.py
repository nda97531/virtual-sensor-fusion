from abc import ABC
from copy import deepcopy
from typing import List, Union, Tuple, Dict

import pandas as pd
import torch as tr
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from vsf.flow.flow_functions import auto_classification_loss
from vsf.flow.torch_callbacks import TorchCallback, CallbackAction


class BaseFlow(ABC):
    def __init__(self, model: tr.nn.Module,
                 optimizer: tr.optim.Optimizer,
                 device: str,
                 cls_loss_fn: Union[tr.nn.Module, callable, list, str] = 'classification_auto',
                 callbacks: List[TorchCallback] = None,
                 callback_criterion: str = 'loss'):
        """
        Abstract base class for train/valid/test flow

        Args:
            model: model object
            optimizer: optimizer object
            device: hardware device to run, example: 'cpu', 'cuda:0', 'cuda:1'
            cls_loss_fn: classification loss function
            callbacks: list of callback objects
            callback_criterion: criterion to run callback; for example: checkpoint with best 'loss' or 'f1-score', etc.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.cls_loss_fn = auto_classification_loss if cls_loss_fn == 'classification_auto' else cls_loss_fn

        self.train_log = []
        self.valid_log = []

        self.callbacks = callbacks
        self.callback_criterion = callback_criterion

    def run_callbacks(self, epoch: int) -> List[CallbackAction]:
        """
        Run all callback functions

        Args:
            epoch: epoch index

        Returns:
            list of CallbackAction
        """
        actions = []

        for callback in self.callbacks:
            if isinstance(callback, TorchCallback):
                actions.append(callback.on_epoch_end(epoch, self.model,
                                                     self.train_log[-1][self.callback_criterion],
                                                     self.valid_log[-1][self.callback_criterion]))
            elif isinstance(callback, ReduceLROnPlateau):
                callback.step(self.valid_log[-1][self.callback_criterion])
            else:
                raise ValueError(f'Unsupported callback: {type(callback)}')

        return actions

    def tensor_to_device(self, x):
        if isinstance(x, dict):
            x = {k: v.to(self.device) for k, v in x.items()}
        elif isinstance(x, tr.Tensor):
            x = x.to(self.device)
        else:
            raise ValueError(f'Unsupported data type in flow: {type(x)}')
        return x

    def train_epoch(self, dataloader: Union[DataLoader, Dict[str, DataLoader]]) -> None:
        """
        Run a training epoch. This function ensures that the model is switched to training mode

        Args:
            dataloader: DataLoader object or a dict of objects
        """
        self.model = self.model.train()
        training_log = self._train_epoch(dataloader)
        print(f'Train: {training_log}')
        self.train_log.append(training_log)

    def _train_epoch(self, dataloader: Union[DataLoader, Dict[str, DataLoader]]) -> dict:
        """
        DO NOT call this method anywhere else but in the `train_loop` method.
        Run a training epoch.

        Args:
            dataloader: DataLoader object or a dict of objects

        Returns:
            a dict of training log. Example: {'loss': 0.1, 'f1': 0.99, 'lr': 0.001}
        """
        raise NotImplementedError

    def valid_epoch(self, dataloader: Union[DataLoader, Dict[str, DataLoader]]) -> None:
        """
        Run a validation epoch. This function ensures that the model is switched to evaluation mode

        Args:
            dataloader: DataLoader object or a dict of objects
        """
        self.model = self.model.eval()
        with tr.no_grad():
            valid_log = self._valid_epoch(dataloader)
        print(f'Valid: {valid_log}')
        self.valid_log.append(valid_log)

    def _valid_epoch(self, dataloader: Union[DataLoader, Dict[str, DataLoader]]) -> dict:
        """
        DO NOT call this method anywhere else but in the `valid_loop` method.
        Run a validation epoch.

        Args:
            dataloader: DataLoader object or a dict of objects
        """
        raise NotImplementedError

    def _test_epoch(self, dataloader: Union[DataLoader, Dict[str, DataLoader]], model: tr.nn.Module) -> any:
        """
        DO NOT call this method anywhere else but in the `run_test_epoch` method.
        Run a test epoch.

        Args:
            dataloader: DataLoader object or a dict of objects
            model: model used for testing

        Returns:
            any form of test score
        """
        raise NotImplementedError

    def run_test_epoch(self, dataloader: Union[DataLoader, Dict[str, DataLoader]],
                       model_state_dict: dict = None) -> any:
        """
        Run a test epoch. This function ensures that the model is switched to evaluation mode

        Args:
            dataloader: DataLoader object or a dict of objects
            model_state_dict: if provided, load this state dict before testing (without changing `self.model`)

        Returns:
            any form of test score
        """
        if model_state_dict is not None:
            model = deepcopy(self.model)
            model.load_state_dict(model_state_dict)
        else:
            model = self.model
        model = model.eval()

        with tr.no_grad():
            test_result = self._test_epoch(dataloader, model)
        return test_result

    def run(self,
            train_loader: Union[DataLoader, Dict[str, DataLoader]],
            valid_loader: Union[DataLoader, Dict[str, DataLoader]],
            num_epochs: int
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            print(f"-----------------\nEpoch {epoch}/{num_epochs}")

            self.train_epoch(train_loader)
            if valid_loader is not None:
                self.valid_epoch(valid_loader)

            callback_actions = self.run_callbacks(epoch)
            if CallbackAction.STOP_TRAINING in callback_actions:
                break

        train_log = pd.DataFrame(self.train_log)
        valid_log = pd.DataFrame(self.valid_log)
        return train_log, valid_log
