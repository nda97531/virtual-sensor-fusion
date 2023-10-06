import os
from enum import Enum

import numpy as np
import torch as tr


class CallbackAction(Enum):
    NONE = None
    STOP_TRAINING = 'stop_training'


class TorchCallback:
    def on_epoch_end(self, epoch: int, model: tr.nn.Module, train_metric: float, valid_metric: float) -> CallbackAction:
        raise NotImplementedError()


class ModelCheckpoint(TorchCallback):
    def __init__(self, num_epochs: int, save_path: str, smaller_better: bool = True, save_best_only=True,
                 save_weights_only=True, queue_length: int = 1):
        assert queue_length >= 1
        self.num_epochs = num_epochs
        self.save_path = save_path
        self.smaller_better = smaller_better
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.current_best_result = float('inf') if smaller_better else -float('inf')
        self.current_best_epoch = -1
        self.latest_results = [float('inf') if smaller_better else -float('inf')] * queue_length

        os.makedirs(os.path.split(self.save_path)[0], exist_ok=True)

    def is_better(self, new_result: float):
        if self.smaller_better:
            return new_result < self.current_best_result
        else:
            return new_result > self.current_best_result
        
    def update_latest_result(self, new_result: float):
        self.latest_results.pop(0)
        self.latest_results.append(new_result)
        return np.median(self.latest_results)

    def on_epoch_end(self, epoch: int, model: tr.nn.Module, train_metric: float, valid_metric: float) -> CallbackAction:
        assert epoch != 0, 'Epoch starts at 1'

        new_result = self.update_latest_result(valid_metric)
        save_path = self.save_path.format(new_result)

        # if save every epoch
        if not self.save_best_only:
            tr.save(model.state_dict() if self.save_weights_only else model, save_path)
            print(f"Save model to {save_path}.")

        # if only save improved model
        elif self.is_better(new_result):
            tr.save(model.state_dict() if self.save_weights_only else model, save_path)
            print(f"Model improved from {self.current_best_result} to {new_result}. "
                  f"Save model to {save_path}.")
            self.current_best_result = new_result
            self.current_best_epoch = epoch

        else:
            print(f'Not improved from {self.current_best_result} at epoch {self.current_best_epoch}')

        # save last epoch
        if self.num_epochs == epoch:
            save_path, extension = os.path.splitext(save_path)
            save_path = f'{save_path}_last_epoch{extension}'
            tr.save(model.state_dict() if self.save_weights_only else model, save_path)
            print(f"Save last epoch to {save_path}.")

        return CallbackAction.NONE


class EarlyStop(ModelCheckpoint):
    def __init__(self, patience: int, smaller_better: bool = True, queue_length: int = 1):
        assert queue_length >= 1
        self.patience = patience
        self.smaller_better = smaller_better
        self.epoch_without_improvements = 0
        self.current_best_result = float('inf') if smaller_better else -float('inf')
        self.latest_results = [float('inf') if smaller_better else -float('inf')] * queue_length

    def on_epoch_end(self, epoch: int, model: tr.nn.Module, train_metric: float, valid_metric: float):
        new_result = self.update_latest_result(valid_metric)

        if self.is_better(new_result):
            self.current_best_result = new_result
            self.epoch_without_improvements = 0
        else:
            self.epoch_without_improvements += 1

        if self.epoch_without_improvements >= self.patience:
            print(f"Model does not improve from {self.current_best_result}. Stopping")
            return CallbackAction.STOP_TRAINING
        return CallbackAction.NONE
