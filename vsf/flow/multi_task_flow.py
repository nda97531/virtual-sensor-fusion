from typing import Union, Tuple

import torch as tr
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsf.flow.base_flow import BaseFlow
from vsf.flow.flow_functions import f1_score_from_prob


class MultiModalLabelledFlow(BaseFlow):
    """
    Flow for VSF with 1 labelled multi-modal dataset
    """

    def _train_epoch(self, dataloader: DataLoader) -> None:
        train_loss = 0
        y_true = []
        y_pred = []

        for batch, (xs, y) in tqdm(enumerate(dataloader), ncols=0):
            # list of modals
            xs = [x.to(self.device) for x in xs]
            y = y.to(self.device)

            # Compute prediction and loss
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record batch log
            train_loss += loss.item()
            y_true.append(y.to('cpu'))
            y_pred.append(pred.to('cpu'))

        # record epoch log
        train_loss /= len(dataloader)
        metric = f1_score_from_prob(tr.concatenate(y_true), tr.concatenate(y_pred))
        self.train_log.append({'loss': train_loss, 'metric': metric})
        print(f'Train: {self.train_log[-1]}')

    def _valid_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        pass

    def _test_epoch(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> any:
        pass
