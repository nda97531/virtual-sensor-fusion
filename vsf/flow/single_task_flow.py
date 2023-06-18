from typing import Union, Tuple
import torch as tr
from tqdm import tqdm
from torch.utils.data import DataLoader

from vsf.flow.base_flow import BaseFlow
from vsf.flow.flow_functions import cal_f1_score


class SingleTaskFlow(BaseFlow):
    def _train_loop(self, dataloader: Union[DataLoader, Tuple[DataLoader]]):
        train_loss = 0
        y_true = []
        y_pred = []

        for batch, (x, y) in tqdm(enumerate(dataloader), ncols=0):
            x = x.to(self.device)
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
        metric = cal_f1_score(tr.concatenate(y_true), tr.concatenate(y_pred))
        self.train_log.append({'loss': train_loss, 'metric': metric})
        print(f'Train: {self.train_log[-1]}')

    def _valid_loop(self, dataloader: Union[DataLoader, Tuple[DataLoader]]) -> None:
        num_batches = len(dataloader)
        valid_loss = 0

        y_true = []
        y_pred = []
        with tr.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                valid_loss += self.loss_fn(pred, y).item()
                y_true.append(y)
                y_pred.append(pred)

        valid_loss /= num_batches
        y_true = tr.concatenate(y_true).to('cpu')
        y_pred = tr.concatenate(y_pred).to('cpu')
        metric = cal_f1_score(y_true, y_pred)

        # record epoch log
        self.valid_log.append({'loss': valid_loss, 'metric': metric})
        print(f'Valid: {self.valid_log[-1]}')


if __name__ == '__main__':
    flow = SingleTaskFlow(
        model=None,
        optimizer=None,
        device='cuda:0',
        loss_fn='classification_auto',
        callbacks=None
    )
    flow.run()
