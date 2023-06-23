import pandas as pd
import torch as tr
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics

from vsf.flow.base_flow import BaseFlow
from vsf.flow.flow_functions import f1_score_from_prob


class SingleTaskFlow(BaseFlow):
    def _train_epoch(self, dataloader: DataLoader) -> None:
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
        metric = f1_score_from_prob(tr.concatenate(y_true), tr.concatenate(y_pred))
        self.train_log.append({'loss': train_loss, 'metric': metric})
        print(f'Train: {self.train_log[-1]}')

    def _valid_epoch(self, dataloader: DataLoader) -> None:
        num_batches = len(dataloader)
        valid_loss = 0

        y_true = []
        y_pred = []
        # no need to call tr.no_grad() because the parent class handles it
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
        metric = f1_score_from_prob(y_true, y_pred)

        # record epoch log
        self.valid_log.append({'loss': valid_loss, 'metric': metric})
        print(f'Valid: {self.valid_log[-1]}')

    def _test_epoch(self, dataloader: DataLoader, model: tr.nn.Module) -> pd.DataFrame:
        y_true = []
        y_pred = []
        # no need to call tr.no_grad() because the parent class handles it
        for x, y in dataloader:
            x = x.to(self.device)
            y = y.to(self.device)
            pred = model(x)
            y_true.append(y)
            y_pred.append(pred)

        y_true = tr.concatenate(y_true).to('cpu')
        y_pred = tr.concatenate(y_pred).to('cpu').argmax(axis=-1)
        report = metrics.classification_report(y_true, y_pred, digits=4, output_dict=True)
        report = pd.DataFrame(report)
        return report


if __name__ == '__main__':
    flow = SingleTaskFlow(
        model=None,
        optimizer=None,
        device='cuda:0',
        loss_fn='classification_auto',
        callbacks=None
    )
    flow.run()
