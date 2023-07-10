from collections import defaultdict
import pandas as pd
import torch as tr
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsf.flow.base_flow import BaseFlow
from vsf.flow.flow_functions import f1_score_from_prob


class VSFFlow(BaseFlow):
    def _train_epoch(self, dataloader: DataLoader) -> dict:
        train_cls_loss = 0
        train_contrast_loss = 0
        y_true = []
        y_preds = defaultdict(list)

        for batch, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader), ncols=0):
            x = self.tensor_to_device(x)
            y = y.to(self.device)

            # Compute prediction and losses
            class_logits_dict, contrast_loss = self.model(x)
            cls_loss = sum(self.loss_fn(class_logits, y) for class_logits in class_logits_dict.values())

            # add all losses together
            loss = cls_loss + contrast_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record batch log
            train_cls_loss += cls_loss.item()
            train_contrast_loss += contrast_loss.item()
            y_true.append(y)
            for modal in class_logits_dict.keys():
                y_preds[modal].append(class_logits_dict.get(modal))

        # record epoch log
        train_cls_loss /= len(dataloader)
        train_contrast_loss /= len(dataloader)
        y_true = tr.concatenate(y_true).to('cpu')
        scores = {
            f'metric_{modal}': f1_score_from_prob(y_true, tr.concatenate(y_preds[modal]).to('cpu'))
            for modal in y_preds.keys()
        }

        training_log = {'loss': train_cls_loss + train_contrast_loss,
                        'cls loss': train_cls_loss, 'contrastive loss': train_contrast_loss}
        training_log.update(scores)
        training_log['lr'] = self.optimizer.param_groups[0]['lr']
        return training_log

    def _valid_epoch(self, dataloader: DataLoader) -> dict:
        valid_cls_loss = 0
        valid_contrast_loss = 0

        y_true = []
        y_preds = defaultdict(list)
        # no need to call tr.no_grad() because the parent class handles it
        for x, y in dataloader:
            x = self.tensor_to_device(x)
            y = y.to(self.device)
            class_logits_dict, contrast_loss = self.model(x)
            cls_loss = sum(self.loss_fn(class_logits, y) for class_logits in class_logits_dict.values())

            valid_cls_loss += cls_loss.item()
            valid_contrast_loss += contrast_loss.item()
            y_true.append(y)
            for modal in class_logits_dict.keys():
                y_preds[modal].append(class_logits_dict.get(modal))

        # record epoch log
        valid_cls_loss /= len(dataloader)
        valid_contrast_loss /= len(dataloader)
        y_true = tr.concatenate(y_true).to('cpu')
        scores = {
            f'metric_{modal}': f1_score_from_prob(y_true, tr.concatenate(y_preds[modal]).to('cpu'))
            for modal in y_preds.keys()
        }

        valid_log = {'loss': valid_cls_loss + valid_contrast_loss,
                     'cls loss': valid_cls_loss, 'contrastive loss': valid_contrast_loss}
        valid_log.update(scores)
        return valid_log

    def _test_epoch(self, dataloader: DataLoader, model: tr.nn.Module) -> pd.DataFrame:
        y_true = []
        y_preds = defaultdict(list)
        # no need to call tr.no_grad() because the parent class handles it
        for x, y in dataloader:
            x = self.tensor_to_device(x)
            y = y.to(self.device)
            class_logits_dict, contrast_loss = model(x)
            y_true.append(y)
            for modal in class_logits_dict.keys():
                y_preds[modal].append(class_logits_dict.get(modal))

        y_true = tr.concatenate(y_true).to('cpu')
        y_preds = {modal: tr.concatenate(y_preds[modal]).argmax(dim=-1).to('cpu') for modal in y_preds.keys()}
        reports = {
            modal: pd.DataFrame(metrics.classification_report(y_true, y_preds[modal], digits=4, output_dict=True))
            for modal in y_preds.keys()
        }
        # concatenate all score DFs
        for modal in reports.keys():
            reports[modal].index = [f'{modal}_{name}' for name in reports[modal].index]
        report = pd.concat(reports)
        return report


if __name__ == '__main__':
    flow = VSFFlow(
        model=None,
        optimizer=None,
        device='cuda:0',
        loss_fn='classification_auto',
        callbacks=None
    )
    flow.run()
