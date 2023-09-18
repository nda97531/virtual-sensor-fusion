from collections import defaultdict
from typing import Dict

import pandas as pd
import torch as tr
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from vsf.flow.base_flow import BaseFlow
from vsf.flow.flow_functions import f1_score_from_prob, ypred_2_categorical


def mix_batch(x_cls: dict, y_cls: tr.Tensor, x_contrast: dict) -> tuple:
    """
    Concatenate data for classification and contrastive learning together; then shuffle them.

    Args:
        x_cls: dict[modal] = data tensor; all modals have the same batch size
        y_cls: classification label tensor, length = x_cls batch size
        x_contrast: dict[modal] = data tensor; all modals have the same batch size

    Returns:
        - dict[modal] = data tensor; new batch size = cls batch size + contrast batch size;
            bs of each modal is different because some modals are not used for all tasks
        - mask cls dict, same format as the dict above
        - mask contrast dict, same format as the dict above
        - classification label tensor
    """
    # batch size of classification and contrastive learning data
    bs_cls = len(y_cls)
    bs_contrast = len(next(iter(x_contrast.values())))

    # create random index to shuffle data and label
    shuffle_idx = tr.randperm(bs_cls + bs_contrast)
    # first half in the index array is for classification
    cls_idx_condition = shuffle_idx < bs_cls
    cls_shuffle_idx = shuffle_idx[cls_idx_condition]
    contrast_shuffle_idx = shuffle_idx[~cls_idx_condition] - bs_cls

    x_dict = {}
    cls_mask = {}
    contrast_mask = {}
    # create mask and shuffle data and mask for each modal
    for modal in (x_cls.keys() | x_contrast.keys()):
        if modal not in x_cls.keys():
            cls_mask[modal] = tr.tensor([False] * bs_contrast)
            x_dict[modal] = x_contrast[modal][contrast_shuffle_idx]

        elif modal not in x_contrast.keys():
            cls_mask[modal] = tr.tensor([True] * bs_cls)
            x_dict[modal] = x_cls[modal][cls_shuffle_idx]

        else:
            # put cls first when concatenating
            cls_mask[modal] = tr.tensor([True] * bs_cls + [False] * bs_contrast)[shuffle_idx]
            x_dict[modal] = tr.cat([x_cls[modal], x_contrast[modal]])[shuffle_idx]

        contrast_mask[modal] = ~cls_mask[modal]
    # shuffle cls label
    y_cls = y_cls[cls_shuffle_idx]
    return x_dict, cls_mask, contrast_mask, y_cls


class VsfE2eFlow(BaseFlow):
    def _train_epoch(self, dataloader: Dict[str, DataLoader]) -> dict:
        num_iter = min(len(dl) for dl in dataloader.values())
        dataloader = {k: iter(dl) for k, dl in dataloader.items()}

        training_log = defaultdict(float)
        y_true = []
        y_preds = defaultdict(list)

        for _ in tqdm(range(num_iter), ncols=0):
            # get batch data
            batch_data = {k: next(dl) for k, dl in dataloader.items()}
            x_cls, y_cls = batch_data['cls']
            x_contrast = batch_data['contrast']
            # send data to gpu
            x_cls = self.tensor_to_device(x_cls)
            y_cls = y_cls.to(self.device)
            x_contrast = self.tensor_to_device(x_contrast)

            # concat x_cls and x_contrast, and shuffle
            x_dict, cls_mask, contrast_mask, y_cls = mix_batch(x_cls, y_cls, x_contrast)

            # compute prediction
            cls_logit_dict, contrast_loss, modal_cls_loss = self.model(
                x_dict,
                head_kwargs={'cls_mask': cls_mask, 'contrast_mask': contrast_mask}
            )
            cls_loss = sum(self.cls_loss_fn(logit, y_cls) for logit in cls_logit_dict.values()) / len(cls_logit_dict)
            loss = cls_loss + contrast_loss + modal_cls_loss

            # optimise
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record batch log
            training_log['cls_loss'] += cls_loss.item()
            training_log['contrast_loss'] += contrast_loss.item()
            training_log['modal_cls_loss'] += modal_cls_loss.item()
            y_true.append(y_cls)
            for modal in cls_logit_dict.keys():
                y_preds[modal].append(cls_logit_dict.get(modal))

        # record epoch log
        # losses
        training_log = {k: v / num_iter for k, v in training_log.items()}
        training_log['loss'] = sum(training_log.values())
        # metric scores
        y_true = tr.concatenate(y_true).to('cpu')
        for modal in y_preds.keys():
            training_log[f'f1_{modal}'] = f1_score_from_prob(y_true, tr.concatenate(y_preds[modal]).to('cpu'))
        # learning rate
        training_log['lr'] = self.optimizer.param_groups[0]['lr']
        return training_log

    def _valid_epoch(self, dataloader: Dict[str, DataLoader]) -> dict:
        valid_log = defaultdict(float)
        y_true = []
        y_preds = defaultdict(list)

        # no need to call tr.no_grad() because the parent class handles it.
        # run cls data
        for x, y in dataloader['cls']:
            x = self.tensor_to_device(x)
            y = y.to(self.device)

            cls_mask = tr.tensor([True] * len(y))
            contrast_mask = tr.tensor([False] * len(y))
            cls_logit_dict, _, _ = self.model(
                x,
                head_kwargs={
                    'cls_mask': {modal: cls_mask for modal in x.keys()},
                    'contrast_mask': {modal: contrast_mask for modal in x.keys()}
                }
            )
            cls_loss = sum(self.cls_loss_fn(logit, y) for logit in cls_logit_dict.values()) / len(cls_logit_dict)

            valid_log['cls_loss'] += cls_loss.item()
            y_true.append(y)
            for modal in cls_logit_dict.keys():
                y_preds[modal].append(cls_logit_dict[modal])

        # run contrast data
        for x in dataloader['contrast']:
            x = self.tensor_to_device(x)

            cls_mask = tr.tensor([False] * len(next(iter(x.values()))))
            contrast_mask = ~cls_mask
            _, contrast_loss, modal_cls_loss = self.model(
                x,
                head_kwargs={
                    'cls_mask': {modal: cls_mask for modal in x.keys()},
                    'contrast_mask': {modal: contrast_mask for modal in x.keys()}
                }
            )
            valid_log['contrast_loss'] += contrast_loss.item()
            valid_log['modal_cls_loss'] += modal_cls_loss.item()

        # record epoch log
        # losses
        valid_log = dict(valid_log)
        valid_log['cls_loss'] /= len(dataloader['cls'])
        valid_log['contrast_loss'] /= len(dataloader['contrast'])
        valid_log['modal_cls_loss'] /= len(dataloader['contrast'])
        valid_log['loss'] = sum(valid_log.values())
        # metric scores
        y_true = tr.concatenate(y_true).to('cpu')
        for modal in y_preds.keys():
            valid_log[f'f1_{modal}'] = f1_score_from_prob(y_true, tr.concatenate(y_preds[modal]).to('cpu'))

        return valid_log

    def _test_epoch(self, dataloader: DataLoader, model: tr.nn.Module) -> pd.DataFrame:
        y_true = []
        y_preds = defaultdict(list)

        # no need to call tr.no_grad() because the parent class handles it
        for x, y in dataloader:
            x = self.tensor_to_device(x)
            y = y.to(self.device)

            cls_mask = tr.tensor([True] * len(y))
            contrast_mask = tr.tensor([False] * len(y))
            cls_logit_dict, _, _ = model(
                x,
                head_kwargs={
                    'cls_mask': {modal: cls_mask for modal in x.keys()},
                    'contrast_mask': {modal: contrast_mask for modal in x.keys()}
                }
            )
            y_true.append(y)
            for modal in cls_logit_dict.keys():
                y_preds[modal].append(cls_logit_dict[modal])

        # calculate score report
        y_true = tr.concatenate(y_true).to('cpu')
        y_preds = {modal: ypred_2_categorical(tr.concatenate(y_preds[modal])).to('cpu')
                   for modal in y_preds.keys()}
        reports = {
            modal: pd.DataFrame(metrics.classification_report(y_true, y_preds[modal], digits=4, output_dict=True))
            for modal in y_preds.keys()
        }
        # concatenate score DFs of all modals
        for modal in reports.keys():
            reports[modal].index = [f'{modal}_{name}' for name in reports[modal].index]
        report = pd.concat(reports.values())
        return report


if __name__ == '__main__':
    flow = VsfE2eFlow(
        model=None,
        optimizer=None,
        device='cuda:0',
        callbacks=None
    )
    flow.run()
