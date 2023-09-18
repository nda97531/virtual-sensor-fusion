"""
VSF distributor receives features from backbone models, run additional layers if necessary, and calculate losses
"""
from typing import Union
import torch as tr
import torch.nn as nn
from vsf.loss_functions.contrastive_loss import ContrastiveLoss


class VsfDistributor(nn.Module):
    def __init__(self, input_dims: dict, num_classes: dict,
                 contrastive_loss_func: ContrastiveLoss, cls_dropout: float = 0.5) -> None:
        """
        Distributor for VSF with a single labelled multi-modal dataset.
        All modals are contrasted together, some of them output class probabilities.

        Args:
            input_dims: dict[modal name]: input feature dimension (int); dict order influences contrastive loss modal order
            num_classes: dict[modal name]: number of output classes (int);
                all keys of this dict must also be presented in `input_dims`;
                dict order influences class logit dict output order
            cls_dropout: dropout rate before classifier
        """
        super().__init__()

        # classifiers
        self.classifiers = nn.ModuleDict({
            modal: nn.Linear(input_dims[modal], num_classes[modal])
            for modal in num_classes.keys()
        })
        self.input_dims = input_dims
        self.contrastive_loss_func = contrastive_loss_func
        self.dropout = nn.Dropout(p=cls_dropout)

    def forward(self, x_dict: dict, cls_mask: dict = None, contrast_mask: dict = None) -> tuple:
        """
        Run forward pass of the distributor

        Args:
            x_dict: dict, keys are modals, values shape [batch size, feature],
                samples with the same index in batches are of the same time T.
            cls_mask: dict[modal] = indices within a batch of samples used for classification
            contrast_mask: same as `cls_mask` but for contrastive loss

        Returns:
            a tuple of 2 elements:
                - a dict, keys are modal names, values are class logits predicted from that modal,
                    value tensor shape is [batch, num class]
                - contrastive loss (pytorch float; None if `cal_loss` == False)
        """
        # classification
        class_logit = {
            modal: self.classifiers[modal](self.dropout(x_dict[modal][cls_mask[modal]]))
            for modal in self.classifiers.keys()
            if modal in x_dict
        }

        # features for contrastive loss
        contrast_features = [
            x_dict[modal][contrast_mask[modal]]
            for modal in self.input_dims.keys()
            if modal in x_dict and contrast_mask[modal].any()
        ]
        if len(contrast_features):
            contrast_features = tr.stack(contrast_features)
            # calculate contrastive loss
            contrast_loss = self.contrastive_loss_func(contrast_features)
        else:
            contrast_loss = None

        # [batch, channel]
        return class_logit, contrast_loss
