"""
VSF distributor receives features from backbone models, run additional layers if necessary, and calculate losses
"""
from typing import Union
import torch as tr
import torch.nn as nn
from vsf.loss_functions.contrastive_loss import ContrastiveLoss


class VsfDistributor(nn.Module):
    def __init__(self, input_dims: dict, num_classes: dict,
                 contrastive_loss_func: ContrastiveLoss, cls_dropout: float = 0.5, contrast_feature_dim: int = None,
                 main_modal: str = None) -> None:
        """
        Distributor for VSF with a single labelled multi-modal dataset.
        All modals are contrasted together, some of them output class probabilities.

        Args:
            input_dims: dict[modal name]: input feature dimension (int)
            num_classes: dict[modal name]: number of output classes (int);
                all keys if this dict must also be presented in `input_dims`
            cls_dropout: dropout rate before classifier
            contrast_feature_dim: feature dimension for FC before contrastive loss; default: don't use connect FC
            main_modal: name of the main modal
        """
        super().__init__()
        # connect FCs, used before contrastive loss
        self.connect_fc = nn.ModuleDict({
            modal: nn.Linear(dim, contrast_feature_dim) if contrast_feature_dim else nn.Identity()
            for modal, dim in input_dims.items()
        })
        # classifiers
        self.classifiers = nn.ModuleDict({
            modal: nn.Linear(input_dims[modal], num_classes[modal])
            for modal in num_classes.keys()
        })
        self.contrastive_loss_func = contrastive_loss_func
        self.dropout = nn.Dropout(p=cls_dropout)
        self.main_modal = main_modal

    def forward(self, x_dict: dict,
                cls_mask: Union[dict, str] = 'all', contrast_mask: Union[dict, str] = 'all') -> tuple:
        """
        Run forward pass of the distributor

        Args:
            x_dict: dict, keys are modals, values shape [batch size, feature],
                samples with the same index in batches are of the same time T.
            cls_mask: dict[modal] = indices within a batch of samples used for classification;
                besides Tensor mask, string values [all|none] are acceptable
            contrast_mask: same as `cls_mask` but for contrastive loss

        Returns:
            a tuple of 2 elements:
                - a dict, keys are modal names, values are class logits predicted from that modal,
                    value tensor shape is [batch, num class]
                - contrastive loss (pytorch float; None if `cal_loss` == False)
        """
        # classification
        if cls_mask == 'all':
            cls_mask = {modal: tr.tensor([True] * len(x_dict[modal]))
                        for modal in x_dict.keys()}
        if cls_mask != 'none':
            class_logit = {modal: self.classifiers[modal](self.dropout(x_dict[modal][cls_mask[modal]]))
                           for modal in self.classifiers.keys()}
        else:
            class_logit = None

        # contrastive
        if contrast_mask == 'all':
            contrast_mask = {modal: tr.tensor([True] * len(x_dict[modal]))
                             for modal in x_dict.keys()}
        if contrast_mask != 'none':
            # features for contrastive loss
            contrast_features = tr.stack([self.connect_fc[modal](x_dict[modal][contrast_mask[modal]])
                                          for modal in self.connect_fc.keys()])
            # calculate contrastive loss
            contrast_loss = self.contrastive_loss_func(contrast_features)
        else:
            contrast_loss = None

        # [batch, channel]
        return class_logit, contrast_loss
