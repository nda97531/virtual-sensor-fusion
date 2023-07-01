"""
VSF distributor receives features from backbone models, run additional layers if necessary, and calculate losses
"""

import torch as tr
import torch.nn as nn

from vsf.networks.contrastive_loss import ContrastiveLoss


class OneSetDistributor(nn.Module):
    def __init__(self, input_dims: dict, contrast_feature_dim: int, num_classes: dict,
                 contrastive_loss_func: ContrastiveLoss, main_modal: str = None) -> None:
        """
        Distributor for VSF with a single labelled multi-modal dataset.
        All modals are contrasted together, some of them output class probabilities.

        Args:
            input_dims: dict[modal name]: input feature dimension (int)
            contrast_feature_dim: feature dimension for contrastive loss
            num_classes: dict[modal name]: number of output classes (int);
                all keys if this dict must also be presented in `input_dims`
            main_modal: name of the main modal
        """
        super().__init__()
        # connect FCs, used before contrastive loss
        self.connect_fc = nn.ModuleDict({
            modal: nn.Linear(dim, contrast_feature_dim)
            for modal, dim in input_dims.items()
        })
        # classifiers
        self.classifiers = nn.ModuleDict({
            modal: nn.Linear(input_dims[modal], num_classes[modal])
            for modal in num_classes.keys()
        })
        self.contrastive_loss_func = contrastive_loss_func

        self.main_modal = main_modal

    def forward(self, x_dict: dict, cal_loss: bool = True) -> tuple:
        """
        Run forward pass of the distributor

        Args:
            x_dict: dict[modal]: feature tensor shape [batch size, feature]; batch size of all modals must be the same;
                and samples with the same index in batches are of the same time T.
            cal_loss: whether to calculate contrastive loss. if False, return None instead of float

        Returns:
            a tuple of 2 elements:
                - a dict: dict[modal name] = predicted class probabilities tensor shape [batch, num class]
                - contrastive loss (pytorch float; None if `cal_loss` == False)
        """
        # classification
        class_probs = {
            modal: self.classifiers[modal](x_dict[modal])
            for modal in self.classifiers.keys()
        }

        if cal_loss:
            # features for contrastive loss
            contrast_features = tr.stack([
                self.connect_fc[modal](x_dict[modal]) for modal in self.connect_fc.keys()
            ])
            # calculate contrastive loss
            contrast_loss = self.contrastive_loss_func(contrast_features)
        else:
            contrast_loss = None

        # [batch, channel]
        return class_probs, contrast_loss
