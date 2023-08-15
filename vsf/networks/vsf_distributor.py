"""
VSF distributor receives features from backbone models, run additional layers if necessary, and calculate losses
"""

import torch as tr
import torch.nn as nn

from vsf.networks.contrastive_loss import ContrastiveLoss
from vsf.tensor_dict import TensorDict


class OneSetDistributor(nn.Module):
    def __init__(self, input_dims: dict, num_classes: dict,
                 contrastive_loss_func: ContrastiveLoss, contrast_feature_dim: int = None,
                 main_modal: str = None) -> None:
        """
        Distributor for VSF with a single labelled multi-modal dataset.
        All modals are contrasted together, some of them output class probabilities.

        Args:
            input_dims: dict[modal name]: input feature dimension (int)
            num_classes: dict[modal name]: number of output classes (int);
                all keys if this dict must also be presented in `input_dims`
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

        self.main_modal = main_modal

    def forward(self, x_dict: TensorDict, cls_mask: tr.Tensor = None, contrast_mask: tr.Tensor = None,
                cal_contrast_loss: bool = True) -> tuple:
        """
        Run forward pass of the distributor

        Args:
            x_dict: TensorDict, values shape [modal, batch size, feature], keys are modals
                samples with the same index in batches are of the same time T.
            cls_mask: indices within a batch of samples used for classification
            contrast_mask: indices within a batch of samples used for contrastive loss
            cal_contrast_loss: whether to calculate contrastive loss. if False, return None instead of float

        Returns:
            a tuple of 2 elements:
                - a TensorDict, keys are modal names, values are class logits predicted from that modal,
                    value tensor shape is [batch, num class]
                - contrastive loss (pytorch float; None if `cal_loss` == False)
        """
        if cls_mask is None:
            cls_mask = tr.arange(len(x_dict.values()[0]))
        if contrast_mask is None:
            contrast_mask = tr.arange(len(x_dict.values()[0]))

        # classification (use tuple instead of TensorDict because each modal may have a different number of classes,
        # so tensors cannot be stacked)
        class_logits = TensorDict(
            x=tuple(self.classifiers[modal](x_dict.get(modal)[cls_mask]) for modal in self.classifiers.keys()),
            keys=list(self.classifiers.keys())
        )

        if cal_contrast_loss:
            # features for contrastive loss
            contrast_features = tr.stack([
                self.connect_fc[modal](x_dict.get(modal)[contrast_mask])
                for modal in self.connect_fc.keys()
            ])
            # calculate contrastive loss
            contrast_loss = self.contrastive_loss_func(contrast_features)
        else:
            contrast_loss = None

        # [batch, channel]
        return class_logits, contrast_loss
