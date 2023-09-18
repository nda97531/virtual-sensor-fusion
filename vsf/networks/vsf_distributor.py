"""
VSF distributor receives features from backbone models, run additional layers if necessary, and calculate losses
"""
import torch as tr
import torch.nn as nn
import torch.nn.functional as F
from vsf.layers.gradient_reversal import GradientReversal
from vsf.loss_functions.contrastive_loss import ContrastiveLoss


class VsfDistributor(nn.Module):
    def __init__(self, input_dims: dict, num_classes: dict, contrastive_loss_func: ContrastiveLoss,
                 modal_cls_reversal_lambda: float = None, cls_dropout: float = 0.5) -> None:
        """
        Distributor for VSF with a single labelled multi-modal dataset.
        All modals are contrasted together, some of them output class probabilities.

        Args:
            input_dims: dict[modal name]: input feature dimension (int);
                dict order influences contrastive loss modal order
            num_classes: dict[modal name]: number of output classes (int);
                all keys of this dict must also be presented in `input_dims`;
                dict order influences class logit dict output order
            modal_cls_reversal_lambda: lambda hyper-param for Gradient Reversal Layer, default: don't use GRL
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

        self.modal_classifier = None
        if modal_cls_reversal_lambda is not None:
            assert len(set(input_dims.values())) == 1, \
                ('If use GRL and/or contrastive loss, '
                 f'all modals must have the same feature space size. Found: {input_dims}')
            feature_dim = next(iter(input_dims.values()))

            self.modal_classifier = nn.Sequential(
                GradientReversal(modal_cls_reversal_lambda),
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, len(input_dims)),
                GradientReversal(modal_cls_reversal_lambda),
            )

    def forward(self, x_dict: dict, cls_mask: dict = None, contrast_mask: dict = None) -> tuple:
        """
        Run forward pass of the distributor

        Args:
            x_dict: dict, keys are modals, values shape [batch size, feature],
                samples with the same index in batches are of the same time T.
            cls_mask: dict[modal] = indices within a batch of samples used for classification
            contrast_mask: same as `cls_mask` but for contrastive loss

        Returns:
            a tuple of 3 elements:
                - a dict, keys are modal names, values are class logits predicted from that modal,
                    value tensor shape is [batch, num class]
                - contrastive loss (pytorch float; None if contrast_mask results in empty data)
                - minus modal classification loss (same as contrastive loss)
        """
        # classification
        class_logit = {
            modal: self.classifiers[modal](self.dropout(x_dict[modal][cls_mask[modal]]))
            for modal in self.classifiers.keys()
            if modal in x_dict
        }

        # features for contrastive loss
        contrast_features = {
            modal_idx: x_dict[modal][contrast_mask[modal]]
            for modal_idx, modal in enumerate(self.input_dims.keys())
            if modal in x_dict and contrast_mask[modal].any()
        }
        if len(contrast_features):
            modal_indices = next(iter(contrast_features.values())).new_tensor(
                list(contrast_features.keys()), dtype=tr.long)
            # shape [modal, batch size, feature]
            contrast_features = tr.stack(list(contrast_features.values()))
            # calculate contrastive loss
            contrast_loss = self.contrastive_loss_func(contrast_features)
        else:
            contrast_loss = None

        # gradient reversal with modals
        if self.modal_classifier and len(contrast_features):
            modal_logit = self.modal_classifier(contrast_features.reshape([-1, contrast_features.shape[2]]))
            modal_label = tr.repeat_interleave(modal_indices, contrast_features.shape[1])
            modal_cls_loss = -F.cross_entropy(modal_logit, modal_label)
        else:
            modal_cls_loss = None

        return class_logit, contrast_loss, modal_cls_loss
