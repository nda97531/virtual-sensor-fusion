from typing import Dict, Union
import torch as tr
import torch.nn as nn

from vsf.networks.vsf_distributor import VsfDistributor


class BasicClsModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, dropout: float = 0.5) -> None:
        """
        Basic classification model with a backbone and a classifier

        Args:
            backbone: backbone model
            classifier: classifier model
            dropout: dropout rate between backbone and classifier
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: tr.Tensor, backbone_kwargs: dict = {}, classifier_kwargs: dict = {}):
        """
        Args:
            x: [batch, length, channel]
            backbone_kwargs: kwargs for backbone model
            classifier_kwargs: kwargs for classifier model

        Returns:
            output tensor
        """
        # change to [batch, channel, length]
        x = tr.permute(x, [0, 2, 1])

        x = self.backbone(x, **backbone_kwargs)
        # [batch, channel]
        x = self.dropout(x)
        x = self.classifier(x, **classifier_kwargs)
        return x


class FusionClsModel(nn.Module):
    def __init__(self, backbones: nn.ModuleDict, backbone_output_dims: dict, classifier: nn.Module,
                 dropout: float = 0.5) -> None:
        """
        A feature-level fusion model that concatenates features of backbones before the classifier

        Args:
            backbones: a module dict of backbone models
            classifier: classifier model
            backbone_output_dims: output channel dim of each backbone, this dict has the same keys as `backbones`
            dropout: dropout rate between backbone and classifier
        """
        super().__init__()
        self.backbones = backbones
        self.classifiers = classifier
        self.dropout = nn.Dropout(p=dropout)
        self.connect_fc = nn.ModuleDict({
            modal: nn.Linear(backbone_output_dims[modal], backbone_output_dims[modal])
            if backbone_output_dims[modal] else nn.Identity()
            for modal in backbones.keys()
        }) # activation function is implemented in `forward` function

    def forward(self, x_dict: Dict[str, tr.Tensor], backbone_kwargs: dict = {}, classifier_kwargs: dict = {}):
        """
        Args:
            x_dict: dict[input stream name] = tensor shape [batch, length, channel]
            backbone_kwargs: kwargs for backbone model
            classifier_kwargs: kwargs for classifier model

        Returns:
            output tensor
        """
        x = tr.cat([
            self.connect_fc[modal](
                self.backbones[modal](
                    tr.permute(x_dict[modal], [0, 2, 1]), **backbone_kwargs
                )
            )
            for modal in self.backbones.keys()
        ], dim=1)
        x = nn.functional.relu(x)
        # [batch, channel]

        x = self.dropout(x)
        x = self.classifiers(x, **classifier_kwargs)
        return x


class VsfModel(nn.Module):
    MODAL_FUSION_CLS = 'fusion_cls'
    MODAL_FUSION_CTR = 'fusion_contrast'

    def __init__(self, backbones: nn.ModuleDict, distributor_head: VsfDistributor,
                 connect_feature_dims: Union[int, dict] = {}) -> None:
        """
        Combine backbones and heads, including classifier and contrastive loss head

        Args:
            backbones: a module dict of backbone models
            distributor_head: model head
            connect_feature_dim: feature dimension of FC layers used between backbone and distributor; default: don't use;
                this can be a list of 2 (applied for all modal), or a dict with keys are modal names, values are lists of 2
        """
        super().__init__()
        self.backbones = backbones
        self.distributor = distributor_head

        # connect FCs, used between backbone and distributor
        if isinstance(connect_feature_dims, list) or isinstance(connect_feature_dims, tuple):
            connect_feature_dims = {key: connect_feature_dims for key in backbones.keys()}
        self.connect_fc = nn.ModuleDict({
            modal: nn.Sequential(nn.Linear(in_feat, out_feat), nn.ReLU())
            for modal, (in_feat, out_feat) in connect_feature_dims.items()
        })

        self.apply_fusion_cls = (self.MODAL_FUSION_CLS in self.connect_fc.keys()) or \
            (self.MODAL_FUSION_CLS in self.distributor.classifiers.keys())
        
        self.apply_fusion_ctr = (self.MODAL_FUSION_CTR in self.connect_fc.keys()) or \
            (self.MODAL_FUSION_CTR in self.distributor.input_dims.keys())

    def forward(self, x_dict: Dict[str, tr.Tensor], backbone_kwargs: dict = {}, head_kwargs: dict = {}):
        """
        Args:
            x_dict: dict[input stream name] = tensor shape [batch, length, channel]
            backbone_kwargs: kwargs for backbone model
            head_kwargs: kwargs for head model

        Returns:
            a tuple of 2 elements:
                - a dict, keys are modal names, values are class logits predicted from that modal,
                    value tensor shape is [batch, num class]
                - contrastive loss (pytorch float)
        """
        # run backbones by order in backbones dict
        # dict[modal] = [batch, channel]
        x_dict = {
            modal: self.backbones[modal](tr.permute(x_dict[modal], [0, 2, 1]), **backbone_kwargs)
            for modal in self.backbones.keys()
            if modal in x_dict
        }
        # add fusion feature to x_dict, cls and contrast fusion are done separately because they may use
        # different number of modals
        if self.apply_fusion_cls:
            x_fus_cls = [
                feat[head_kwargs['cls_mask'][modal]]
                for modal, feat in x_dict.items() if head_kwargs['cls_mask'][modal].any()
            ]
            if len(x_fus_cls):
                x_dict[self.MODAL_FUSION_CLS] = tr.cat(x_fus_cls, dim=1)
                head_kwargs['cls_mask'][self.MODAL_FUSION_CLS] = tr.tensor([True] * len(x_dict[self.MODAL_FUSION_CLS]))
                head_kwargs['contrast_mask'][self.MODAL_FUSION_CLS] = tr.tensor([False] * len(x_dict[self.MODAL_FUSION_CLS]))

        if self.apply_fusion_ctr:
            x_fus_contrast = [
                feat[head_kwargs['contrast_mask'][modal]]
                for modal, feat in x_dict.items() if head_kwargs['contrast_mask'][modal].any()
            ]
            if len(x_fus_contrast):
                x_dict[self.MODAL_FUSION_CTR] = tr.cat(x_fus_contrast, dim=1)
                head_kwargs['contrast_mask'][self.MODAL_FUSION_CTR] = tr.tensor([True] * len(x_dict[self.MODAL_FUSION_CTR]))
                head_kwargs['cls_mask'][self.MODAL_FUSION_CTR] = tr.tensor([False] * len(x_dict[self.MODAL_FUSION_CTR]))

        # run connect FCs, keep order of x_dict
        x_dict = {
            modal: self.connect_fc[modal](x_dict[modal]) if modal in self.connect_fc.keys() else x_dict[modal]
            for modal in x_dict.keys()
        }

        class_logits, contrast_loss = self.distributor(x_dict, **head_kwargs)
        return class_logits, contrast_loss
