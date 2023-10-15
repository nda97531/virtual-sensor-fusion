from typing import Dict, Union

import torch as tr
import torch.nn as nn
import torch.nn.functional as F

from vsf.networks.vsf_distributor import VsfDistributor


class BasicClsModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        """
        Basic classification model with a backbone and a classifier

        Args:
            backbone: backbone model
            classifier: classifier model
        """
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

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

        # relu between backbone and classifier
        x = F.relu(x)

        # [batch, channel]
        x = self.classifier(x, **classifier_kwargs)
        return x


class FusionClsModel(nn.Module):
    def __init__(self, backbones: nn.ModuleDict, backbone_output_dims: dict, classifier: nn.Module) -> None:
        """
        A feature-level fusion model that concatenates features of backbones before the classifier

        Args:
            backbones: a module dict of backbone models
            backbone_output_dims: output channel dim of each backbone, this dict has the same keys as `backbones`
            classifier: classifier model
        """
        super().__init__()
        self.backbones = backbones
        self.classifiers = classifier
        self.connect_fc = nn.ModuleDict({
            modal: nn.Linear(backbone_output_dims[modal], backbone_output_dims[modal])
            if backbone_output_dims[modal] else nn.Identity()
            for modal in backbones.keys()
        })  # activation function is implemented in `forward` function

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
                F.relu(
                    self.backbones[modal](
                        tr.permute(x_dict[modal], [0, 2, 1]), **backbone_kwargs)))
            for modal in self.backbones.keys()
        ], dim=1)

        # relu between backbone and classifier
        x = F.relu(x)

        # [batch, channel]
        x = self.classifiers(x, **classifier_kwargs)
        return x


class VsfModel(nn.Module):
    def __init__(self, backbones: nn.ModuleDict, distributor_head: VsfDistributor,
                 connect_feature_dims: Union[list, dict] = {},
                 cls_fusion_modals: list = (), ctr_fusion_modals: list = ()) -> None:
        """
        Combine backbones and heads, including classifier and contrastive loss head

        Args:
            backbones: a module dict of backbone models
            distributor_head: model head
            connect_feature_dims: feature dimension of FC layers used between backbone and distributor;
                this can be a list of 2 (applied for all modal),
                or a dict with keys are modal names, values are lists of 2
                default: don't use;
            cls_fusion_modals: list of fusion modals for classification; each item represents a fusion modal, and is a
                combined string of fused modals, separated by "+". For example if we want to fuse waist and wrist,
                skeleton and waist, we have ['waist+wrist', 'skeleton+waist']
            ctr_fusion_modals: same as `cls_fusion_modals` but for contrastive learning
        """
        super().__init__()
        self.backbones = backbones
        self.distributor = distributor_head

        # connect FCs, used between backbone and distributor
        if isinstance(connect_feature_dims, list) or isinstance(connect_feature_dims, tuple):
            connect_feature_dims = {key: connect_feature_dims for key in backbones.keys()}
        # backbone output isn't activated, so add a ReLU function before connect FCs
        self.connect_fc = nn.ModuleDict({
            modal: nn.Sequential(nn.ReLU(), nn.Linear(in_feat, out_feat))
            for modal, (in_feat, out_feat) in connect_feature_dims.items()
        })

        self.cls_fusion_modals = {f'{fused_modals}_cls': fused_modals.split('+') for fused_modals in cls_fusion_modals}
        self.ctr_fusion_modals = {f'{fused_modals}_ctr': fused_modals.split('+') for fused_modals in ctr_fusion_modals}

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
        for comb_name, fused_modals in self.cls_fusion_modals.items():
            x_fus_cls = [
                x_dict[modal][head_kwargs['cls_mask'][modal]]
                for modal in fused_modals if head_kwargs['cls_mask'][modal].any()
            ]
            if len(x_fus_cls):
                x_dict[comb_name] = tr.cat(x_fus_cls, dim=1)
                head_kwargs['cls_mask'][comb_name] = tr.tensor([True] * len(x_dict[comb_name]))
                head_kwargs['contrast_mask'][comb_name] = tr.tensor([False] * len(x_dict[comb_name]))

        for comb_name, fused_modals in self.ctr_fusion_modals.items():
            x_fus_contrast = [
                x_dict[modal][head_kwargs['contrast_mask'][modal]]
                for modal in fused_modals if head_kwargs['contrast_mask'][modal].any()
            ]
            if len(x_fus_contrast):
                x_dict[comb_name] = tr.cat(x_fus_contrast, dim=1)
                head_kwargs['contrast_mask'][comb_name] = tr.tensor([True] * len(x_dict[comb_name]))
                head_kwargs['cls_mask'][comb_name] = tr.tensor([False] * len(x_dict[comb_name]))

        # run connect FCs, keep order of x_dict
        x_dict = {
            modal: F.relu(self.connect_fc[modal](x_dict[modal])) if modal in self.connect_fc.keys()
            else F.relu(x_dict[modal])
            for modal in x_dict.keys()
        }

        class_logits, contrast_loss = self.distributor(x_dict, **head_kwargs)
        return class_logits, contrast_loss
