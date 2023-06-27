from typing import Dict

import torch as tr
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, dropout: float = 0.5) -> None:
        """
        Combine a backbone and a classifier

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


class FusionModel(nn.Module):
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
        })

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
