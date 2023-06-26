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
    def __init__(self, backbones: nn.ModuleDict, classifier: nn.Module, dropout: float = 0.5) -> None:
        """
        A feature-level fusion model that concatenates features of backbones before the classifier

        Args:
            backbones: a module dict of backbone models
            classifier: classifier model
            dropout: dropout rate between backbone and classifier
        """
        super().__init__()
        self.backbones = backbones
        self.classifiers = classifier
        self.dropout = nn.Dropout(p=dropout)

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
            self.backbones[key](tr.permute(x_dict[key], [0, 2, 1]), **backbone_kwargs)
            for key in x_dict.keys()
        ], dim=1)
        # [batch, channel]

        x = self.dropout(x)
        x = self.classifier(x, **classifier_kwargs)
        return x
