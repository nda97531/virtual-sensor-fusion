import torch as tr
import torch.nn as nn


class CompleteModel(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module, dropout: float = 0.5) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: tr.Tensor, backbone_kwargs: dict = {}, classifier_kwargs: dict = {}):
        """

        Args:
            x: [batch, ..., channel]
            backbone_kwargs:
            classifier_kwargs:

        Returns:

        """
        # change to [batch, channel, ...]
        x = tr.permute(x, [0, 2, 1])

        x = self.backbone(x, **backbone_kwargs)
        # [batch, channel]
        x = self.dropout(x)
        x = self.classifier(x, **classifier_kwargs)
        return x

# class MultiTaskModel(nn.Module):
#     def __init__(self, backbone: nn.Module, classifiers: nn.ModuleList) -> None:
#         super().__init__()
#         self.backbone = backbone
#         self.classifiers = classifiers
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.classifier(x)
#         return x
