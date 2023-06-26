from typing import Union

import torch as tr
import torch.nn as nn


class BasicClassifier(nn.Module):
    def __init__(self, n_features_in: int, n_classes_out: int):
        """
        FC classifier for single task

        Args:
            n_features_in: input dim
            n_classes_out: output dim
        """
        super().__init__()
        self.fc = nn.Linear(n_features_in, n_classes_out)

    def forward(self, x):
        x = self.fc(x)
        return x


class MaskedClassifiers(nn.Module):
    def __init__(self, n_features_in: int, n_classes_out: list):
        """
        FC classifiers for multiple classification tasks

        Args:
            n_features_in: input dim
            n_classes_out: list of output dims
        """
        super().__init__()
        self.fcs = nn.ModuleList()
        for n_c in n_classes_out:
            self.fcs.append(nn.Linear(n_features_in, n_c))

    def forward(self, x, mask: Union[tr.Tensor, str]):
        """
        Forward function

        Args:
            x: input tensor
            mask: an integer Tensor containing classifier index for each sample in x;
                if 'all', then all data will be passed through all classifiers

        Returns:
            a tuple, element at index I
        """
        if mask == 'all':
            output = tuple(self.fcs[i](x) for i in range(len(self.fcs)))
        else:
            output = tuple(self.fcs[i](x[mask == i]) for i in range(len(self.fcs)))
        return output


if __name__ == '__main__':
    model = MaskedClassifiers(n_features_in=10, n_classes_out=[2, 2])
    data = tr.ones([8, 10])
    mask = tr.Tensor([0, 1, 1, 1, 1, 1, 1, 1]).bool()
    output = model(data, mask)
    print(*output, sep='\n')
