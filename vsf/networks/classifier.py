import torch as tr
import torch.nn as nn
from typing import Union


class FCClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        """

        Args:
            n_features:
            n_classes:
        """
        super().__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class MultiFCClassifiers(nn.Module):
    def __init__(self, n_features: int, n_classes: list):
        """

        Args:
            n_features:
            n_classes:
        """
        super().__init__()
        self.fcs = nn.ModuleList()
        for n_c in n_classes:
            self.fcs.append(nn.Linear(n_features, n_c))

    def forward(self, x, mask: Union[tr.Tensor, str]):
        """

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
    model = MultiFCClassifiers(n_features=10, n_classes=[2, 2])
    data = tr.ones([8, 10])
    mask = tr.Tensor([0, 1, 1, 1, 1, 1, 1, 1]).bool()
    output = model(data, mask)
    print(*output, sep='\n')
