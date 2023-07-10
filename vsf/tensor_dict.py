from typing import Union

import torch as tr


class TensorDict:
    def __init__(self, x: Union[tr.Tensor, tuple], keys: list):
        """
        Simulate a dictionary of pytorch tensors, which support back prop

        Args:
            x: a tensor shape [N, ...] or a tuple of N tensors
            keys: list of N keys
        """
        assert len(x) == len(keys), 'number of keys and values must be the same'
        self.x = x
        self.dict_keys = keys

    def get(self, key):
        idx = self.dict_keys.index(key)
        return self.x[idx]

    def values(self):
        return self.x

    def keys(self):
        return self.dict_keys
