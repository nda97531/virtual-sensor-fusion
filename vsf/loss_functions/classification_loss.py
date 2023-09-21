import torch as tr
import torch.nn.functional as F


class AutoCrossEntropyLoss:
    """
    Calculate CE/binary-CE loss from input logit;
    """
    def __call__(self, inp: tr.Tensor, target: tr.Tensor):
        """
        Calculate loss

        Args:
            inp: logit prediction
                - binary-class: shape [batch size] or [batch size, 1]
                - multi-class: shape [batch size, num class]
            target: label tensor (dtype: long), shape [batch size]

        Returns:
            a scalar
        """
        assert len(inp.shape) <= 2, \
            'This loss function only accepts input of shape [batch size] or [batch size, num class]'

        # if multi-class classification
        if (len(inp.shape) > 1) and (inp.shape[1] > 1):
            cls_loss = F.cross_entropy(inp, target)

        # if binary classification
        else:
            if len(inp.shape) > 1:
                inp = inp.squeeze(1)
            cls_loss = F.binary_cross_entropy_with_logits(inp, target.float())

        return cls_loss
