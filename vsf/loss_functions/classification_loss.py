import torch as tr
import torch.nn.functional as F


class AutoCrossEntropyLoss:
    def __init__(self, confidence_loss_weight: float = 0):
        """
        Calculate CE/binary-CE loss from input logit; Plus Entropy loss for more confident if necessary.

        Args:
            cal_confidence_loss: weight of the entropy loss (NOT cross entropy) to make the model more confident
                (regardless of what the prediction is). If this = 0, don't calculate confidence loss.
        """
        assert 0 <= confidence_loss_weight <= 1, \
            'confidence_loss_weight must be in range [0, 1] because classification loss is still the main goal'
        self.conf_loss_weight = confidence_loss_weight

    @staticmethod
    def confidence_loss(inp: tr.Tensor, is_binary: bool, reduce: str = 'mean'):
        """
        Calculate confidence loss

        Args:
            inp: prediction tensor, shape [batch size] if `is_binary`, else [batch size, num class]
            is_binary: whether this is binary class or multi-class
            reduce: reduce the batch dimension, accepted values are [mean|none|sum]

        Returns:
            a scalar
        """
        if is_binary:
            assert len(inp.shape) == 1, f'Expected binary class input, received tensor shape {inp.shape}'
            inp = tr.sigmoid(inp)
            conf_loss = -inp * tr.log(inp) - (1 - inp) * tr.log(1 - inp)
        else:
            assert len(inp.shape) == 2, f'Expected multi class input, received tensor shape {inp.shape}'
            conf_loss = (-tr.softmax(inp, dim=1) * tr.log_softmax(inp, dim=1)).sum(1)

        if reduce == 'mean':
            conf_loss = tr.mean(conf_loss)
        elif reduce == 'sum':
            conf_loss = tr.sum(conf_loss)
        elif reduce != 'none':
            raise ValueError('Accepted values for `reduce` are [mean|none|sum]')
        
        assert not conf_loss.isnan().any(), 'Confidence loss went wrong :( NAN'
        return conf_loss

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
            if self.conf_loss_weight > 0:
                conf_loss = self.confidence_loss(inp, is_binary=False)
                cls_loss += (conf_loss * self.conf_loss_weight)

        # if binary classification
        else:
            if len(inp.shape) > 1:
                inp = inp.squeeze(1)
            cls_loss = F.binary_cross_entropy_with_logits(inp, target.float())
            if self.conf_loss_weight > 0:
                conf_loss = self.confidence_loss(inp, is_binary=True)
                cls_loss += (conf_loss * self.conf_loss_weight)

        if cls_loss.isnan().any():
            _=1
        return cls_loss
