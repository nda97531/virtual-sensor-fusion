import torch as tr
import torch.nn.functional as F

from sklearn import metrics


def auto_classification_loss(inp: tr.Tensor, target: tr.Tensor):
    if len(inp.shape) > 1:
        # if multi-class classification
        if inp.shape[1] > 1:
            loss = F.cross_entropy(inp, target)
            return loss

        # if binary classification
        else:
            inp = inp.squeeze(1)

    # if binary classification
    loss = F.binary_cross_entropy_with_logits(inp, target.float())
    return loss


def ypred_2_categorical(y_pred):
    """
    Convert array to categorical values

    Args:
        y_pred: 2d array/tensor shape [num samples, num class], if num class==1, treat this as a
            probability array for binary classification; if 1d array, treat as categorical

    Returns:
        categorical 1D array of y_pred
    """
    if len(y_pred.shape) > 1:
        # if multi-class classification
        if y_pred.shape[1] > 1:
            y_pred = y_pred.argmax(1)

        # if binary classification
        else:
            y_pred = (tr.sigmoid(y_pred.squeeze(1)) >= 0.5).long()
    return y_pred


def f1_score_from_prob(y_true, y_pred):
    """
    Calculate F1 score from categorical labels and predicted probabilities

    Args:
        y_true: 1d array/tensor shape [num samples]
        y_pred: 2d array/tensor shape [num samples, num class], if num class==1, treat this as a
            probability array for binary classification; if 1d array, treat as categorical

    Returns:
        float: f1 score
    """
    y_pred = ypred_2_categorical(y_pred)
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return f1


def classification_report(y_true, y_pred, *args, **kwargs):
    """
    Calculate classification report using sklearn

    Args:
        y_true: 1d array/tensor shape [num samples]
        y_pred: 2d array/tensor shape [num samples, num class], if num class==1, treat this as a
            probability array for binary classification; if 1d array, treat as categorical
        args, kwargs: arguments for sklearn classification_report

    Returns:
        classification_report
    """
    y_pred = ypred_2_categorical(y_pred)
    cls_report = metrics.classification_report(y_true, y_pred, *args, **kwargs)
    return cls_report
