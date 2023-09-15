import torch as tr
from sklearn import metrics


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
            y_pred = (y_pred.squeeze(1) >= 0).long()
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
