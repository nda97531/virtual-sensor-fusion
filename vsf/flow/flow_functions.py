def prob_2_categorical(y_pred):
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
