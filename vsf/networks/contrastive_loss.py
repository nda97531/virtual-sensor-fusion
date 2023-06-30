import torch as tr


def cocoa2_loss(features_dict: dict, temp: float = 0.1, scale_loss: float = 1 / 32, lambda_: float = 3.9e-3):
    """
    Calculate contrastive loss from features of multiple modals.
    Source: https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py

    Args:
        features_dict: dict[modal name] = Tensor shape [batch, feature]
        temp: temperature param
        scale_loss: weight of positive distance
        lambda_: weight of negative distance

    Returns:
        a pytorch float, contrastive loss between modalities
    """
    assert len(features_dict) > 1, 'At least 2 modals are required for contrastive loss'
    all_features = tr.stack([modal_tensor for modal_tensor in features_dict.values()])
    num_modal, batch_size, n_channel = all_features.shape

    # # positive pairs
    # # [batch size, modal, modal]
    pos_error = tr.matmul(all_features.permute([1, 0, 2]), all_features.permute([1, 2, 0]))
    pos_error = 1 - pos_error
    pos_error = tr.exp(pos_error / temp)
    # [batch size,]; each is sum of distances between all possible pair of modals in a batch index
    pos_error = tr.mean(pos_error, dim=(1, 2)).sum()

    # # negative pairs
    # [modal, batch, batch]
    neg_error = tr.matmul(all_features, all_features.permute([0, 2, 1]))
    neg_error = tr.exp(neg_error / temp)
    tri_mask = tr.full_like(neg_error, fill_value=True, dtype=bool)
    tri_mask[:, tr.arange(batch_size), tr.arange(batch_size)] = False
    neg_error = neg_error[tri_mask].reshape([-1, batch_size, batch_size - 1])
    neg_error = neg_error.mean(dim=-1).sum()

    loss = pos_error * scale_loss + neg_error * lambda_
    return loss


def cocoa_loss():
    # https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py
    pass


if __name__ == '__main__':
    features = {
        'acc': tr.rand([8, 15]),
        'gyro': tr.rand([8, 15]),
        'skeleton': tr.rand([8, 15]),
    }
    cocoa2_loss(features, temp=1)

    # def num_dist(a, b):
    #     num = a * (a - 1) / 2
    #     m_dist = b ** 2 - b
    #     return num * m_dist
    #
    #
    # print(num_dist(3, 4))
    # print(num_dist(4, 3))
