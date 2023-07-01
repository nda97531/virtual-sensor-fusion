import itertools

import numpy as np
import torch as tr
import torch.nn.functional as F


def info_nce_loss(modal1: tr.Tensor, modal2: tr.Tensor, temp: float = 1):
    """
    InfoNCE loss between 2 modalities

    Args:
        modal1: features of modal 1, tensor shape [batch, feature]
        modal2: features of modal 2, tensor shape [batch, feature]
        temp: temperature param

    Returns:
        a pytorch float
    """
    # tensor shape [batch, batch]
    sim = tr.matmul(modal1, modal2.permute([1, 0]))
    sim = sim / temp
    # create positive idx tensor on the same device as `sim`
    positive_pair_idx = sim.new(np.arange(sim.shape[0])).long()
    error = F.cross_entropy(input=sim, target=positive_pair_idx)
    return error


def cmc_loss(all_features: tr.Tensor, main_modal_idx: int = None, temp: float = 1):
    """
    Calculate CMC loss for multiple modals

    Args:
        all_features: feature of all modals in a batch, Tensor shape [modal, batch, feature]
        main_modal_idx: index of the main modal in `all_features`,
            if None, calculate InfoNCE between all possible pairs,
            if provided, only calculate for pairs containing it
        temp: temperature param

    Returns:
        a pytorch float
    """
    assert len(all_features) > 1, 'At least 2 modals are required for contrastive loss'

    if len(all_features) == 2:
        return info_nce_loss(modal1=all_features[0], modal2=all_features[1], temp=temp)

    error = 0
    for modal1_idx, modal2_idx in itertools.combinations(range(len(all_features)), 2):
        if (main_modal_idx is None) or (main_modal_idx in {modal1_idx, modal2_idx}):
            error += info_nce_loss(all_features[modal1_idx], all_features[modal2_idx], temp=temp)
    return error


def cocoa2_loss(all_features: tr.Tensor, temp: float = 1, scale_loss: float = 1 / 32, lambda_: float = 3.9e-3):
    """
    Calculate COCOA2 loss (sum of positive distance and negative similarity) from features of multiple modals.
    Source: https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py

    Args:
        all_features: feature of all modals in a batch, Tensor shape [modal, batch, feature]
        temp: temperature param
        scale_loss: weight of positive distance
        lambda_: weight of negative distance

    Returns:
        a pytorch float
    """
    assert len(all_features) > 1, 'At least 2 modals are required for contrastive loss'
    num_modal, batch_size, n_channel = all_features.shape

    # # positive pairs
    # # [batch size, modal, modal]
    pos_similarity = tr.matmul(all_features.permute([1, 0, 2]), all_features.permute([1, 2, 0]))
    pos_distance = 1 - pos_similarity
    pos_distance = tr.exp(pos_distance / temp)
    # [batch size,]; each is sum of distances between all possible pair of modals in a batch index
    pos_distance = tr.mean(pos_distance, dim=(1, 2)).sum()

    # # negative pairs
    # [modal, batch, batch]
    neg_similarity = tr.matmul(all_features, all_features.permute([0, 2, 1]))
    neg_similarity = tr.exp(neg_similarity / temp)
    tri_mask = tr.full_like(neg_similarity, fill_value=True, dtype=bool)
    tri_mask[:, tr.arange(batch_size), tr.arange(batch_size)] = False
    neg_similarity = neg_similarity[tri_mask].reshape([-1, batch_size, batch_size - 1])
    neg_similarity = neg_similarity.mean(dim=-1).sum()

    error = pos_distance * scale_loss + neg_similarity * lambda_
    return error


def cocoa_loss(all_features: tr.Tensor, temp: float = 1):
    """
    Calculate COCOA loss (with Cross Entropy) from features of multiple modals.
    Source: https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py

    Args:
        all_features: feature of all modals in a batch, Tensor shape [modal, batch, feature]
        temp: temperature param

    Returns:
        a pytorch float
    """
    assert len(all_features) > 1, 'At least 2 modals are required for contrastive loss'
    num_modal, batch_size, n_channel = all_features.shape

    # Positive pairs
    # similarity between all pairs of modals for each batch, shape [batch, modal, modal]
    pos_similarity = tr.matmul(all_features.permute([1, 0, 2]), all_features.permute([1, 2, 0]))
    pos_similarity = tr.exp(pos_similarity / temp)
    tri_mask = tr.full_like(pos_similarity, fill_value=True, dtype=tr.bool)
    tri_mask[:, tr.arange(num_modal), tr.arange(num_modal)] = False
    pos_similarity = pos_similarity[tri_mask].reshape([-1, num_modal, num_modal - 1])
    # shape [batch, modal, modal] -> [batch]
    pos_similarity = pos_similarity.sum(dim=[1, 2])

    # Negative pairs
    # similarity between all pairs of batch indices for each modal, shape [modal, batch, batch]
    neg_similarity = tr.matmul(all_features, all_features.permute([0, 2, 1]))
    neg_similarity = tr.exp(neg_similarity / temp)
    tri_mask = tr.full_like(neg_similarity, fill_value=True, dtype=tr.bool)
    tri_mask[:, tr.arange(batch_size), tr.arange(batch_size)] = False
    neg_similarity = neg_similarity[tri_mask].reshape([-1, batch_size, batch_size - 1])
    # shape [modal, batch, batch] -> [batch]
    neg_similarity = neg_similarity.mean(dim=2).sum(dim=0)

    probs = pos_similarity / (pos_similarity + neg_similarity)
    # create positive index tensor on the same device as `probs`
    pos_idx = probs.new(np.ones(batch_size))
    error = F.binary_cross_entropy(input=probs, target=pos_idx)
    return error


if __name__ == '__main__':
    features = {
        'acc': tr.rand([8, 15]),
        'gyro': tr.rand([8, 15]) * -1,
        'skeleton': tr.normal(mean=tr.zeros([8, 15]), std=tr.ones([8, 15])),
    }
    features = tr.stack(list(features.values()))
    # error = info_nce_loss(features['acc'], features['skeleton'])

    print(f'CMC: {cmc_loss(features)}')
    print(f'CMC with main modal: {cmc_loss(features, main_modal_idx=0)}')
    print(f'COCOA: {cocoa_loss(features)}')
    print(f'COCOA2: {cocoa2_loss(features)}')
