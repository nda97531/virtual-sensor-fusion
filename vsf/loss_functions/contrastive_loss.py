import itertools

import torch as tr
import torch.nn.functional as F
from torch import nn
import numpy as np


def torch_cosine_similarity(tensor1: tr.Tensor, tensor2: tr.Tensor,
                            length1: tr.Tensor = None, length2: tr.Tensor = None,
                            eps: float = 1e-6) -> tr.Tensor:
    """
    Calculate pairwise cosine similarity between 2 torch Tensors
    Args:
        tensor1: tensor shape [batch size 1, feature]
        tensor2: tensor shape [batch size 2, feature]
        length1: length of vectors in tensor1, tensor shape [batch size 1]
        length2: length of vectors in tensor2, tensor shape [batch size 2]
        eps: small epsilon to avoid division by 0

    Returns:
        tensor shape [batch size 1, batch size 2]
    """
    # tensor shape [batch, batch]
    if length1 is None:
        length1 = tr.sqrt((tensor1 ** 2).sum(dim=-1, keepdims=True))
    if length2 is None:
        length2 = tr.sqrt((tensor2 ** 2).sum(dim=-1, keepdims=True))
    sim = tr.matmul(tensor1, tensor2.permute([1, 0])) / \
          tr.maximum(tr.matmul(length1, length2.permute([1, 0])), tr.tensor(eps))
    return sim


def infonce_loss(view1: tr.Tensor, view2: tr.Tensor,
                 temp: float = 0.1, eps: float = 1e-6, reduction: str = 'mean'):
    """
    InfoNCE loss between 2 views. Every item in view1 matches the item at the same index in view2.
    View2 can have more items than view1 (means more negative pairs).

    Args:
        view1: features of view 1, tensor shape [batch, feature]
        view2: features of view 2, tensor shape [batch, feature]
        temp: temperature param
        eps: small epsilon to avoid division by 0
        reduction: Specifies the reduction to apply to the output:
            ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.

    Returns:
        a pytorch float
    """
    assert view1.shape == view2.shape
    batch_size = len(view1)
    n = 2 * batch_size

    views = tr.cat((view1, view2), dim=0)
    sim = torch_cosine_similarity(views, views, eps=eps)
    sim = sim[~tr.eye(n, dtype=bool)].reshape(n, n - 1) / temp

    positive_idx = tr.tensor(
        list(range(batch_size - 1, n - 1)) + list(range(0, batch_size)),
        dtype=tr.long,
        device=sim.device
    )

    loss = F.cross_entropy(sim, positive_idx, reduction=reduction)
    return loss


class ContrastiveLoss(nn.Module):
    def forward(self, all_features: tr.Tensor):
        """
        Calculate loss

        Args:
            all_features: feature of all modals in a batch, Tensor shape [modal, batch, feature]

        Returns:
            pytorch float
        """
        raise NotImplementedError()


class MultiviewNTXentLoss(ContrastiveLoss):
    def __init__(self, main_modal_idx: int = None, ignore_submodal: bool = False,
                 temp: float = 0.1, eps: float = 1e-6, *args, **kwargs):
        """
        CMC loss for multiple modals

        Args:
            main_modal_idx: index of the main modal in `all_features`,
                if None, calculate InfoNCE between all possible pairs,
                if provided, only calculate for pairs containing it
            ignore_submodal: only relevant if `main_modal_idx` is provided;
                whether to detach sub-modals when optimising CMC loss
            temp: temperature param
            eps: epsilon added to norm2 when calculating cosine similarity to avoid division by 0
        """
        super().__init__(*args, **kwargs)
        self.main_modal_idx = main_modal_idx
        self.temp = temp
        self.eps = eps
        # only ignore submodal if a main modal exists
        self.ignore_submodal = ignore_submodal and (main_modal_idx is not None)

    def forward(self, all_features: tr.Tensor):
        """
        Args:
            all_features: tensor shape [modal, batch, feature]

        Returns:
            CMC loss
        """
        assert len(all_features) > 1, 'At least 2 modals are required for contrastive loss'

        loss = 0
        num_components = 0
        for modal1_idx, modal2_idx in itertools.combinations(range(len(all_features)), 2):
            if self.main_modal_idx is None:
                loss += infonce_loss(all_features[modal1_idx], all_features[modal2_idx],
                                     temp=self.temp, eps=self.eps)
                num_components += 1

            elif self.main_modal_idx in {modal1_idx, modal2_idx}:
                modal1_feat = all_features[modal1_idx]
                modal2_feat = all_features[modal2_idx]
                if self.ignore_submodal:
                    if modal1_idx == self.main_modal_idx:
                        modal2_feat = modal2_feat.detach()
                    else:
                        modal1_feat = modal1_feat.detach()
                loss += infonce_loss(modal1_feat, modal2_feat,
                                     temp=self.temp, eps=self.eps)
                num_components += 1

        loss /= num_components
        return loss


class CocoaLoss(ContrastiveLoss):
    def __init__(self, temp: float = 0.1, eps: float = 1e-6, *args, **kwargs):
        """
        Calculate COCOA loss (with Cross Entropy) from features of multiple modals.
        Source: https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py

        Args:
            temp: temperature param
            eps: epsilon added to norm2 when calculating cosine similarity to avoid division by 0
        """
        super().__init__(*args, **kwargs)
        self.temp = temp
        self.eps = tr.tensor(eps)

    def forward(self, all_features: tr.Tensor):
        assert len(all_features) > 1, 'At least 2 modals are required for contrastive loss'
        assert all_features.shape[1] > 1, 'At least 2 batch items are required for contrastive loss'
        num_modal, batch_size, n_channel = all_features.shape
        feature_norm2 = tr.maximum(tr.sqrt((all_features ** 2).sum(dim=-1, keepdims=True)), self.eps)

        # Positive pairs
        # similarity between all pairs of modals for each batch, shape [batch, modal, modal]
        pos_similarity = tr.matmul(all_features.permute([1, 0, 2]), all_features.permute([1, 2, 0]))
        pos_similarity /= tr.matmul(feature_norm2.permute([1, 0, 2]), feature_norm2.permute([1, 2, 0]))
        # remove diagonal (similarity with itself)
        tri_mask = tr.full_like(pos_similarity, fill_value=True, dtype=tr.bool, requires_grad=False)
        tri_mask[:, tr.arange(num_modal), tr.arange(num_modal)] = False
        pos_similarity = pos_similarity[tri_mask].reshape([-1, num_modal, num_modal - 1])

        pos_similarity = tr.exp(pos_similarity / self.temp)
        # shape [batch, modal, modal-1] -> [batch]
        pos_similarity = pos_similarity.sum(dim=[1, 2])

        # Negative pairs
        # similarity between all pairs of batch indices for each modal, shape [modal, batch, batch]
        neg_similarity = tr.matmul(all_features, all_features.permute([0, 2, 1]))
        neg_similarity /= tr.matmul(feature_norm2, feature_norm2.permute([0, 2, 1]))
        # remove diagonal (similarity with itself)
        tri_mask = tr.full_like(neg_similarity, fill_value=True, dtype=tr.bool, requires_grad=False)
        tri_mask[:, tr.arange(batch_size), tr.arange(batch_size)] = False
        neg_similarity = neg_similarity[tri_mask].reshape([-1, batch_size, batch_size - 1])

        neg_similarity = tr.exp(neg_similarity / self.temp)
        # shape [modal, batch, batch-1] -> [batch]
        neg_similarity = neg_similarity.mean(dim=2).sum(dim=0)

        probs = pos_similarity / (pos_similarity + neg_similarity)
        # create positive index tensor on the same device as `probs`
        pos_idx = probs.new_tensor([1.] * batch_size)
        error = F.binary_cross_entropy(input=probs, target=pos_idx)
        return error


class Cocoa2Loss(ContrastiveLoss):
    def __init__(self, temp: float = 0.1, eps: float = 1e-6, scale_loss: float = 1 / 32, lambda_: float = 3.9e-3,
                 *args, **kwargs):
        """
        Calculate COCOA loss (with Cross Entropy) from features of multiple modals.
        Source: https://github.com/cruiseresearchgroup/COCOA/blob/main/src/losses.py

        Args:
            temp: temperature param
            scale_loss: weight of positive distance
            lambda_: weight of negative distance
        """
        super().__init__(*args, **kwargs)
        self.temp = temp
        self.eps = tr.tensor(eps)
        self.scale_loss = scale_loss
        self.lambda_ = lambda_

    def forward(self, all_features: tr.Tensor):
        assert len(all_features) > 1, 'At least 2 modals are required for contrastive loss'
        num_modal, batch_size, n_channel = all_features.shape
        feature_norm2 = tr.maximum(tr.sqrt((all_features ** 2).sum(dim=-1, keepdims=True)), self.eps)

        # # positive pairs
        # # [batch size, modal, modal]
        pos_similarity = tr.matmul(all_features.permute([1, 0, 2]), all_features.permute([1, 2, 0]))
        pos_similarity /= tr.matmul(feature_norm2.permute([1, 0, 2]), feature_norm2.permute([1, 2, 0]))
        pos_distance = 1 - pos_similarity
        pos_distance = tr.exp(pos_distance / self.temp)
        # [batch size,]; each is sum of distances between all possible pair of modals in a batch index
        pos_distance = tr.mean(pos_distance, dim=(1, 2)).sum()

        # # negative pairs
        # [modal, batch, batch]
        neg_similarity = tr.matmul(all_features, all_features.permute([0, 2, 1]))
        neg_similarity /= tr.matmul(feature_norm2, feature_norm2.permute([0, 2, 1]))

        tri_mask = tr.full_like(neg_similarity, fill_value=True, dtype=tr.bool, requires_grad=False)
        tri_mask[:, tr.arange(batch_size), tr.arange(batch_size)] = False
        neg_similarity = neg_similarity[tri_mask].reshape([-1, batch_size, batch_size - 1])

        neg_similarity = tr.exp(neg_similarity / self.temp)
        neg_similarity = neg_similarity.mean(dim=-1).sum()

        error = pos_distance * self.scale_loss + neg_similarity * self.lambda_
        return error


def cmkm_loss(modal1: tr.Tensor, modal2: tr.Tensor, gamma_thres: float = 0.7, top_k: int = 1,
              temp: float = 0.1, eps: float = 1e-6):
    """
    Calculate contrastive loss with cross-modal knowledge mining, but don't use another pre-trained model like the
    original version.
    Source: https://github.com/razzu/cmc-cmkm/blob/main/models/cmc_cvkm.py

    Args:
        modal1: features of modal 1, tensor shape [batch, feature]
        modal2: features of modal 2, tensor shape [batch, feature]
        gamma_thres: cosine similarity threshold to prune false negatives
        top_k: top K similar pairs to set as positive
        temp: temperature param
        eps: small epsilon to avoid division by 0

    Returns:
        a pytorch float
    """
    assert modal1.shape == modal2.shape
    batch_size = len(modal1)

    # calculate intra-modal similarity
    length1 = tr.sqrt((modal1 ** 2).sum(dim=-1, keepdims=True))
    length2 = tr.sqrt((modal2 ** 2).sum(dim=-1, keepdims=True))
    modal1_sim = torch_cosine_similarity(modal1, modal1, length1, length1, eps)
    modal2_sim = torch_cosine_similarity(modal2, modal2, length2, length2, eps)

    # initialize mask
    positive_mask = modal1.new_tensor(np.eye(batch_size)).repeat([2, 1]).bool()

    # mine positive
    eye_mask = modal1.new_tensor(np.eye(batch_size))
    topk_idx_1 = tr.topk(modal1_sim - eye_mask, k=top_k).indices
    topk_idx_2 = tr.topk(modal2_sim - eye_mask, k=top_k).indices

    upper_row_idx = tr.arange(batch_size).unsqueeze(1)
    positive_mask[upper_row_idx, topk_idx_2] = True
    lower_row_idx = tr.arange(batch_size, 2 * batch_size).unsqueeze(1)
    positive_mask[lower_row_idx, topk_idx_1] = True
    negative_mask = ~positive_mask

    # prune negative
    prune_idx_1 = modal1_sim.mean(dim=0) > gamma_thres
    prune_idx_2 = modal2_sim.mean(dim=0) > gamma_thres

    negative_mask[tr.arange(batch_size).unsqueeze(1), prune_idx_2] = False
    negative_mask[tr.arange(batch_size, 2 * batch_size).unsqueeze(1), prune_idx_1] = False

    # calculate cross modal similarity
    # tensor shape [batch, batch]
    cross_sim = tr.exp(torch_cosine_similarity(modal1, modal2, length1, length2, eps) / temp)
    # tensor shape [2batch, batch]
    cross_sim = tr.cat([cross_sim, cross_sim.T], dim=0)

    numerators = tr.sum(cross_sim * positive_mask, dim=1)
    denominators = tr.sum(cross_sim * negative_mask, dim=1)

    # add intra-modal negative pairs
    intra_modal_sims = tr.exp(tr.cat([modal1_sim, modal2_sim], dim=0))
    intra_negative_mask = ~modal1.new_tensor(np.eye(batch_size)).repeat([2, 1]).bool()
    intra_negative_mask[upper_row_idx, topk_idx_1] = False
    intra_negative_mask[lower_row_idx, topk_idx_2] = False
    intra_negative_mask[upper_row_idx, prune_idx_1] = False
    intra_negative_mask[lower_row_idx, prune_idx_2] = False

    denominators += tr.sum(intra_modal_sims * intra_negative_mask, dim=1)

    # add intra-modal positives
    intra_positive_idx = modal1.new_tensor(np.zeros([2 * batch_size, batch_size])).bool()
    intra_positive_idx[upper_row_idx, topk_idx_1] = True
    intra_positive_idx[lower_row_idx, topk_idx_2] = True
    numerators += tr.sum(intra_modal_sims * intra_positive_idx, dim=1)

    loss = -tr.log((numerators / (numerators + denominators)))
    loss = loss.mean()
    return loss


if __name__ == '__main__':
    features = {
        'acc': tr.rand([8, 15]),
        'gyro': tr.rand([8, 15]) * -1,
        'skeleton': tr.normal(mean=tr.zeros([8, 15]), std=tr.ones([8, 15])),
    }
    # features = tr.stack(list(features.values()))
    lo = cmkm_loss(features['acc'], features['skeleton'], gamma_thres=0.7)

    print(f'CMC: {MultiviewNTXentLoss(main_modal_idx=None)(features)}')
    print(f'CMC with main modal: {MultiviewNTXentLoss(main_modal_idx=0)(features)}')
    print(f'COCOA: {CocoaLoss()(features)}')
    print(f'COCOA2: {Cocoa2Loss()(features)}')
