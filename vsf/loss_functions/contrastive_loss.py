import itertools

import torch as tr
import torch.nn.functional as F
from torch import nn


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


def filtered_infonce_loss(modal1: tr.Tensor, modal2: tr.Tensor, cos_thres: float = 0.5,
                          temp: float = 0.1, eps: float = 1e-6):
    """
    InfoNCE loss between 2 modalities

    Args:
        modal1: features of modal 1, tensor shape [batch, feature]
        modal2: features of modal 2, tensor shape [batch, feature]
        cos_thres: only include pairs with cosine similarity less than or equal to this threshold when
            calculating loss
        temp: temperature param
        eps: small epsilon to avoid division by 0

    Returns:
        a pytorch float
    """
    assert modal1.shape == modal2.shape

    # cosine sim
    length1 = tr.sqrt((modal1 ** 2).sum(dim=-1, keepdims=True))
    length2 = tr.sqrt((modal2 ** 2).sum(dim=-1, keepdims=True))
    # tensor shape [batch, batch]
    cross_sim = torch_cosine_similarity(modal1, modal2, length1, length2, eps) / temp
    modal1_sim = torch_cosine_similarity(modal1, modal1, length1, length1, eps)
    modal2_sim = torch_cosine_similarity(modal2, modal2, length2, length2, eps)

    # mask: False: too similar pairs; True: dissimilar pairs for infoNCE loss
    batch_sim_mask = (modal1_sim <= cos_thres) | (modal2_sim <= cos_thres)
    # must include diagonal because it's y_true
    batch_size = len(cross_sim)
    batch_sim_mask[range(batch_size), range(batch_size)] = True

    if not batch_sim_mask.all():
        error = 0
        count = 0        
        for i in range(batch_size):
            # exclude too similar pairs
            item_sim = cross_sim[i][batch_sim_mask[i]]
            if len(item_sim) > 1:
                item_label = i - (~batch_sim_mask[i, :i]).sum()
                error += F.cross_entropy(item_sim, item_label)
                count += 1
        error = (error / count) if count else modal1.new_tensor(0.)
    else:
        # create positive idx tensor on the same device as `sim`
        positive_pair_idx = cross_sim.new_tensor(range(cross_sim.shape[0]), dtype=tr.long)
        error = F.cross_entropy(cross_sim, positive_pair_idx)

    return error


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


class CMCLoss(ContrastiveLoss):
    def __init__(self, main_modal_idx: int = None, ignore_submodal: bool = False,
                 cos_thres: float = 0.5, temp: float = 0.1, eps: float = 1e-6, *args, **kwargs):
        """
        CMC loss for multiple modals

        Args:
            main_modal_idx: index of the main modal in `all_features`,
                if None, calculate InfoNCE between all possible pairs,
                if provided, only calculate for pairs containing it
            ignore_submodal: only relevant if `main_modal_idx` is provided;
                whether to detach sub-modals when optimising CMC loss
            cos_thres: only include pairs with cosine similarity less than or equal to this threshold when
                calculating infoNCE loss
            temp: temperature param
            eps: epsilon added to norm2 when calculating cosine similarity to avoid division by 0
        """
        super().__init__(*args, **kwargs)
        self.main_modal_idx = main_modal_idx
        self.temp = temp
        self.cos_thres = cos_thres
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

        error = 0
        num_components = 0
        for modal1_idx, modal2_idx in itertools.combinations(range(len(all_features)), 2):
            if self.main_modal_idx is None:
                error += filtered_infonce_loss(all_features[modal1_idx], all_features[modal2_idx],
                                               cos_thres=self.cos_thres, temp=self.temp, eps=self.eps)
                num_components += 1

            elif self.main_modal_idx in {modal1_idx, modal2_idx}:
                modal1_feat = all_features[modal1_idx]
                modal2_feat = all_features[modal2_idx]
                if self.ignore_submodal:
                    if modal1_idx == self.main_modal_idx:
                        modal2_feat = modal2_feat.detach()
                    else:
                        modal1_feat = modal1_feat.detach()
                error += filtered_infonce_loss(modal1_feat, modal2_feat, 
                                               cos_thres=self.cos_thres, temp=self.temp, eps=self.eps)
                num_components += 1

        error /= num_components
        return error


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


if __name__ == '__main__':
    features = {
        'acc': tr.rand([8, 15]),
        'gyro': tr.rand([8, 15]) * -1,
        'skeleton': tr.normal(mean=tr.zeros([8, 15]), std=tr.ones([8, 15])),
    }
    features = tr.stack(list(features.values()))
    # error = info_nce_loss(features['acc'], features['skeleton'])

    print(f'CMC: {CMCLoss(main_modal_idx=None)(features)}')
    print(f'CMC with main modal: {CMCLoss(main_modal_idx=0)(features)}')
    print(f'COCOA: {CocoaLoss()(features)}')
    print(f'COCOA2: {Cocoa2Loss()(features)}')
