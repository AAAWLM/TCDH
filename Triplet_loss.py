'''
    Pytorch adaptation of https://omoindrot.github.io/triplet-loss
    https://github.com/omoindrot/tensorflow-triplet-loss
'''
import torch
import torch.nn as nn


class TripletMarginLoss(nn.Module):
    def __init__(self, margin, p=2., mining ='batch_all'):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.mining = mining
        self.loss_fn = batch_all_triplet_loss

    def forward(self, embeddings, labels):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        return self.loss_fn(labels, embeddings, self.margin, self.p)


def batch_all_triplet_loss(labels, embeddings, margin, p):
    pairwise_dist = torch.cdist(embeddings, embeddings, p=p)
    # [batchsize，batchsize]
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)

    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin

    mask = _get_triplet_mask(labels)

    triplet_loss = mask.float() * triplet_loss

    # Remove negative losses (easy triplets)
    triplet_loss[triplet_loss < 0] = 0
    # Count number of positive triplets (where triplet_loss > 0.pth)
    valid_triplets = triplet_loss[triplet_loss > 1e-16]
    num_positive_triplets = valid_triplets.size(0)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

    return triplet_loss


def _get_triplet_mask(labels):
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)

    valid_labels = ~i_equal_k & i_equal_j

    return valid_labels & distinct_indices

