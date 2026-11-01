"""Pairwise logistic ranking over positive/negative examples in a training batch."""
import torch.nn.functional as F


def pairwise_logistic_loss(logits, targets):
    values = logits.float()
    positive = values[targets == 1]
    negative = values[targets == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        return values.sum() * 0
    return F.softplus(negative[:, None] - positive[None, :]).mean()
