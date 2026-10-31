"""Batch Mixup for training images and binary soft targets only."""
import numpy as np
import torch


def mix_batch(images, labels, alpha):
    if alpha == 0:
        return images, labels
    weight = float(np.random.beta(alpha, alpha))
    order = torch.randperm(len(labels), device=images.device)
    return (weight * images + (1 - weight) * images[order],
            weight * labels + (1 - weight) * labels[order])
