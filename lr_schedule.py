"""Learning rate for each planned batch: linear warmup, then cosine without restarts."""

import math


def warmup_cosine_lr(step, total_steps, warmup_steps, peak_lr, min_lr):
    if not 0 <= step < total_steps or not 1 < warmup_steps < total_steps - 1:
        raise ValueError('Schedule needs at least two warmup and two cosine batches')
    if not 0 < min_lr <= peak_lr:
        raise ValueError('Expected 0 < min_lr <= peak_lr')
    if step < warmup_steps:
        fraction = step / (warmup_steps - 1)
        return min_lr + (peak_lr - min_lr) * fraction
    fraction = (step - warmup_steps) / (total_steps - warmup_steps - 1)
    return min_lr + .5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * fraction))
