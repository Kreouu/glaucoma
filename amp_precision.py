"""AMP dtype metadata; historical checkpoints without precision use FP16."""

import torch


def amp_dtype(config):
    precision = config.get('precision', 'fp16')
    if precision == 'fp16':
        return torch.float16
    if precision == 'bf16':
        return torch.bfloat16
    raise ValueError(f'Unsupported AMP precision: {precision}')


def precision_description(config):
    amp_dtype(config)
    return f"{config.get('precision', 'fp16').upper()} autocast, FP32 sigmoid"
