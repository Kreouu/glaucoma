"""Training-only stochastic depth; state_dict and eval computation remain unchanged."""
import torch
from timm.layers import DropPath


def configure_drop_path(model,max_rate):
    blocks=[block for stage in model.stages for block in stage.blocks]
    rates=torch.linspace(0,max_rate,len(blocks)).tolist()
    for block,rate in zip(blocks,rates):block.drop_path=DropPath(rate)
    return dict(blocks=len(blocks),max_rate=max_rate,policy='linear depth schedule from zero to max_rate')
