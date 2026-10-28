"""SwinV2-Tiny window8, local ImageNet weights and explicit input resolution."""

import timm
import torch
from timm.models.swin_transformer_v2 import checkpoint_filter_fn

MODEL_NAME = 'swinv2_tiny_window8_256.ms_in1k'


def build_model(pretrained_path=None, size=384):
    if size not in (256, 384):
        raise ValueError('This comparison supports 256/window8 and 384/window12 only')
    model = timm.create_model(MODEL_NAME, pretrained=False, img_size=size,
                              window_size=size // 32, pretrained_window_sizes=(8, 8, 8, 8))
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
    model.reset_classifier(1)
    return model
