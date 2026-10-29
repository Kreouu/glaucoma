"""Single-logit InceptionNeXt-Tiny with explicit local ImageNet-1k weights."""

import timm
import torch


def build_model(pretrained_path=None):
    model = timm.create_model('inception_next_tiny.sail_in1k', pretrained=False)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True), strict=True)
    model.reset_classifier(1)
    return model
