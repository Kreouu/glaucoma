"""Single-logit torchvision EfficientNet-B3; explicit local ImageNet weights only."""

import torch
from torchvision.models import efficientnet_b3


def build_model(pretrained_path=None):
    model = efficientnet_b3(weights=None)
    if pretrained_path:
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    return model
