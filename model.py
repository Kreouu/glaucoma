"""Single-logit EfficientNet-B0; never downloads weights implicitly."""

import torch
from torchvision.models import efficientnet_b0


def build_model(pretrained_path=None):
    model = efficientnet_b0(weights=None)
    if pretrained_path:
        # Expects an original torchvision ImageNet state_dict, not a task checkpoint.
        model.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=True))
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    return model
