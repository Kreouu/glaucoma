"""Torchvision EfficientNetV2-S; explicit ImageNet-1k file, no implicit download."""
import torch
from torchvision.models import efficientnet_v2_s


def build_model(pretrained_path=None):
    model=efficientnet_v2_s(weights=None)
    if pretrained_path:model.load_state_dict(torch.load(pretrained_path,map_location='cpu',weights_only=True),strict=True)
    model.classifier[1]=torch.nn.Linear(model.classifier[1].in_features,1)
    return model
