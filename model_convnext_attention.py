"""ConvNeXt-Tiny with trainable spatial pooling, initialized as average pooling."""
import torch
from model_convnext import build_model as build_backbone


class SpatialPool(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.score = torch.nn.Conv2d(channels, 1, 1, bias=False)
        torch.nn.init.zeros_(self.score.weight)

    def forward(self, features):
        weights = self.score(features).float().flatten(2).softmax(-1)
        pooled = (features.float().flatten(2) * weights).sum(-1)
        return pooled.to(features.dtype).unsqueeze(-1).unsqueeze(-1)


def build_model(pretrained_path=None):
    model = build_backbone(pretrained_path)
    model.head.global_pool = SpatialPool(model.num_features)
    return model
