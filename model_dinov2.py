"""Generic DINOv2-S/14 from an explicit official checkpoint; no download."""
import timm
import torch
from timm.models.vision_transformer import checkpoint_filter_fn


def build_model(pretrained_path=None):
    model = timm.create_model(
        'vit_small_patch14_dinov2.lvd142m', pretrained=False,
        num_classes=0, dynamic_img_size=True,
    )
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
    model.reset_classifier(1)
    return model
