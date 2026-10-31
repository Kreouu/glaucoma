"""ConvNeXt-Small with explicit ImageNet-1k initialization; no implicit download."""
import timm
import torch
from timm.models.convnext import checkpoint_filter_fn


def build_model(pretrained_path=None):
    model=timm.create_model('convnext_small.fb_in1k',pretrained=False)
    if pretrained_path:
        state=torch.load(pretrained_path,map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint_filter_fn(state,model),strict=True)
    model.reset_classifier(1)
    return model
