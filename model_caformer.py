"""Local generic ImageNet22k-to1k CAFormer; retain the pretrained MLP head features."""
import timm
import torch
from timm.models.metaformer import checkpoint_filter_fn


def build_model(pretrained_path=None):
    model=timm.create_model('caformer_s18.sail_in22k_ft_in1k',pretrained=False)
    if pretrained_path:
        state=torch.load(pretrained_path,map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint_filter_fn(state,model),strict=True)
    model.head.fc.fc2=torch.nn.Linear(model.head.fc.fc2.in_features,1)
    model.num_classes=1
    return model
