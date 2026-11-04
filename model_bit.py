"""BiT ResNetV2-50x1 with explicit generic21K->1K weight loading."""
import timm
import torch
from torchvision import transforms
from dataset import Letterbox
MODEL='resnetv2_50x1_bit.goog_in21k_ft_in1k'
MEAN=[.5,.5,.5]
STD=[.5,.5,.5]


def build_model(pretrained_path=None):
    model=timm.create_model(MODEL,pretrained=False)
    if pretrained_path:
        state=torch.load(pretrained_path,map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
    model.reset_classifier(1)
    return model


def configure_preprocessing(transform):
    boxes=[t for t in transform.transforms if isinstance(t,Letterbox)]
    assert len(boxes)==1 and boxes[0].size==448
    assert isinstance(transform.transforms[-1],transforms.Normalize)
    transform.transforms[-1]=transforms.Normalize(MEAN,STD)
