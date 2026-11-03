"""Explicit generic EVA02 initialization; fixed336 input and model-card preprocessing."""
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from timm.models.eva import checkpoint_filter_fn
from dataset import Letterbox

MODEL = 'eva02_small_patch14_336.mim_in22k_ft_in1k'
MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]


def build_model(pretrained_path=None):
    model = timm.create_model(MODEL, pretrained=False)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint_filter_fn(state, model), strict=True)
    model.reset_classifier(1)
    return model


class EvaLetterbox(Letterbox):
    def __call__(self, image):
        resized = ImageOps.contain(image, (self.size, self.size), Image.Resampling.BICUBIC)
        canvas = Image.new('RGB', (self.size, self.size))
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas


def configure_preprocessing(transform):
    boxes = [i for i, t in enumerate(transform.transforms) if isinstance(t, Letterbox)]
    assert len(boxes) == 1 and isinstance(transform.transforms[-1], transforms.Normalize)
    index = boxes[0]
    assert transform.transforms[index].size == 336
    transform.transforms[index] = EvaLetterbox(336)
    transform.transforms[-1] = transforms.Normalize(MEAN, STD)
