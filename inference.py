"""Frozen single-image AMP inference shared by prediction and timing."""

import csv
import hashlib
import math
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import Letterbox
from model import build_model
from amp_precision import amp_dtype, precision_description


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(folder):
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))


def resolve_images(names, folder):
    paths = image_paths(folder)
    mapping = {}
    for path in paths:
        key = path.name.casefold()
        if key in mapping:
            raise ValueError(f'Ambiguous filename: {path.name}')
        mapping[key] = path
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError('Duplicate input names')
    resolved = []
    for name in names:
        if not name or Path(name).name != name or name.casefold() not in mapping:
            raise ValueError(f'Missing or invalid image: {name}')
        resolved.append(mapping[name.casefold()])
    return resolved


class Predictor:
    def __init__(self, checkpoint):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.config = saved['config']
        self.autocast_dtype = amp_dtype(self.config)
        self.precision_description = precision_description(self.config)
        if self.autocast_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ValueError('This checkpoint requires BF16 support')
        project = Path(__file__).resolve().parent
        sources = ['dataset.py', 'model.py']
        if 'amp_precision.py' in self.config['source_sha256']:
            sources.append('amp_precision.py')
        if self.config['model'] == 'efficientnet_b3':
            sources.append('model_b3.py')
        if self.config['model'] == 'convnext_tiny':
            import timm
            if timm.__version__ != self.config['timm_version']:
                raise ValueError('timm version differs from training')
            sources.append('model_convnext.py')
        if self.config['model'] == 'swinv2_tiny':
            import timm
            if timm.__version__ != self.config['timm_version']:
                raise ValueError('timm version differs from training')
            sources.append('model_swinv2.py')
        for name in sources:
            if file_sha256(project / name) != self.config['source_sha256'][name]:
                raise ValueError(f'Inference source changed since training: {name}')
        if not self.config['amp']:
            raise ValueError('This inference implementation requires the AMP baseline')
        self.threshold = self.config['threshold']
        if self.config['model'] == 'efficientnet_b3':
            from model_b3 import build_model as selected_builder
        else:
            selected_builder = build_model
        if self.config['model'] == 'swinv2_tiny':
            from model_swinv2 import build_model as swin_builder
            self.model = swin_builder(size=self.config['size'])
        elif self.config['model'] == 'convnext_tiny':
            from model_convnext import build_model as convnext_builder
            self.model = convnext_builder()
        else:
            self.model = selected_builder()
        self.model.load_state_dict(saved['model'])
        self.model = self.model.cuda().eval().requires_grad_(False)
        self.transform = transforms.Compose([
            Letterbox(self.config['size']), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def preprocess(self, path):
        with Image.open(path) as image:
            if self.config['jpeg_draft'] and image.format == 'JPEG':
                image.draft('RGB', (self.config['size'], self.config['size']))
            return self.transform(image.convert('RGB'))

    @torch.inference_mode()
    def predict_one(self, path):
        image = self.preprocess(path).unsqueeze(0).cuda()
        with torch.autocast('cuda', dtype=self.autocast_dtype):
            logit = self.model(image).flatten()[0]
        probability = logit.float().sigmoid().item()
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid model probability')
        return probability, int(probability >= self.threshold)


def write_prediction(writer, stream, name, probability, label):
    writer.writerow([name, format(probability, '.17g'), label])
    stream.flush()


def read_submission(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['Image_name', 'Probability', 'Pred_Label']:
            raise ValueError('Submission must have exactly Image_name,Probability,Pred_Label')
        rows = list(reader)
    if not rows or len({r['Image_name'].casefold() for r in rows}) != len(rows):
        raise ValueError('Empty submission or duplicate names')
    for row in rows:
        p = float(row['Probability'])
        if not math.isfinite(p) or not 0 <= p <= 1 or row['Pred_Label'] not in ('0', '1'):
            raise ValueError(f'Invalid submission row: {row["Image_name"]}')
    return rows
