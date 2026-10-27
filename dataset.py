"""Official train/valid image loading without modifying the source data."""

import csv
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms


class Letterbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        resized = ImageOps.contain(image, (self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size))
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas


class FundusDataset(Dataset):
    def __init__(self, root, split, size=256, limit=None, seed=42):
        if split not in ('train', 'valid'):
            raise ValueError('Only train and valid are supported by the training loader')
        root = Path(root)
        with (root / f'{split}.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if any(row['True_Label'] not in ('0', '1') for row in rows):
            raise ValueError(f'{split}: invalid label')
        if limit is not None and limit < len(rows):
            # Stratify only for smoke runs so both metrics and BCE see both classes.
            if limit < 2:
                raise ValueError('Sample limit must be at least 2')
            import random
            rng = random.Random(seed)
            groups = [[r for r in rows if r['True_Label'] == label] for label in ('0', '1')]
            if not all(groups):
                raise ValueError('Both classes are required')
            positives = max(1, min(len(groups[1]), limit - 1, round(limit * len(groups[1]) / len(rows))))
            negatives = min(len(groups[0]), limit - positives)
            positives = limit - negatives
            rows = rng.sample(groups[0], negatives) + rng.sample(groups[1], positives)
            rng.shuffle(rows)
        folder = root / split
        candidates = defaultdict(list)
        for path in folder.iterdir():
            if path.is_file():
                candidates[path.name.casefold()].append(path)
        self.samples = []
        for row in rows:
            name = row['Image_name']
            if not name or Path(name).name != name or name in ('.', '..'):
                raise ValueError(f'Invalid image name: {name}')
            path = folder / name
            if not path.is_file():
                matches = candidates[name.casefold()]
                if len(matches) != 1:
                    raise ValueError(f'Missing or ambiguous image: {name}')
                path = matches[0]
            self.samples.append((name, path, int(row['True_Label'])))
        steps = [Letterbox(size)]
        if split == 'train':
            steps.append(transforms.RandomHorizontalFlip())
        steps += [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        return tensor, torch.tensor(float(label)), name
