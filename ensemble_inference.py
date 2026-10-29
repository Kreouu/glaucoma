"""Equal-probability ConvNeXt-512 inference with shared preprocessing and optional horizontal flip."""
import json
from pathlib import Path

import torch

from inference import Predictor, file_sha256


class ConvNeXtProbabilityPredictor:
    def __init__(self, manifest_path):
        manifest_path = Path(manifest_path)
        self.manifest = json.loads(manifest_path.read_text())
        assert self.manifest['method'] == 'equal_probability_mean'
        root = Path(__file__).resolve().parent
        for name, digest in self.manifest['source_sha256'].items():
            assert file_sha256(root/name) == digest, name
        self.predictors = []
        for entry in self.manifest['members']:
            checkpoint = manifest_path.parent/entry['file']
            assert file_sha256(checkpoint) == entry['sha256']
            predictor = Predictor(checkpoint)
            assert predictor.config['model'] == 'convnext_tiny' and predictor.config['size'] == 512
            assert predictor.config['seed'] == entry['seed']
            self.predictors.append(predictor)
        assert [p.config['seed'] for p in self.predictors] in ([43], [42,43,44])
        for predictor in self.predictors:
            for key in ('size','precision','jpeg_draft','preprocessing','threshold','timm_version'):
                assert predictor.config[key] == self.predictors[0].config[key], key
        self.horizontal_flip = self.manifest['horizontal_flip']

    @torch.inference_mode()
    def probabilities(self, image):
        with torch.autocast('cuda',dtype=self.predictors[0].autocast_dtype):
            logits = torch.stack([p.model(image).flatten() for p in self.predictors])
        return logits.float().sigmoid()

    @torch.inference_mode()
    def predict_one(self, path):
        image = self.predictors[0].preprocess(path).unsqueeze(0).cuda()
        probabilities = self.probabilities(image)
        if self.horizontal_flip:
            probabilities = (probabilities+self.probabilities(image.flip(-1)))/2
        probability = probabilities.mean().item()
        assert 0 <= probability <= 1
        return probability, int(probability >= .5)
