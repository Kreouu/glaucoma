"""Fixed EMA ensemble recipes: probability/logit mean, optionally horizontal-flip averaging."""
import json
from pathlib import Path

import torch
from ema_variant_inference import EMAVariantPredictor

from ensemble_inference import ConvNeXtProbabilityPredictor
from inference import file_sha256


class EMA9995Predictor(EMAVariantPredictor):
    def __init__(self,recipe_path):
        path=Path(recipe_path);self.recipe=json.loads(path.read_text())
        assert self.recipe['aggregation'] in ('probability','logit')
        assert file_sha256(Path(__file__))==self.recipe['source_sha256']
        weights=path.parent/self.recipe['weights_manifest']
        assert file_sha256(weights)==self.recipe['weights_manifest_sha256']
        self.loader=ConvNeXtProbabilityPredictor(weights)
        assert not self.loader.horizontal_flip
        assert all(p.config['weight_type']=='ema' and p.config['ema_decay']==.9995 for p in self.loader.predictors)

