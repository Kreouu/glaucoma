"""Fixed EMA ensemble recipes: probability/logit mean, optionally horizontal-flip averaging."""
import json
from pathlib import Path

import torch

from ensemble_inference import ConvNeXtProbabilityPredictor
from inference import file_sha256


class EMAVariantPredictor:
    def __init__(self,recipe_path):
        path=Path(recipe_path);self.recipe=json.loads(path.read_text())
        assert self.recipe['aggregation'] in ('probability','logit')
        assert file_sha256(Path(__file__))==self.recipe['source_sha256']
        weights=path.parent/self.recipe['weights_manifest']
        assert file_sha256(weights)==self.recipe['weights_manifest_sha256']
        self.loader=ConvNeXtProbabilityPredictor(weights)
        assert not self.loader.horizontal_flip
        assert all(p.config['weight_type']=='ema' and p.config['ema_decay']==.999 for p in self.loader.predictors)

    @torch.inference_mode()
    def logits(self,image):
        with torch.autocast('cuda',dtype=self.loader.predictors[0].autocast_dtype):
            values=torch.stack([p.model(image).flatten() for p in self.loader.predictors])
        return values.float()

    @staticmethod
    def aggregate(logits,aggregation):
        return logits.mean(dim=0).sigmoid() if aggregation=='logit' else logits.sigmoid().mean(dim=0)

    @torch.inference_mode()
    def predict_one(self,path):
        image=self.loader.predictors[0].preprocess(path).unsqueeze(0).cuda()
        logits=self.logits(image)
        if self.recipe['horizontal_flip']:logits=torch.cat([logits,self.logits(image.flip(-1))],dim=0)
        probability=self.aggregate(logits,self.recipe['aggregation']).item()
        assert 0<=probability<=1
        return probability,int(probability>=.5)
