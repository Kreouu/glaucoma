"""Frozen four ConvNeXt EMA members, equal raw FP32 logit mean, no TTA."""
import json
from pathlib import Path
import torch
from inference import Predictor,file_sha256


class FourModelPredictor:
    def __init__(self,manifest):
        path=Path(manifest);self.manifest=json.loads(path.read_text())
        assert self.manifest['aggregation']=='equal_raw_fp32_logit_mean' and not self.manifest['horizontal_flip']
        root=Path(__file__).resolve().parent
        for name,digest in self.manifest['source_sha256'].items():assert file_sha256(root/name)==digest,name
        assert [(m['id'],m['seed']) for m in self.manifest['members']]==[('wsl42',42),('wsl43',43),('wsl44',44),('ubuntu42',42)]
        self.predictors=[]
        for member in self.manifest['members']:
            p=path.parent/member['file'];assert file_sha256(p)==member['sha256']
            predictor=Predictor(p);c=predictor.config
            assert c['model']=='convnext_tiny' and c['size']==512 and c['seed']==member['seed']
            assert c['weight_type']=='ema' and c['ema_decay']==.999 and not c.get('trim_black',False)
            assert c['lr']==1e-4 and c['pos_weight']==3 and c.get('color_jitter',0)==0
            self.predictors.append(predictor)
        for p in self.predictors:
            for key in ('size','precision','jpeg_draft','preprocessing','threshold','timm_version'):assert p.config[key]==self.predictors[0].config[key],key

    @torch.inference_mode()
    def logits(self,image):
        with torch.autocast('cuda',dtype=self.predictors[0].autocast_dtype):values=torch.stack([p.model(image).flatten() for p in self.predictors])
        return values.float()

    @torch.inference_mode()
    def predict_one(self,path):
        image=self.predictors[0].preprocess(path).unsqueeze(0).cuda()
        probability=self.logits(image).mean(0).sigmoid().item()
        assert 0<=probability<=1
        return probability,int(probability>=.5)
