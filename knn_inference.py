"""Frozen train-only feature bank; no test adaptation, filename lookup or label overrides."""
import json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from four_model_inference import FourModelPredictor
from inference import file_sha256


class KNNPredictor:
    def __init__(self,recipe):
        path=Path(recipe);self.recipe=json.loads(path.read_text());root=Path(__file__).resolve().parent
        assert self.recipe['mode'] in ('knn','mixed')
        assert file_sha256(root/'knn_inference.py')==self.recipe['source_sha256']
        assert file_sha256(path.parent/self.recipe['weights_manifest'])==self.recipe['weights_manifest_sha256']
        self.loader=FourModelPredictor(path.parent/self.recipe['weights_manifest'])
        bank_path=path.parent/self.recipe['index'];assert file_sha256(bank_path)==self.recipe['index_sha256']
        bank=torch.load(bank_path,map_location='cpu',weights_only=True);c=bank['config']
        assert c['split']=='train' and c['rows']==51396 and c['train_csv_sha256']=='ec81a23ba7758eccdd09dba027ce1a50998d81c18ed30ab6d2d608b3397f67d7'
        assert c['feature_model_sha256']==self.loader.manifest['members'][3]['sha256']
        assert bank['vectors'].shape==(51396,768) and bank['labels'].shape==(51396,)
        self.vectors=F.normalize(bank['vectors'].float().cuda(),dim=1);self.labels=bank['labels'].float().cuda()
        self.prior=float(bank['labels'].float().mean());assert 0<self.prior<1
        torch.backends.cuda.matmul.allow_tf32=False

    @torch.inference_mode()
    def evaluate_tensor(self,image,check_anchor=False):
        models=self.loader.predictors;values=[];feature=None;anchor_diff=0.
        with torch.autocast('cuda',dtype=models[0].autocast_dtype):
            for i,p in enumerate(models):
                if i==3:
                    feature=p.model.forward_head(p.model.forward_features(image),pre_logits=True)
                    value=p.model.get_classifier()(feature).flatten()
                    if check_anchor:anchor_diff=float((value-p.model(image).flatten()).abs().max())
                else:value=p.model(image).flatten()
                values.append(value)
        logits=torch.stack(values).float();query=F.normalize(feature.float(),dim=1)
        similarity=(query@self.vectors.T).flatten();scores,indices=similarity.topk(self.recipe['k'],largest=True,sorted=True)
        weights=(scores/self.recipe['temperature']).softmax(0);posterior=(weights*self.labels[indices]).sum()
        eps=self.recipe['prior_smoothing'];posterior=posterior*(1-eps)+eps*self.prior
        knn_logit=torch.logit(posterior)+math.log(self.recipe['loss_pos_weight']);base=logits.mean()
        return dict(baseline=base.sigmoid().item(),knn=knn_logit.sigmoid().item(),mixed=((4*base+knn_logit)/5).sigmoid().item(),anchor_logit_difference=anchor_diff)

    @torch.inference_mode()
    def predict_one(self,path):
        image=self.loader.predictors[0].preprocess(path).unsqueeze(0).cuda()
        value=self.evaluate_tensor(image)[self.recipe['mode']]
        assert 0<=value<=1
        return value,int(value>=.5)
