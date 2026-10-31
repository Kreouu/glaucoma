"""Six fixed EMA checkpoints from two training environments; shared no-TTA logit mean."""
import json
from pathlib import Path
from inference import Predictor,file_sha256
from four_model_inference import FourModelPredictor


class SixModelPredictor(FourModelPredictor):
    def __init__(self,manifest):
        path=Path(manifest);self.manifest=json.loads(path.read_text())
        assert self.manifest['aggregation']=='equal_raw_fp32_logit_mean' and not self.manifest['horizontal_flip']
        root=Path(__file__).resolve().parent
        for name,digest in self.manifest['source_sha256'].items():assert file_sha256(root/name)==digest,name
        expected=[('wsl42',42),('wsl43',43),('wsl44',44),('ubuntu42',42),('ubuntu43',43),('ubuntu44',44)]
        assert [(m['id'],m['seed']) for m in self.manifest['members']]==expected
        self.predictors=[]
        for member in self.manifest['members']:
            checkpoint=path.parent/member['file'];assert file_sha256(checkpoint)==member['sha256']
            predictor=Predictor(checkpoint);c=predictor.config
            assert c['model']=='convnext_tiny' and c['size']==512 and c['seed']==member['seed']
            assert c['weight_type']=='ema' and c['ema_decay']==.999 and not c.get('trim_black',False)
            assert c['precision']=='fp16' and c['lr']==1e-4 and c['lr_schedule']=='constant' and c['pos_weight']==3 and c.get('color_jitter',0)==0
            self.predictors.append(predictor)
        for p in self.predictors:
            for key in ('size','precision','jpeg_draft','preprocessing','threshold','timm_version','csv_sha256'):assert p.config[key]==self.predictors[0].config[key],key
