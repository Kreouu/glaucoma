import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,timm
from timm.models.convnext import checkpoint_filter_fn
from model_convnext_base import build_model
torch.set_num_threads(2)
p=ROOT/'pretrain_model/convnext_base_fb_in22k_ft_in1k.bin'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='b4b331a41bbb5e94677e1012ce5a136d413abc3a9cb840945abc81d319253bf1'
state=torch.load(p,map_location='cpu',weights_only=True)
torch.manual_seed(42);model=build_model(p).eval()
filtered=checkpoint_filter_fn(state,timm.create_model('convnext_base.fb_in22k_ft_in1k',pretrained=False))
features={k:v for k,v in filtered.items() if not k.startswith('head.fc.')}
for k,v in features.items():assert torch.equal(model.state_dict()[k],v),k
assert all(p.requires_grad for p in model.parameters()) and len(list(model.buffers()))==0
with torch.inference_mode():out=model(torch.zeros(2,3,64,64))
assert out.shape==(2,1) and torch.isfinite(out).all()
r=dict(status='PASS',feature_tensors=len(features),all_trainable=True,buffers=0,parameters=sum(p.numel() for p in model.parameters()),shape=list(out.shape),pretrained_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='CPU strict full initialization plus every retained feature tensor equality; GPU training/runtime not tested')
ROOT/'docs/evidence/convnext_base22k_v1/cpu_check.json'.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
