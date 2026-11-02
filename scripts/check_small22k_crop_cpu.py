"""Strict Small22K initialization and official feature equivalence, CPU only."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,timm
from timm.models.convnext import checkpoint_filter_fn
from model_convnext_small import build_model
torch.set_num_threads(2)
p=ROOT/'pretrain_model/convnext_small_fb_in22k_ft_in1k.bin';h=hashlib.sha256(p.read_bytes()).hexdigest();assert h=='c912f45cd738512978a900cc8aefade65bc44e507321f0144378a2d98aa5e914'
state=torch.load(p,map_location='cpu',weights_only=True)
torch.manual_seed(42);model=build_model(p).eval();official=timm.create_model('convnext_small.fb_in22k_ft_in1k',pretrained=False).eval();filtered=checkpoint_filter_fn(state,official);official.load_state_dict(filtered,strict=True)
features={k:v for k,v in filtered.items() if not k.startswith('head.fc.')}
for k,v in features.items():assert torch.equal(model.state_dict()[k],v),k
assert all(p.requires_grad for p in model.parameters()) and not list(model.buffers())
with torch.inference_mode():
 x=torch.randn(2,3,64,64);a=model.forward_features(x);b=official.forward_features(x);difference=float((a-b).abs().max());assert difference==0
 out=model(x);assert out.shape==(2,1) and torch.isfinite(out).all()
r=dict(status='PASS',feature_tensors=len(features),all_trainable=True,buffers=0,parameters=sum(p.numel() for p in model.parameters()),shape=list(out.shape),official_feature_max_difference=difference,pretrained_sha256=h,scope='Strict initialization, every retained feature tensor equality and official architecture forward_features equality; no GPU runtime claim')
(ROOT/'docs/evidence/convnext_small22k_crop_v1/cpu_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
