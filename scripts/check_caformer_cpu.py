"""Strict source weights and every retained tensor checked before GPU use."""
import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,timm
from timm.models.metaformer import checkpoint_filter_fn
from model_caformer import build_model

torch.set_num_threads(2)
p=ROOT/'pretrain_model/caformer_s18_sail_in22k_ft_in1k.bin'
expected='e165b6b027b2a171ba7af40894b509a87478535ac0e129bdf4e038cedeb5fccc'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
state=torch.load(p,map_location='cpu',weights_only=True)
reference=timm.create_model('caformer_s18.sail_in22k_ft_in1k',pretrained=False).eval()
reference.load_state_dict(checkpoint_filter_fn(state,reference),strict=True)
model=build_model(p).eval();kept={k:v for k,v in reference.state_dict().items() if not k.startswith('head.fc.fc2.')}
for k,v in kept.items():assert torch.equal(model.state_dict()[k],v),k
assert all(p.requires_grad for p in model.parameters()) and not list(model.buffers())
with torch.inference_mode():
 x=torch.randn(2,3,64,64);a=model.forward_features(x);b=reference.forward_features(x);out=model(x)
assert torch.equal(a,b) and out.shape==(2,1) and torch.isfinite(out).all()
r=dict(status='PASS',retained_tensors=len(kept),backbone_max_difference=float((a-b).abs().max()),all_trainable=True,buffers=0,parameters=sum(p.numel() for p in model.parameters()),pretrained_sha256=expected,head_policy='Preserve original MLP fc1 and norm; replace only final fc2')
out=ROOT/'docs/evidence/caformer_v1';out.mkdir(parents=True,exist_ok=True);(out/'cpu_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
