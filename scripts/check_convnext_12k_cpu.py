"""Prove the existing Tiny builder retains every 12K feature parameter and output."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,timm
from model_convnext import build_model
from timm.models.convnext import checkpoint_filter_fn

torch.set_num_threads(2)
p=ROOT/'pretrain_model/convnext_tiny_in12k_ft_in1k.bin'
meta=json.loads((ROOT/'licenses/convnext_tiny_12k/source.json').read_text())
assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
state=torch.load(p,map_location='cpu',weights_only=True)
reference=timm.create_model('convnext_tiny.in12k_ft_in1k',pretrained=False).eval()
reference.load_state_dict(checkpoint_filter_fn(state,reference),strict=True)
model=build_model(p).eval();features={k:v for k,v in reference.state_dict().items() if not k.startswith('head.fc.')}
for k,v in features.items():assert torch.equal(v,model.state_dict()[k]),k
assert all(p.requires_grad for p in model.parameters()) and not list(model.buffers())
with torch.inference_mode():
 x=torch.randn(2,3,64,64);a=model.forward_head(model.forward_features(x),pre_logits=True);b=reference.forward_head(reference.forward_features(x),pre_logits=True)
assert torch.equal(a,b)
r=dict(status='PASS',feature_tensors=len(features),features_max_difference=float((a-b).abs().max()),parameters=sum(p.numel() for p in model.parameters()),all_trainable=True,buffers=0,pretrained_sha256=meta['sha256'])
out=ROOT/'docs/evidence/convnext_pretrain12k_v1';out.mkdir(parents=True,exist_ok=True);(out/'cpu_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
