import sys,json,hashlib,inspect
from pathlib import Path
import torch,timm
sys.path.insert(0,'/tmp')
from glaucoma_model_dinov2 import build_model
torch.set_num_threads(1)
p=Path('/home/kreouu/Model_train/glaucoma_codex_20261031/pretrain_model/dinov2_vits14_pretrain.pth')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
s=torch.load(p,map_location='cpu',weights_only=True);model=build_model(p);loaded=model.state_dict()
assert set(s)-{'mask_token'}==set(loaded)-{'head.weight','head.bias'}
for k,v in s.items():
 if k!='mask_token': assert torch.equal(v,loaded[k]),k
assert all(p.requires_grad for p in model.parameters())
model.eval()
with torch.no_grad():
 y=model(torch.zeros(1,3,56,56));assert y.shape==(1,1) and torch.isfinite(y).all()
assert model.patch_embed.patch_size==(14,14)
r={'status':'PASS','scope':'CPU strict official feature loading and small dynamic-size forward only; not GPU training or target-resolution performance','timm':timm.__version__,'torch':torch.__version__,'official_weight_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'official_tensors_identical_except_unused_mask_token':len(s)-1,'removed_tensor':'mask_token (used for masked pretraining, absent from classification encoder)','all_parameters_trainable':True,'parameters':sum(p.numel() for p in model.parameters()),'positional_embedding_shape':list(model.pos_embed.shape),'dynamic_forward_size':56,'output_shape':list(y.shape),'builder_sha256':hashlib.sha256(Path('/tmp/glaucoma_model_dinov2.py').read_bytes()).hexdigest()}
Path('/tmp/dinov2_cpu_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
