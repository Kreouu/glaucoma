import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/ng/workspace/glaucoma')
sys.path.insert(0,'/tmp')
import torch
from model_convnext import build_model as baseline
from model_convnext_attention import build_model,SpatialPool
torch.set_num_threads(1)
path='/home/ng/workspace/glaucoma/pretrain_model/convnext_tiny_fb_in1k.bin'
torch.manual_seed(42);a=baseline(path).eval()
torch.manual_seed(42);b=build_model(path).eval()
assert set(b.state_dict())-set(a.state_dict())=={'head.global_pool.score.weight'}
for k,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[k]),k
x=torch.randn(1,3,128,128)
with torch.no_grad():
 u=a(x);v=b(x)
 torch.testing.assert_close(u,v,rtol=1e-6,atol=1e-6)
pool=SpatialPool(3);f=torch.randn(2,3,4,4,requires_grad=True);out=pool(f)
torch.testing.assert_close(out,f.mean((2,3),keepdim=True),rtol=1e-6,atol=1e-6)
(out[:,0].sum()).backward();g=pool.score.weight.grad
assert torch.isfinite(g).all() and g.abs().sum()>0
assert not list(b.buffers()) and all(p.requires_grad for p in b.parameters())
r={'status':'PASS','scope':'CPU initialization, preserved backbone/head, spatial gradient; GPU pending','unchanged_tensors':len(a.state_dict()),'added_parameters':b.head.global_pool.score.weight.numel(),'initial_logit_max_difference':float((u-v).abs().max()),'pool_matches_average_at_initialization':True,'attention_gradient_nonzero_finite':True,'all_parameters_trainable':True,'buffer_free':True,'builder_sha256':hashlib.sha256(Path('/tmp/model_convnext_attention.py').read_bytes()).hexdigest()}
Path('/tmp/attention_cpu_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
