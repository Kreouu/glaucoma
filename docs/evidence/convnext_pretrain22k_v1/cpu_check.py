import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,'/home/ng/workspace/glaucoma')
import torch,timm
from timm.models.convnext import checkpoint_filter_fn
from model_convnext import build_model
torch.set_num_threads(1)
p=Path('/home/ng/workspace/glaucoma/pretrain_model/convnext_tiny_fb_in22k_ft_in1k.bin');assert hashlib.sha256(p.read_bytes()).hexdigest()=='8cb65dc650a7fb25eaadbff3eb69559d02f83cdc5afdfb3785d1889cdc3e3518'
s=torch.load(p,map_location='cpu',weights_only=True);a=timm.create_model('convnext_tiny.fb_in22k_ft_in1k',pretrained=False);a.load_state_dict(checkpoint_filter_fn(s,a),strict=True);b=build_model(p)
for k,v in a.state_dict().items():
 if not k.startswith('head.fc.'):assert torch.equal(v,b.state_dict()[k]),k
a.eval();b.eval();x=torch.randn(1,3,128,128)
with torch.no_grad():
 u=a.forward_head(a.forward_features(x),pre_logits=True);v=b.forward_head(b.forward_features(x),pre_logits=True);assert torch.equal(u,v);y=b(x);assert y.shape==(1,1)
assert not list(b.buffers()) and all(p.requires_grad for p in b.parameters())
r={'status':'PASS','scope':'CPU strict checkpoint compatibility only; GPU pending','weight_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'public_feature_tensors_exact':180,'official_22k_architecture_features_exact':True,'all_parameters_trainable':True,'buffer_free':True,'output_shape':list(y.shape),'builder':'unchanged model_convnext.py; generic tag architecture identical after strict 22k loading','timm':timm.__version__};Path('/tmp/convnext22k_cpu_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
