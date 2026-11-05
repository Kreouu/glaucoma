"""Run on4060 only after the original installer is confirmed terminal."""
import hashlib,json,subprocess,sys
from pathlib import Path

ROOT=Path('/home/kreouu/Model_train/glaucoma_submission32068_20261105')
PYTHON=ROOT/'verify210/bin/python'
URLS={
 'torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl':('https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl','628e89bd5110ced7debee2a57c69959725b7fbc64eab81a39dd70e46c7e28ba5'),
 'torchvision-0.25.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl':('https://download-r2.pytorch.org/whl/cu128/torchvision-0.25.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl','1255a0ca2bf987acf9f103b96c5c4cfe3415fc4a1eef17fa08af527a04a4f573')}

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 running=subprocess.run(['pgrep','-af','install_verify210_direct.py'],capture_output=True,text=True).stdout.strip()
 if running:
  print('Original installer still active; no recovery started:',running,flush=True);return
 wheelhouse=ROOT/'wheelhouse';wheelhouse.mkdir(exist_ok=True);wheels=[]
 for name,(url,digest) in URLS.items():
  final=wheelhouse/name
  if not final.exists():
   partial=wheelhouse/(name+'.partial')
   subprocess.run(['curl','--fail','--location','--continue-at','-','--retry','5','--retry-delay','10','--output',str(partial),url],check=True)
   assert sha(partial)==digest,('Downloaded file hash mismatch',name)
   partial.rename(final)
  assert sha(final)==digest;wheels.append(str(final))
 subprocess.run([str(PYTHON),'-m','pip','install','--timeout','60','--retries','5',*wheels],check=True)
 subprocess.run([str(PYTHON),'-m','pip','install','--timeout','60','--retries','5','timm==1.0.22','pillow==12.3.0','numpy==2.5.2'],check=True)
 subprocess.run([str(PYTHON),'-c','import torch,torchvision,timm,PIL; assert torch.__version__=="2.10.0+cu128"; assert torchvision.__version__=="0.25.0+cu128"; assert torch.cuda.is_available(); print(torch.ones(1,device="cuda"))'],check=True)
 print('Environment installed and CUDA smoke passed. Inspect existing210 outputs before explicitly launching run_checks210.py; do not overwrite prior outputs.',flush=True)

if __name__=='__main__':main()
