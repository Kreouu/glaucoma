"""After six-model proof, run a train-only feature-bank screen, then the queued regularizer."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SIX=Path('/home/kreouu/Model_train/glaucoma_ensemble_20261101/outputs/six_model_native_v1')
after_regularization=len(sys.argv)>2 and sys.argv[2]=='--after-regularization'
pid=int(sys.argv[1]);proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    expected=b'run_next_convnext_study.py' if after_regularization else b'run_goal995_ubuntu_postcontrols.py'
    assert expected in proc.read_bytes()
    subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
if after_regularization:
    summary=Path('/home/kreouu/Model_train/glaucoma_next_20261101/outputs/convnext_drop_path_v1/summary.json')
    assert json.loads(summary.read_text())['status']=='PASS'
proof=json.loads((SIX/'full_reload.json').read_text());previous=json.loads((SIX/'verification.json').read_text())
assert proof['status']=='PASS' and proof['images']==5849 and proof['max_probability_difference']==0
if previous['auroc']>=.995 and previous['mean_ms']<100 and previous['model_bytes']<500000000:
    print('STOP: prior verified six-model candidate meets goal',flush=True);sys.exit(0)
subprocess.run([sys.executable,'-u','scripts/verify_train_knn.py'],cwd=ROOT,check=True)
result=json.loads((ROOT/'outputs/train_knn_v1/verification.json').read_text())
if result['goal995_met']:
    print('STOP: verified train-only kNN candidate meets goal',flush=True);sys.exit(0)
if after_regularization:sys.exit(0)
regularizer=Path('/home/kreouu/Model_train/glaucoma_next_20261101')
subprocess.run([sys.executable,'-u','scripts/run_next_convnext_study.py','drop_path'],cwd=regularizer,check=True)
