"""After the live BCE queue exits, complete fixed-LR controls43/44 on the same host."""
import json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    pid=int(sys.argv[1]);p=Path(f'/proc/{pid}/cmdline')
    if p.exists():
        assert b'run_convnext_bce_4060.py' in p.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
    assert (ROOT/'outputs/convnext_bce_4060_v1.exit').read_text().strip()=='0'
    results=[]
    for seed in (43,44):
        run=ROOT/f'outputs/convnext_4060_pair_v1/constant_seed{seed}'
        assert not run.exists()
        cmd=[sys.executable,'-u','train.py','--model','convnext_tiny','--precision','fp16','--data-root','datasets/glaucoma_first','--pretrained-path','pretrain_model/convnext_tiny_fb_in1k.bin','--batch-size','16','--size','512','--jpeg-draft','--workers','4','--lr','0.0001','--lr-schedule','constant','--pos-weight','3','--ema-decay','.999','--epochs','5','--seed',str(seed),'--output-dir',str(run)]
        print('START',seed,flush=True)
        with run.with_suffix('.log').open('x') as f:subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        for kind in ('','ema'):
            subprocess.run([sys.executable,'validate.py','--run-dir',str(run/kind)],cwd=ROOT,check=True)
            report=json.loads((run/kind/'reload_check.json').read_text());assert report['status']=='PASS' and report['images']==5849 and report['max_probability_difference']<=1e-6
            assert hashlib.sha256((run/kind/'best.pt').read_bytes()).hexdigest()==report['checkpoint_sha256']
        results.append(dict(seed=seed,ema=report));print('DONE',json.dumps(results[-1]),flush=True)
    (ROOT/'outputs/convnext_4060_seed_confirmation.json').write_text(json.dumps(dict(status='PASS',results=results),indent=2)+'\n')


if __name__=='__main__':main()
