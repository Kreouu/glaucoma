"""Wait for the existing training job, then run fixed validation and conditional release."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def wait_for_pid(pid):
    # The conda build lacks os.pidfd_open; Ubuntu's system Python exposes it.
    helper="""import os,select,sys
try: fd=os.pidfd_open(int(sys.argv[1]))
except ProcessLookupError: sys.exit(0)
poll=select.poll();poll.register(fd,select.POLLIN);poll.poll();os.close(fd)
"""
    subprocess.run(['/usr/bin/python3','-c',helper,str(pid)],check=True)


def main():
    plan=json.loads((ROOT/'docs/evidence/convnext_512_ema8_v1/post_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    pid=plan['training_wrapper_pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert 'run_convnext_ema8.py' in proc.read_text()
        wait_for_pid(pid)
    assert (ROOT/'outputs/convnext_512_ema8_v1.exit').read_text().strip()=='0'
    def run(*args):subprocess.run([sys.executable,'-u',*args],cwd=ROOT,check=True)
    run('scripts/verify_ema_variants.py','--epochs-study')
    output=ROOT/'outputs/convnext_512_ema8_inference_v1'
    report=json.loads((output/'verification.json').read_text());entry=report['candidates']['logit']
    if not entry['promising']:
        result=dict(status='NO_GAIN',metrics=entry,next='Continue the active goal with another single-factor study; no test inference')
    else:
        recipe=output/'logit.json'
        selection=dict(selected='logit.json',recipe_sha256=hashlib.sha256(recipe.read_bytes()).hexdigest(),
            rule='Fixed no-TTA equal-logit mean; >=0.0003 AUROC gain vs frozen LR6e-5 five-epoch candidate, local latency <100ms',metrics=entry,test_used_for_selection=False,platform_uploaded=False)
        (output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
        run('scripts/reload_ema_variant.py','--recipe',str(recipe))
        run('scripts/predict_ema_variant.py','--recipe',str(recipe),'--names-csv','datasets/glaucoma_first/test.csv',
            '--images','datasets/glaucoma_first/test','--output','outputs/convnext_ema8_logit_submission/submission.csv')
        result=dict(status='CANDIDATE_READY',metrics=entry,next='Independently audit CSV, portable inference, backup and synchronization before goal completion')
    (output/'post_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
