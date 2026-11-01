"""Exclusive native verification after both frozen 22K confirmation seeds finish."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    pid=10569;proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert b'confirm_22k_complement.py' in proc.read_bytes()
        subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
    previous=Path('/home/ng/workspace/glaucoma_22k_confirm_20261101/outputs/convnext22k_confirmation_v1/summary.json')
    report=json.loads(previous.read_text());assert report['status'] in ('PASS','SKIP')
    frozen=json.loads(Path('/home/ng/workspace/glaucoma_five22k_20261101/outputs/five_model_22k_native_v1/verification.json').read_text())
    if report['status']=='SKIP' or frozen['goal995_met']:
        (ROOT/'outputs/seven_queue_result.json').write_text(json.dumps(dict(status='SKIP',reason='Confirmation skipped or prior verified goal already met'))+'\n');return
    subprocess.run([sys.executable,'-u','scripts/verify_seven_model_22k.py'],cwd=ROOT,check=True)


if __name__=='__main__':main()
