"""Wait for live same-host seed confirmation, then evaluate the fixed six-member ensemble."""
import subprocess,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
pid=int(sys.argv[1]);proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    assert b'run_4060_seed_confirmation.py' in proc.read_bytes()
    subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
summary=Path('/home/kreouu/Model_train/glaucoma_codex_20261031/outputs/convnext_4060_seed_confirmation.json')
assert json.loads(summary.read_text())['status']=='PASS'
subprocess.run([sys.executable,'-u','scripts/verify_six_model.py'],cwd=ROOT,check=True)
