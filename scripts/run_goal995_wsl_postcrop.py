"""Exclusive post-crop GPU pipeline: four-member audit, then queued new architecture."""
import subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
pid=int(sys.argv[1]);proc=Path(f'/proc/{pid}/cmdline')
if proc.exists():
    assert b'run_convnext_trim.py' in proc.read_bytes()
    subprocess.run(['/usr/bin/python3','-c','import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)',str(pid)],check=True)
assert Path('/home/ng/workspace/glaucoma/outputs/convnext_trim_v1.exit').read_text().strip()=='0'
subprocess.run([sys.executable,'-u','scripts/verify_four_model.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-u','scripts/run_efficientnetv2_suite.py',str(pid)],cwd=ROOT,check=True)
