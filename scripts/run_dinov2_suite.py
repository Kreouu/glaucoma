"""Screen generic self-supervised features after the existing Small queue finishes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pid = int(sys.argv[1])
    proc = Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        assert b'run_next_convnext_study.py' in proc.read_bytes()
        subprocess.run(['/usr/bin/python3', '-c', 'import os,select,sys\ntry: fd=os.pidfd_open(int(sys.argv[1]))\nexcept ProcessLookupError: sys.exit(0)\np=select.poll();p.register(fd,select.POLLIN);p.poll();os.close(fd)', str(pid)], check=True)
    previous = Path('/home/ng/workspace/glaucoma_next_20261101/outputs/convnext_small_v1/summary.json')
    assert json.loads(previous.read_text())['status'] == 'PASS'
    plan = json.loads((ROOT/'docs/evidence/dinov2_small_v1/plan.json').read_text())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    assert sha(ROOT/plan['pretrained_path']) == plan['pretrained_sha256']
    for split, digest in plan['csv_sha256'].items():
        assert sha(ROOT/f'datasets/glaucoma_first/{split}.csv') == digest
    out = ROOT/'outputs/dinov2_small_v1'
    out.mkdir(exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    results = []

    def run(name, lr, seed, smoke=False):
        path = out/name
        command = [sys.executable, '-u', 'train.py', '--model', 'dinov2_small', '--precision', 'fp16', '--data-root', 'datasets/glaucoma_first', '--pretrained-path', plan['pretrained_path'], '--batch-size', '16', '--size', '392', '--jpeg-draft', '--workers', '4', '--lr', str(lr), '--lr-schedule', 'warmup_cosine', '--warmup-epochs', '1', '--min-lr', str(lr/10), '--pos-weight', '3', '--epochs', '2' if smoke else '8', '--seed', str(seed), '--output-dir', str(path)]
        if smoke:
            command += ['--train-limit', '64', '--valid-limit', '64']
        print('START', name, flush=True)
        with path.with_name(path.name+'.log').open('x') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([sys.executable, 'validate.py', '--run-dir', str(path)], cwd=ROOT, check=True)
        check = json.loads((path/'reload_check.json').read_text())
        assert check['status'] == 'PASS' and check['max_probability_difference'] <= 1e-6
        assert check['images'] == (64 if smoke else 5849)
        assert sha(path/'best.pt') == check['checkpoint_sha256']
        entry = dict(name=name, lr=lr, seed=seed, check=check)
        print('DONE', json.dumps(entry), flush=True)
        return entry

    run('smoke', 1e-5, 42, True)
    for name, lr in [('lr1e5_seed42', 1e-5), ('lr5e5_seed42', 5e-5)]:
        results.append(run(name, lr, 42))
        (out/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
    best = max(results, key=lambda x: x['check']['val_auroc'])
    if best['check']['val_auroc'] >= .990:
        for seed in (43, 44):
            results.append(run(f'confirm_seed{seed}', best['lr'], seed))
            (out/'progress.json').write_text(json.dumps(results, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(dict(status='PASS', results=results, test_used=False, scope='Generic official DINOv2; full encoder fine-tuning; native deployment verification still required'), indent=2)+'\n')


if __name__ == '__main__':
    main()
