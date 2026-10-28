"""Short training-throughput benchmark; no model checkpoints or test data used."""

import argparse
import gc
import json
import random
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from dataset import FundusDataset
from model import build_model
from train import seed_worker, sha256


def trial(dataset, indices, workers, warmup, steps):
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = build_model(ROOT / 'pretrain_model/efficientnet_b0_rwightman-7f5810bc.pth').cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3., device='cuda'))
    loader = iterator = None
    if workers is not None:
        loader = DataLoader(Subset(dataset, indices), batch_size=16, shuffle=False,
                            num_workers=workers, pin_memory=True, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(42))
        iterator = iter(loader)
    else:
        resident_images = torch.randn(16, 3, 256, 256, device='cuda')
        resident_labels = (torch.arange(16, device='cuda') % 2).float()

    def batch():
        if iterator is None:
            return resident_images, resident_labels
        images, labels, _ = next(iterator)
        return images, labels

    def update(images, labels, start_event=None, end_event=None):
        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if start_event is not None:
            start_event.record()
        with torch.autocast('cuda', dtype=torch.float16):
            loss = criterion(model(images).flatten(), labels)
        # Keep the scalar synchronizations used by train.py.
        if not torch.isfinite(loss).item():
            raise RuntimeError('Non-finite benchmark loss')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        value = loss.item()
        if end_event is not None:
            end_event.record()
        return value

    for _ in range(warmup):
        update(*batch())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(steps)]
    waits = []
    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    started = time.perf_counter()
    for start_event, end_event in events:
        wait_start = time.perf_counter()
        images, labels = batch()
        waits.append(time.perf_counter() - wait_start)
        update(images, labels, start_event, end_event)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(workers=workers, kind='real_images' if workers is not None else 'GPU_resident_synthetic',
                  measured_batches=steps, images=16 * steps, wall_seconds=elapsed,
                  images_per_second=16 * steps / elapsed,
                  data_wait_ms=sum(waits) * 1000 / steps,
                  data_wait_fraction=sum(waits) / elapsed,
                  cuda_phase_ms=sum(a.elapsed_time(b) for a, b in events) / steps,
                  main_cpu_seconds=(cpu_end.ru_utime + cpu_end.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime),
                  peak_vram_mib=torch.cuda.max_memory_allocated() / 1024**2)
    if iterator is not None:
        assert next(iterator, None) is None  # Exhaust the iterator so workers exit before the next trial.
    del model, optimizer, scaler, iterator, loader
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error('warmup and steps must be positive')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset = FundusDataset(ROOT / 'datasets/glaucoma_first', 'train', size=256, jpeg_draft=True)
    indices = random.Random(42).sample(range(len(dataset)), 16 * (args.warmup + args.steps))
    sources = {name: sha256(ROOT / name) for name in ('dataset.py', 'model.py', 'train.py', 'scripts/benchmark_training.py')}
    metadata = dict(batch_size=16, size=256, pos_weight=3, warmup_batches=args.warmup,
                    steps=args.steps, order=[None, 4, 6, 8, 8, 6, 4, None],
                    gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__), cuda=torch.version.cuda,
                    source_sha256=sources, jpeg_draft=True, pin_memory=True,
                    prefetch_factor=2, persistent_workers=False,
                    note='Fixed training-image order; worker-dependent flip draws may differ. '
                         'CUDA event phase includes host submission gaps, not pure kernel execution. '
                         'GPU-resident synthetic trials exclude loading/transfers but retain Python/optimizer/scalar sync overhead.')
    (args.output_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output_dir / 'training_images.json').write_text(json.dumps([dataset.samples[i][0] for i in indices]) + '\n')
    rows = []
    for index, workers in enumerate(metadata['order'], 1):
        print(f'Trial {index}/8 workers={workers} starting', flush=True)
        result = trial(dataset, indices, workers, args.warmup, args.steps)
        rows.append(result)
        with (args.output_dir / 'trials.jsonl').open('a') as stream:
            stream.write(json.dumps(result) + '\n')
        print(json.dumps(result), flush=True)
    summary = []
    for workers in (4, 6, 8, None):
        selected = [r for r in rows if r['workers'] == workers]
        summary.append(dict(workers=workers, images_per_second=sum(r['images'] for r in selected) / sum(r['wall_seconds'] for r in selected),
                            data_wait_ms=sum(r['data_wait_ms'] for r in selected) / 2,
                            cuda_phase_ms=sum(r['cuda_phase_ms'] for r in selected) / 2,
                            individual_images_per_second=[r['images_per_second'] for r in selected]))
    assert all(sha256(ROOT / name) == digest for name, digest in sources.items())
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('COMPLETE ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
