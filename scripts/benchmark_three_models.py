"""Same 100 validation images, five warmups, batch one, three timed rounds on local GPU."""
import csv
import gc
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference import Predictor, file_sha256, write_prediction
from scripts.audit_three_models import GROUPS


def main():
    output = ROOT / 'outputs/three_model_comparison_v1/runtime'
    output.mkdir(parents=True, exist_ok=False)
    with (ROOT / 'outputs/convnext_512_w3_v1/seed43/valid_manifest.csv').open() as stream:
        manifest = list(csv.DictReader(stream))
    sample = random.Random(20261029).sample(manifest, 105)
    warmup, measured = sample[:5], sample[5:]
    (output / 'sample.json').write_text(json.dumps(dict(seed=20261029, warmup=warmup, measured=measured), indent=2)+'\n')
    result = dict(device=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                  scope='Local GPU only; not official T4 validation', images=100, warmup=5,
                  rounds=3, batch_size=1, sample_sha256=file_sha256(output/'sample.json'), groups={})
    for family, folder in GROUPS.items():
        checkpoint = ROOT / 'outputs' / folder / 'seed43/best.pt'
        predictor = Predictor(checkpoint)
        image_root = ROOT / predictor.config['data_root'] / 'valid'
        # Use official validation images only; verify every path before timing.
        paths = [image_root / r['actual_filename'] for r in measured]
        warm_paths = [image_root / r['actual_filename'] for r in warmup]
        assert all(p.is_file() for p in paths + warm_paths)
        for path in warm_paths:
            predictor.predict_one(path)
        reference = [predictor.predict_one(p) for p in paths]
        timings = []
        maximum_difference = 0.0
        for repeat in range(1, 4):
            durations = []
            with (output / f'{family}_round{repeat}.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Image_name','Probability','Pred_Label'])
                stream.flush()
                for row, path, expected in zip(measured, paths, reference):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    probability, label = predictor.predict_one(path)
                    write_prediction(writer, stream, row['Image_name'], probability, label)
                    torch.cuda.synchronize()
                    durations.append((time.perf_counter()-started)*1000)
                    maximum_difference = max(maximum_difference, abs(probability-expected[0]))
                    assert label == expected[1] and abs(probability-expected[0]) <= 1e-6
            timings.append(durations)
            print(f'{family} round{repeat} {statistics.mean(durations):.3f}ms', flush=True)
        flat = [v for repeat in timings for v in repeat]
        result['groups'][family] = dict(checkpoint_sha256=file_sha256(checkpoint), seed=43,
            mean_ms=statistics.mean(flat), round_mean_ms=[statistics.mean(r) for r in timings],
            max_ms=max(flat), max_probability_difference=maximum_difference, milliseconds=timings,
            pipeline='JPEG open/draft/RGB/letterbox/normalize/H2D/FP16 forward/FP32 sigmoid/CSV write+flush/synchronize; warm file cache')
        del predictor
        gc.collect()
        torch.cuda.empty_cache()
    (output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
