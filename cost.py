"""100 images, batch=1, three rounds; reads/writes included in synchronized timing."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from inference import Predictor, file_sha256, image_paths, read_submission, write_prediction


def consistency(reference, actual, labels):
    reference = np.asarray(reference)
    actual = np.asarray(actual)
    old_order = np.sign(reference[:, None] - reference[None, :])
    new_order = np.sign(actual[:, None] - actual[None, :])
    ranking_changed = np.any(old_order != new_order, axis=1)
    error = np.abs(reference - actual)
    mismatch = ranking_changed | (error > 1e-3) | np.asarray(labels, dtype=bool)
    return mismatch, error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission', type=Path)
    parser.add_argument('pictures_100', type=Path)
    parser.add_argument('pictures_5', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('model/best.pt'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/cost'))
    args = parser.parse_args()
    pictures, warmup = image_paths(args.pictures_100), image_paths(args.pictures_5)
    if len(pictures) != 100 or len(warmup) != 5:
        parser.error('Exactly 100 measurement images and 5 warmup images are required')
    keys = [p.name.casefold() for p in pictures]
    if len(set(keys)) != 100 or set(keys) & {p.name.casefold() for p in warmup}:
        parser.error('Measurement names must be unique and disjoint from warmup names')
    submission = read_submission(args.submission)
    lookup = {r['Image_name'].casefold(): r for r in submission}
    if any(key not in lookup for key in keys):
        parser.error('Measurement images must all occur in submission')
    reference = [lookup[key] for key in keys]
    predictor = Predictor(args.checkpoint)
    manifest_path = args.submission.with_suffix('.manifest.json')
    if not manifest_path.is_file():
        parser.error('Prediction manifest is required alongside the submission')
    manifest = json.loads(manifest_path.read_text())
    if manifest['checkpoint_sha256'] != file_sha256(args.checkpoint) or manifest['submission_sha256'] != file_sha256(args.submission):
        parser.error('Submission/checkpoint hash differs from its prediction manifest')
    source = Path(__file__).resolve().parent
    for name in ('predict.py', 'cost.py', 'inference.py', 'dataset.py', 'model.py'):
        if file_sha256(source / name) != manifest['source_sha256'][name]:
            parser.error(f'Inference source differs from prediction manifest: {name}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for path in warmup:
        predictor.predict_one(path)
    torch.cuda.synchronize()
    timings, mismatches, max_error = [], np.zeros(100, dtype=bool), 0.0
    for repeat in range(3):
        probabilities, label_changes, durations = [], [], []
        with (args.output_dir / f'predictions_round_{repeat + 1}.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
            stream.flush()
            for path, row in zip(pictures, reference):
                torch.cuda.synchronize()
                start = time.perf_counter()
                probability, label = predictor.predict_one(path)
                write_prediction(writer, stream, row['Image_name'], probability, label)
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - start) * 1000)
                probabilities.append(probability)
                label_changes.append(label != int(row['Pred_Label']))
        changed, error = consistency([float(r['Probability']) for r in reference], probabilities, label_changes)
        mismatches |= changed
        max_error = max(max_error, float(error.max()))
        timings.append(durations)
        print(f'Round {repeat + 1}/3 mean={np.mean(durations):.3f}ms inconsistent={int(changed.sum())}', flush=True)
    means = np.mean(timings, axis=0)
    average = float(means.mean())
    result = dict(images=100, warmup_images=5, rounds=3, batch_size=1,
                  precision='FP16 autocast, FP32 sigmoid', threshold=predictor.threshold,
                  mean_ms=average, inconsistent_images=int(mismatches.sum()), max_probability_difference=max_error,
                  consistency_pass=not bool(mismatches.any()), within_100ms_on_this_machine=average <= 100,
                  gpu=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  checkpoint_sha256=file_sha256(args.checkpoint), submission_sha256=file_sha256(args.submission),
                  sample_scope='User-supplied timing images; official sample provenance must be documented separately',
                  note='This report does not certify official T4 compliance.')
    (args.output_dir / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output_dir / 'timings.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Image_name', 'round_1_ms', 'round_2_ms', 'round_3_ms', 'mean_ms'])
        for index, row in enumerate(reference):
            writer.writerow([row['Image_name'], *(t[index] for t in timings), means[index]])
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['consistency_pass'] and average <= 100 else 1


if __name__ == '__main__':
    raise SystemExit(main())
