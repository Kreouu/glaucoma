"""Generate submission CSV in source CSV order using frozen batch=1 inference."""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import PIL
import torch
import torchvision

from inference import Predictor, file_sha256, read_submission, resolve_images, write_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('model/best.pt'))
    parser.add_argument('--input-csv', type=Path, required=True)
    parser.add_argument('--images-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('submission.csv'))
    args = parser.parse_args()
    with args.input_csv.open(encoding='utf-8-sig', newline='') as stream:
        names = [row['Image_name'] for row in csv.DictReader(stream)]
    paths = resolve_images(names, args.images_dir)
    if not names:
        raise ValueError('Empty input CSV')
    predictor = Predictor(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Image_name', 'Probability', 'Pred_Label'])
        for index, (name, path) in enumerate(zip(names, paths), 1):
            probability, label = predictor.predict_one(path)
            write_prediction(writer, stream, name, probability, label)
            if index == 1 or index % 250 == 0 or index == len(names):
                elapsed = time.perf_counter() - started
                print(f'Predict {index}/{len(names)} elapsed={elapsed:.1f}s '
                      f'ETA={elapsed / index * (len(names) - index):.1f}s', flush=True)
    rows = read_submission(args.output)
    if [r['Image_name'] for r in rows] != names:
        raise ValueError('Output order/coverage mismatch')
    if any(int(r['Pred_Label']) != int(float(r['Probability']) >= predictor.threshold) for r in rows):
        raise ValueError('Output labels do not match the declared threshold')
    source = Path(__file__).resolve().parent
    manifest = dict(images=len(rows), batch_size=1, precision='FP16 autocast, FP32 sigmoid',
                    threshold=predictor.threshold, size=predictor.config['size'],
                    jpeg_draft=predictor.config['jpeg_draft'], tta=False,
                    checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                    checkpoint_bytes=args.checkpoint.stat().st_size,
                    input_csv_sha256=file_sha256(args.input_csv), submission_sha256=file_sha256(args.output),
                    source_sha256={n: file_sha256(source / n) for n in ('predict.py', 'cost.py', 'inference.py', 'dataset.py', 'model.py')},
                    gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                    torch=str(torch.__version__), torchvision=str(torchvision.__version__),
                    pillow=PIL.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'PASS: {len(rows)} unique rows; order, probability range and labels checked.', flush=True)


if __name__ == '__main__':
    main()
