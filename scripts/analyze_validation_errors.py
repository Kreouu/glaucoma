"""Descriptive validation-only quality audit; does not fit or change predictions."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


def quality(path):
    with Image.open(path) as image:
        width, height = image.size
        image.draft('RGB', (256, 256))
        image.thumbnail((256, 256))
        rgb = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    foreground = rgb.max(axis=2) > .08
    gray = rgb.mean(axis=2)
    values = gray[foreground]
    # A descriptive gradient proxy, not a calibrated clinical image-quality score.
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:] - gray[:-1])
    mx = foreground[:, 1:] & foreground[:, :-1]
    my = foreground[1:] & foreground[:-1]
    return dict(width=width, height=height, dark_fraction=float(1-foreground.mean()),
                brightness=float(values.mean()) if values.size else 0.,
                contrast=float(values.std()) if values.size else 0.,
                gradient=float((dx[mx].mean() + dy[my].mean()) / 2)
                if mx.any() and my.any() else 0.)


def metrics(rows):
    positive = np.array([r['probability'] for r in rows if r['label'] == 1])
    negative = np.array([r['probability'] for r in rows if r['label'] == 0])
    negative.sort()
    if len(positive) and len(negative):
        below = np.searchsorted(negative, positive, side='left')
        equal = np.searchsorted(negative, positive, side='right') - below
        auc = float((below + equal / 2).sum() / (len(positive)*len(negative)))
    else:
        auc = None
    return dict(images=len(rows), positives=len(positive), auroc=auc,
                false_negatives_at_05=int((positive < .5).sum()),
                false_positives_at_05=int((negative >= .5).sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.run_dir/'valid_predictions.csv'] + [
        args.run_dir/f'seed{s}_valid_predictions.csv' for s in (42, 43, 44)]
    tables = [list(csv.DictReader(p.open())) for p in paths]
    official = list(csv.DictReader((args.data_root/'valid.csv').open()))
    assert len(official) == 5849 and sum(int(r['True_Label']) for r in official) == 191
    keys = [(r['Image_name'], r['True_Label']) for r in official]
    assert len(set(name for name, _ in keys)) == len(keys)
    for table in tables:
        assert [(r['Image_name'], r['True_Label']) for r in table] == keys
    probabilities = np.array([[float(r['Probability']) for r in t] for t in tables])
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    with ThreadPoolExecutor(max_workers=4) as pool:
        qualities = list(pool.map(quality, [args.data_root/'valid'/n for n, _ in keys]))
    rows = [dict(image_name=name, label=int(label), probability=float(probabilities[0, i]),
                 seed_probability_std=float(probabilities[1:, i].std()), **qualities[i])
            for i, (name, label) in enumerate(keys)]
    overall = metrics(rows)
    verification = json.loads((args.run_dir/'verification.json').read_text())
    assert verification['status'] == 'PASS' and verification['images'] == len(rows)
    assert abs(overall['auroc'] - verification['auroc']) < 1e-12
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir/'image_quality.csv').open('x') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for feature in ('dark_fraction', 'brightness', 'contrast', 'gradient', 'seed_probability_std'):
        cuts = np.quantile([r[feature] for r in rows], [.25, .5, .75])
        bins = np.searchsorted(cuts, [r[feature] for r in rows], side='right')
        groups[feature] = dict(cuts=cuts.tolist(), quartiles=[
            metrics([r for i, r in enumerate(rows) if bins[i] == q]) for q in range(4)])
    for label, reverse, title in ((1, False, 'lowest_positive'), (0, True, 'highest_negative')):
        selected = sorted([r for r in rows if r['label'] == label],
                          key=lambda r: r['probability'], reverse=reverse)[:24]
        sheet = Image.new('RGB', (6*210, 4*235), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected):
            x, y = (i % 6)*210, (i // 6)*235
            with Image.open(args.data_root/'valid'/row['image_name']) as image:
                image.draft('RGB', (210, 210))
                thumb = ImageOps.contain(image.convert('RGB'), (206, 200))
            sheet.paste(thumb, (x, y))
            draw.text((x+2, y+202), row['image_name'], fill='black')
            draw.text((x+2, y+217), f"p={row['probability']:.4f} sd={row['seed_probability_std']:.3f}", fill='black')
        sheet.save(args.output_dir/f'{title}.jpg')
    summary = dict(scope='Descriptive fixed validation audit; no test use, no label changes or parameter fitting.',
                   caveat='Quartile AUCs are descriptive, have different case mixes and few positives; no causal claims.',
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                   overall=overall, groups=groups)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(overall), flush=True)


if __name__ == '__main__':
    main()
