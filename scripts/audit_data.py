"""Read-only audit of official train/valid data; never opens the test split."""

import argparse
import csv
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def inspect_image(path):
    result = {}
    try:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        result['file_sha256'] = digest.hexdigest()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                rgb = image.convert('RGB')
                result.update(width=rgb.width, height=rgb.height, mode=image.mode)
                pixels = hashlib.sha256()
                pixels.update(f'{rgb.width}x{rgb.height}:RGB:'.encode())
                pixels.update(rgb.tobytes())
                result['pixel_sha256'] = pixels.hexdigest()
            result['warnings'] = [str(w.message) for w in caught]
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/data_audit'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.data_root.resolve()
    issues, records, summaries = [], [], {}

    def issue(split, name, kind, detail=''):
        issues.append(dict(split=split, image_name=name, kind=kind, detail=detail))

    for split in ('train', 'valid'):
        csv_path = root / f'{split}.csv'
        folder = root / split
        if not folder.is_dir():
            parser.error(f'Missing directory: {folder}')
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'Image_name', 'Age', 'True_Label'}
            if not required.issubset(reader.fieldnames or []):
                parser.error(f'{csv_path}: expected columns {sorted(required)}')
            rows = list(reader)
        names = Counter(row['Image_name'] for row in rows)
        for name, count in names.items():
            if count > 1:
                issue(split, name, 'duplicate_csv_name', str(count))
        files = {p.name: p for p in folder.iterdir() if p.is_file()}
        casefold_names = defaultdict(list)
        for name in files:
            casefold_names[name.casefold()].append(name)
        matched_files = set()
        case_mismatches = 0
        labels, ages, jobs = Counter(), [], []
        for row in rows:
            name, label = row['Image_name'], row['True_Label']
            labels[label] += 1
            if label not in ('0', '1'):
                issue(split, name, 'invalid_label', str(label))
            try:
                age = float(row['Age'])
                if not math.isfinite(age) or not 0 <= age <= 120:
                    raise ValueError
                ages.append(age)
            except (ValueError, TypeError):
                issue(split, name, 'missing_or_invalid_age', str(row['Age']))
            if not name or Path(name).name != name or name in ('.', '..'):
                issue(split, str(name), 'invalid_image_name')
                continue
            path = folder / name
            if not path.is_file():
                candidates = casefold_names.get(name.casefold(), [])
                if len(candidates) == 1:
                    path = files[candidates[0]]
                    case_mismatches += 1
                else:
                    issue(split, name, 'ambiguous_image_name' if candidates else 'missing_image')
                    continue
            matched_files.add(path.name)
            jobs.append((name, label, path))
        for name in files.keys() - matched_files:
            issue(split, name, 'unlisted_file')
        print(f'{split}: checking {len(jobs)} images with {args.workers} workers', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(inspect_image, (job[2] for job in jobs))
            for count, ((name, label, path), result) in enumerate(zip(jobs, results), 1):
                records.append(dict(split=split, image_name=name, actual_filename=path.name, label=label, **result))
                if 'error' in result:
                    issue(split, name, 'image_error', result['error'])
                for warning in result.get('warnings', []):
                    issue(split, name, 'image_warning', warning)
                if count % 2000 == 0:
                    print(f'{split}: {count}/{len(jobs)}', flush=True)
        good = [r for r in records if r['split'] == split and 'pixel_sha256' in r]
        sizes = Counter((r['width'], r['height']) for r in good)
        ratios = [r['width'] / r['height'] for r in good]
        summaries[split] = dict(
            csv_rows=len(rows), label_counts=dict(labels), decoded_images=len(good),
            filename_case_mismatches=case_mismatches,
            age_valid_count=len(ages), age_min=min(ages, default=None), age_max=max(ages, default=None),
            width_min=min((r['width'] for r in good), default=None),
            width_max=max((r['width'] for r in good), default=None),
            height_min=min((r['height'] for r in good), default=None),
            height_max=max((r['height'] for r in good), default=None),
            aspect_ratio_min=min(ratios, default=None), aspect_ratio_max=max(ratios, default=None),
            common_sizes=[dict(width=w, height=h, count=n) for (w, h), n in sizes.most_common(10)],
            csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        )

    duplicates = []
    for key in ('file_sha256', 'pixel_sha256'):
        groups = defaultdict(list)
        for record in records:
            if key in record:
                groups[record[key]].append(record)
        for digest, group in groups.items():
            if len(group) < 2:
                continue
            cross_split = len({r['split'] for r in group}) > 1
            conflicting = len({r['label'] for r in group}) > 1
            duplicates.append(dict(method=key, sha256=digest, cross_split=cross_split,
                                   conflicting_labels=conflicting,
                                   members=[{k: r[k] for k in ('split', 'image_name', 'label')} for r in group]))
    blocking = [i for i in issues if i['kind'] not in ('image_warning', 'unlisted_file')]
    leakage = [d for d in duplicates if d['cross_split'] or d['conflicting_labels']]
    status = 'REVIEW_REQUIRED' if blocking or leakage or issues or any(s['filename_case_mismatches'] for s in summaries.values()) else 'PASS'
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), data_root=str(root),
                  status=status, scope='train and valid only; no test images or metadata read',
                  note='Exact bytes and decoded RGB pixels checked; no near-duplicate or patient identity check.',
                  splits=summaries, issue_counts=dict(Counter(i['kind'] for i in issues)),
                  duplicate_groups=len(duplicates), cross_split_or_conflicting_groups=len(leakage))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, content in [('summary.json', report), ('duplicates.json', duplicates)]:
        (out / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
    with (out / 'issues.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['split', 'image_name', 'kind', 'detail'])
        writer.writeheader()
        writer.writerows(issues)
    with (out / 'image_manifest.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    lines = ['# 训练与验证数据审计', '', f'状态：**{status}**', '',
             '仅检查 train/valid；不读取测试集。原始数据未修改。', '',
             '| 划分 | CSV行数 | 成功解码 | 标签分布 |', '|---|---:|---:|---|']
    for split, summary in summaries.items():
        lines.append(f"| {split} | {summary['csv_rows']} | {summary['decoded_images']} | {summary['label_counts']} |")
    lines += ['', f"问题计数：{report['issue_counts']}",
              f'重复组：{len(duplicates)}（文件哈希与像素哈希分别计数）；跨划分或标签冲突组：{len(leakage)}。', '',
              '文件名大小写不一致数量：' + str({s: v['filename_case_mismatches'] for s, v in summaries.items()}), '',
              '详见 summary.json、issues.csv、duplicates.json 和 image_manifest.jsonl。',
              '完全重复检查不能排除近似重复或同患者泄漏；年龄异常需在训练前明确处理。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 1 if status != 'PASS' else 0


if __name__ == '__main__':
    raise SystemExit(main())
