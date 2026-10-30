"""Minimal image-only baseline with official validation and reproducible artifacts."""

import argparse
import csv
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import FundusDataset
from model import build_model
from amp_precision import amp_dtype


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def print_progress(stage, epoch, step, total, seen, loss, started):
    if step == 1 or step % 25 == 0 or step == total:
        elapsed = time.perf_counter() - started
        remaining = elapsed / step * (total - step)
        print(f'{stage} epoch={epoch} batch={step}/{total} ({100 * step / total:.1f}%) '
              f'loss={loss / seen:.4f} speed={seen / elapsed:.1f} img/s '
              f'elapsed={elapsed / 60:.1f}m ETA={remaining / 60:.1f}m', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pretrained-path', type=Path)
    parser.add_argument('--model', choices=['efficientnet_b0', 'efficientnet_b3', 'swinv2_tiny', 'convnext_tiny', 'inception_next_tiny', 'fastervit_1'], default='efficientnet_b0')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--jpeg-draft', action='store_true', help='Reduced JPEG decode; use the same mode for inference')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--precision', choices=['fp16', 'bf16'], default='fp16')
    parser.add_argument('--lr-schedule', choices=['constant', 'warmup_cosine', 'late_cosine'], default='constant')
    parser.add_argument('--warmup-epochs', type=float, default=1.0)
    parser.add_argument('--decay-start-epoch', type=int, default=5, help='Completed constant-LR epochs before late cosine decay')
    parser.add_argument('--min-lr', type=float, default=3e-7)
    parser.add_argument('--pos-weight', type=float, default=1.0, help='Positive training loss weight; validation BCE remains unweighted')
    parser.add_argument('--ema-decay', type=float, default=0.0, help='ConvNeXt shadow EMA; zero disables it')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-limit', type=int)
    parser.add_argument('--valid-limit', type=int)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.size, args.lr) <= 0 or args.workers < 0:
        parser.error('epochs, batch-size, size and lr must be positive; workers must be nonnegative')
    if not np.isfinite(args.pos_weight) or args.pos_weight <= 0:
        parser.error('pos-weight must be finite and positive')
    if not np.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        parser.error('ema-decay must be in [0, 1)')
    if args.ema_decay and args.model != 'convnext_tiny':
        parser.error('This EMA comparison supports ConvNeXt-Tiny only')
    if args.lr_schedule == 'warmup_cosine':
        if not np.isfinite([args.lr, args.min_lr, args.warmup_epochs]).all():
            parser.error('Schedule parameters must be finite')
        if not 0 < args.warmup_epochs < args.epochs or not 0 < args.min_lr <= args.lr:
            parser.error('Expected 0 < warmup-epochs < epochs and 0 < min-lr <= lr')
    if args.lr_schedule == 'late_cosine':
        if not np.isfinite([args.lr, args.min_lr]).all() or not 0 < args.min_lr <= args.lr:
            parser.error('Expected finite 0 < min-lr <= lr')
        if not 0 <= args.decay_start_epoch < args.epochs:
            parser.error('Expected 0 <= decay-start-epoch < epochs')
    if not torch.cuda.is_available():
        parser.error('CUDA is required for this baseline')
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        parser.error('This GPU does not support BF16')
    autocast_dtype = amp_dtype(vars(args))
    # A new directory prevents accidental overwrites of a previous experiment.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train_data = FundusDataset(args.data_root, 'train', args.size, args.train_limit, args.seed, args.jpeg_draft)
    valid_data = FundusDataset(args.data_root, 'valid', args.size, args.valid_limit, args.seed, args.jpeg_draft)
    for split, data in [('train', train_data), ('valid', valid_data)]:
        if {s[2] for s in data.samples} != {0, 1}:
            parser.error(f'{split} must contain both labels')
        with (args.output_dir / f'{split}_manifest.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Image_name', 'actual_filename', 'True_Label'])
            writer.writerows((name, path.name, label) for name, path, label in data.samples)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(loss='BCEWithLogitsLoss', optimizer='AdamW',
                  weight_decay=0.01, amp=True, age_used=False, threshold=0.5,
                  validation_loss='unweighted BCEWithLogitsLoss',
                  preprocessing='RGB, aspect-preserving letterbox, ImageNet normalization',
                  augmentation='training horizontal flip p=0.5 only',
                  train_count=len(train_data), valid_count=len(valid_data),
                  smoke_run=args.train_limit is not None or args.valid_limit is not None,
                  torch_version=str(torch.__version__), torchvision_version=str(torchvision.__version__),
                  gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                  pretrained_sha256=sha256(args.pretrained_path) if args.pretrained_path else None,
                  csv_sha256={split: sha256(args.data_root / f'{split}.csv') for split in ('train', 'valid')})
    project = Path(__file__).resolve().parent
    config['source_sha256'] = {name: sha256(project / name) for name in ('train.py', 'dataset.py', 'model.py')}
    config['source_sha256']['amp_precision.py'] = sha256(project / 'amp_precision.py')
    config['grad_scaler_enabled'] = args.precision == 'fp16'
    if args.lr_schedule == 'warmup_cosine':
        batches = (len(train_data) + args.batch_size - 1) // args.batch_size
        total_steps = args.epochs * batches
        warmup_steps = round(args.warmup_epochs * batches)
        if not 1 < warmup_steps < total_steps - 1:
            parser.error('Schedule needs at least two warmup and two cosine batches')
        config.update(lr_total_steps=total_steps, lr_warmup_steps=warmup_steps,
                      lr_step_basis='planned training batches; AMP-skipped batches still advance schedule')
        config['source_sha256']['lr_schedule.py'] = sha256(project / 'lr_schedule.py')
    if args.lr_schedule == 'late_cosine':
        batches = (len(train_data) + args.batch_size - 1) // args.batch_size
        total_steps = args.epochs * batches
        constant_steps = args.decay_start_epoch * batches
        if not constant_steps < total_steps - 1:
            parser.error('Schedule needs at least two cosine batches')
        config.update(lr_total_steps=total_steps, lr_constant_steps=constant_steps,
                      lr_step_basis='planned training batches; AMP-skipped batches still advance schedule')
        config['source_sha256']['lr_schedule.py'] = sha256(project / 'lr_schedule.py')
    if args.model == 'efficientnet_b3':
        config['source_sha256']['model_b3.py'] = sha256(project / 'model_b3.py')
    if args.model == 'convnext_tiny':
        import timm
        config['timm_version'] = timm.__version__
        config['source_sha256']['model_convnext.py'] = sha256(project / 'model_convnext.py')
    if args.model == 'fastervit_1':
        import timm
        config['timm_version'] = timm.__version__
        for name in ('model_fastervit.py', 'vendor/fastervit/__init__.py',
                     'vendor/fastervit/faster_vit_any_res.py', 'vendor/fastervit/registry.py'):
            config['source_sha256'][name] = sha256(project / name)
        config['fastervit_window_size'] = [7, 7, 16, 16]
        config['fastervit_drop_path'] = 0.0
    if args.model == 'inception_next_tiny':
        import timm
        config['timm_version'] = timm.__version__
        config['source_sha256']['model_inceptionnext.py'] = sha256(project / 'model_inceptionnext.py')
    if args.model == 'swinv2_tiny':
        import timm
        config['timm_version'] = timm.__version__
        config['source_sha256']['model_swinv2.py'] = sha256(project / 'model_swinv2.py')
        config['swin_window_size'] = args.size // 32
        config['swin_pretrained_window_size'] = 8
        config['swin_pretrained_input_size'] = 256
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=project, capture_output=True, text=True)
    config['git_revision'] = revision.stdout.strip() if revision.returncode == 0 else None
    state = subprocess.run(['git', 'status', '--porcelain'], cwd=project, capture_output=True, text=True)
    config['git_dirty'] = bool(state.stdout.strip())
    if args.ema_decay:
        config.update(weight_type='online', ema_update_policy='successful optimizer updates only')
    (args.output_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    if args.ema_decay:
        ema_dir = args.output_dir / 'ema'
        ema_dir.mkdir()
        ema_config = dict(config, output_dir=str(ema_dir), weight_type='ema',
                          ema_initialization='first successful update copies online parameters',
                          training_loss_source='online model')
        (ema_dir / 'config.json').write_text(json.dumps(ema_config, indent=2) + '\n')
        for split in ('train', 'valid'):
            (ema_dir / f'{split}_manifest.csv').write_bytes((args.output_dir / f'{split}_manifest.csv').read_bytes())
    print(json.dumps(config, indent=2), flush=True)
    generator = torch.Generator().manual_seed(args.seed)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                   worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **options)
    valid_loader = DataLoader(valid_data, shuffle=False, **options)
    if args.model == 'efficientnet_b3':
        from model_b3 import build_model as selected_builder
    else:
        selected_builder = build_model
    if args.model == 'swinv2_tiny':
        from model_swinv2 import build_model as swin_builder
        model = swin_builder(args.pretrained_path, size=args.size).cuda()
    elif args.model == 'fastervit_1':
        from model_fastervit import build_model as fastervit_builder
        model = fastervit_builder(args.pretrained_path, size=args.size).cuda()
    elif args.model == 'inception_next_tiny':
        from model_inceptionnext import build_model as inception_builder
        model = inception_builder(args.pretrained_path).cuda()
    elif args.model == 'convnext_tiny':
        from model_convnext import build_model as convnext_builder
        model = convnext_builder(args.pretrained_path).cuda()
    else:
        model = selected_builder(args.pretrained_path).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    positive_weight = torch.tensor(args.pos_weight, device='cuda') if args.pos_weight != 1 else None
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    validation_criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=args.precision == 'fp16')
    ema_model = None
    if args.ema_decay:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        if list(model.buffers()):
            raise ValueError('This EMA comparison expects the buffer-free ConvNeXt implementation')
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
        ema_model.eval().requires_grad_(False)
        ema_best = -float('inf')
    lr_stream = None
    if args.lr_schedule == 'warmup_cosine':
        from lr_schedule import warmup_cosine_lr
    if args.lr_schedule == 'late_cosine':
        from lr_schedule import late_cosine_lr
    if args.lr_schedule != 'constant' or args.precision == 'bf16' or args.model == 'fastervit_1' or args.ema_decay:
        lr_stream = (args.output_dir / 'learning_rate.csv').open('w', newline='')
        lr_writer = csv.writer(lr_stream)
        lr_writer.writerow(['step', 'epoch', 'batch', 'lr', 'amp_scale_before', 'amp_scale_after', 'amp_skipped', 'grad_norm'])
    best = -float('inf')
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        train_loss = 0.0
        train_seen = 0
        skipped_updates = 0
        grad_norm_max = 0.0
        print(f'Train epoch={epoch}/{args.epochs}: {len(train_data)} images, {len(train_loader)} batches', flush=True)
        for step, (images, labels, _) in enumerate(train_loader, 1):
            if lr_stream is not None:
                planned_step = (epoch - 1) * len(train_loader) + step - 1
                if args.lr_schedule == 'warmup_cosine':
                    current_lr = warmup_cosine_lr(planned_step, total_steps, warmup_steps, args.lr, args.min_lr)
                elif args.lr_schedule == 'late_cosine':
                    current_lr = late_cosine_lr(planned_step, total_steps, constant_steps, args.lr, args.min_lr)
                else:
                    current_lr = args.lr
                for group in optimizer.param_groups:
                    group['lr'] = current_lr
                scale_before = scaler.get_scale()
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=autocast_dtype):
                loss = criterion(model(images).flatten(), labels)
            if not torch.isfinite(loss).item():
                raise RuntimeError('Non-finite training loss')
            scaler.scale(loss).backward()
            grad_norm = ''
            if args.precision == 'bf16':
                # BF16 uses no loss scaling. Check gradients without clipping or modifying them.
                grad_norm = torch.nn.utils.get_total_norm(
                    [p.grad for p in model.parameters() if p.grad is not None],
                    error_if_nonfinite=True).item()
                grad_norm_max = max(grad_norm_max, grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if lr_stream is not None:
                scale_after = scaler.get_scale()
                skipped = scale_after < scale_before
                skipped_updates += int(skipped)
                if ema_model is not None and not skipped:
                    ema_model.update_parameters(model)
                lr_writer.writerow([planned_step, epoch, step, current_lr, scale_before, scale_after, int(skipped), grad_norm])
                if step == 1 or step % 25 == 0 or step == len(train_loader):
                    lr_stream.flush()
                    print(f'LR epoch={epoch} batch={step} lr={current_lr:.9g} '
                          f'amp_scale={scale_after:g} skipped_updates={skipped_updates}', flush=True)
            train_loss += loss.item() * len(labels)
            train_seen += len(labels)
            print_progress('Train', epoch, step, len(train_loader), train_seen, train_loss, started)
        model.eval()
        predictions, targets, names = [], [], []
        ema_predictions, ema_valid_loss = [], 0.0
        valid_loss = 0.0
        valid_started = time.perf_counter()
        valid_seen = 0
        print(f'Valid epoch={epoch}: {len(valid_data)} images, {len(valid_loader)} batches', flush=True)
        with torch.inference_mode():
            for step, (images, labels, batch_names) in enumerate(valid_loader, 1):
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=autocast_dtype):
                    logits = model(images).flatten()
                    loss = validation_criterion(logits, labels)
                    if ema_model is not None:
                        ema_logits = ema_model(images).flatten()
                        ema_loss = validation_criterion(ema_logits, labels)
                if ema_model is not None:
                    ema_predictions.extend(ema_logits.float().sigmoid().cpu().tolist())
                    ema_valid_loss += ema_loss.item() * len(labels)
                valid_loss += loss.item() * len(labels)
                predictions.extend(logits.float().sigmoid().cpu().tolist())
                targets.extend(labels.cpu().tolist())
                names.extend(batch_names)
                valid_seen += len(labels)
                print_progress('Valid', epoch, step, len(valid_loader), valid_seen, valid_loss, valid_started)
        if not np.isfinite(predictions).all() or not np.isfinite(valid_loss):
            raise RuntimeError('Non-finite validation output')
        torch.cuda.synchronize()
        metrics = dict(epoch=epoch, train_loss=train_loss / len(train_data),
                       valid_loss=valid_loss / len(valid_data),
                       val_auroc=float(roc_auc_score(targets, predictions)),
                       val_auprc=float(average_precision_score(targets, predictions)),
                       epoch_seconds=time.perf_counter() - started,
                       peak_vram_mib=torch.cuda.max_memory_allocated() / 1024**2)
        if lr_stream is not None:
            metrics.update(lr_last=current_lr, amp_skipped_updates=skipped_updates)
        if args.precision == 'bf16':
            metrics['grad_norm_max'] = grad_norm_max
        with (args.output_dir / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)
        if metrics['val_auroc'] > best:
            best = metrics['val_auroc']
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            scaler=scaler.state_dict(), config=config, metrics=metrics), args.output_dir / 'best.pt')
            with (args.output_dir / 'best_valid_predictions.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['Image_name', 'True_Label', 'Probability'])
                writer.writerows(zip(names, (int(t) for t in targets), predictions))
        if ema_model is not None:
            if not np.isfinite(ema_predictions).all() or not np.isfinite(ema_valid_loss):
                raise RuntimeError('Non-finite EMA validation output')
            ema_metrics = dict(metrics, valid_loss=ema_valid_loss / len(valid_data),
                               val_auroc=float(roc_auc_score(targets, ema_predictions)),
                               val_auprc=float(average_precision_score(targets, ema_predictions)),
                               ema_updates=int(ema_model.n_averaged.item()))
            with (ema_dir / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(ema_metrics) + '\n')
            print('EMA ' + json.dumps(ema_metrics), flush=True)
            if ema_metrics['val_auroc'] > ema_best:
                ema_best = ema_metrics['val_auroc']
                torch.save(dict(model=ema_model.module.state_dict(), config=ema_config, metrics=ema_metrics), ema_dir / 'best.pt')
                with (ema_dir / 'best_valid_predictions.csv').open('w', newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['Image_name', 'True_Label', 'Probability'])
                    writer.writerows(zip(names, (int(t) for t in targets), ema_predictions))
    if lr_stream is not None:
        lr_stream.close()
    (args.output_dir / 'best.sha256').write_text(sha256(args.output_dir / 'best.pt') + '  best.pt\n')
    if ema_model is not None:
        (ema_dir / 'best.sha256').write_text(sha256(ema_dir / 'best.pt') + '  best.pt\n')
    print(f'Completed. Best validation AUROC: {best:.6f}', flush=True)


if __name__ == '__main__':
    main()
