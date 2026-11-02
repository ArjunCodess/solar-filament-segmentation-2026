"""Train a fixed-budget fold without outer-fold selection; save OOF maps once."""
import argparse
import json
from pathlib import Path
import random
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .dataset import CropDataset, load_image_cache
from .models import build_model
from .losses import composite_loss
from .infer import tiled_predict, postprocess, masks_from_labels
from .metrics import binary_scores, grid_miou_diagnostic, label_prediction_scores, aggregate_reference_pq


def evaluate(model, records, device, tile, tta=False, output=None, save_limit=0, mask_only=False, parameters=None):
    rows = []
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
    previous_sha, probabilities = None, None
    saved_physical = set()
    for position, record in enumerate(sorted(records, key=lambda r:(r['sha256'], str(r['id'])))):
        if record['sha256'] != previous_sha:
            x, valid = load_image_cache(record)
            probabilities = tiled_predict(model, x, device, tile, min(tile//4, 128), tta)
            if mask_only:
                probabilities[1:] = 0  # Untrained auxiliary heads must not split baseline instances.
            labels = postprocess(probabilities, valid, **(parameters or {}))
            previous_sha = record['sha256']
        with np.load(record['cache']) as z:
            gt = z['instances']
            packed = z['instance_masks_packed']
        gt_masks = [np.unpackbits(a, count=gt.size).reshape(gt.shape).astype(bool) for a in packed]
        metrics = {**binary_scores(gt > 0, labels > 0),
                   'grid_miou_diagnostic':grid_miou_diagnostic(gt > 0, labels > 0),
                   **label_prediction_scores(gt_masks, labels, probabilities[0])}
        rows.append({'image_id':str(record['id']), **metrics})
        if output and (not save_limit or record['sha256'] in saved_physical or len(saved_physical) < save_limit):
            saved_physical.add(record['sha256'])
            np.savez_compressed(Path(output)/(Path(record['cache']).stem+'.npz'), probabilities=probabilities.astype(np.float16),
                                truth=gt, truth_packed=packed, valid=valid, raw=x[1], image_id=str(record['id']))
        if (position+1)%10 == 0 or position+1 == len(records):
            print(f'Validated {position+1}/{len(records)} annotation views', flush=True)
    # Average independent annotator views within a physical observation first.
    frame = pd.DataFrame(rows)
    if not rows:
        raise ValueError('No validation records')
    physical = {str(r['id']):r['sha256'] for r in records}
    frame['physical'] = frame.image_id.map(physical)
    means = frame.groupby('physical').mean(numeric_only=True).mean().to_dict()
    means['pq'] = aggregate_reference_pq(rows)
    return means, rows


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but this Python environment has no usable CUDA PyTorch')
    manifest = json.loads(Path(args.manifest).read_text())
    records = manifest['records']
    folds = pd.read_csv(Path(args.manifest).with_name('folds.csv'), dtype={'image_id':str}).set_index('image_id').fold
    train = [r for r in records if folds[str(r['id'])] != args.fold]
    val = [r for r in records if folds[str(r['id'])] == args.fold]
    if not train or not val:
        raise ValueError('Empty fold')
    if {r['sha256'] for r in train}&{r['sha256'] for r in val}:
        raise ValueError('Physical-image leakage across folds')
    dataset = CropDataset(train, args.crop, args.samples, args.seed,
                          scale_range=(args.scale_min, args.scale_max),
                          balance_physical=args.balance_physical,
                          foreground_probability=args.foreground_probability)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=args.device == 'cuda', drop_last=False,
                        generator=torch.Generator().manual_seed(args.seed))
    print(json.dumps({'stage':'building_model', 'architecture':args.architecture, 'training_views':len(train),
                      'validation_views':len(val)}), flush=True)
    model = build_model(args.architecture, not args.no_pretrained).to(args.device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location=args.device, weights_only=False)
        if initial['config']['annotation_sha256'] != manifest['annotation_sha256'] or initial['config']['fold'] != args.fold:
            raise ValueError('Warm start must use the same annotations and held-out fold')
        model.load_state_dict(initial['model'])
        del initial
        print(json.dumps({'stage':'warm_start_loaded', 'checkpoint':args.init_checkpoint}), flush=True)
    encoder = list(model.core.encoder.parameters()) if hasattr(model, 'core') else []
    encoder_ids = {id(p) for p in encoder}
    optimizer = torch.optim.AdamW([
        {'params':encoder, 'lr':args.lr*args.encoder_lr_scale},
        {'params':[p for p in model.parameters() if id(p) not in encoder_ids], 'lr':args.lr},
    ], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.device == 'cuda')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['preprocess'] = manifest['preprocess']
    config['annotation_sha256'] = manifest['annotation_sha256']
    (output/'config.json').write_text(json.dumps(config, indent=2))
    best = float('inf')
    history = []
    start_epoch = 0
    resume_step, resume_losses = 0, []
    if args.resume:
        resumed = torch.load(args.resume, map_location=args.device, weights_only=False)
        previous = resumed['config']
        for key in ('architecture', 'fold', 'annotation_sha256', 'epochs', 'samples', 'focal_alpha', 'encoder_lr_scale',
                    'crop', 'batch_size', 'accumulate', 'lr', 'seed', 'no_topology', 'no_auxiliary',
                    'scale_min', 'scale_max', 'balance_physical', 'foreground_probability'):
            defaults = {'scale_min': 1., 'scale_max': 1., 'balance_physical': False, 'foreground_probability': .65}
            if previous.get(key, defaults.get(key)) != config[key]:
                raise ValueError(f'Resume configuration changed: {key}')
        model.load_state_dict(resumed['model'])
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        scaler.load_state_dict(resumed['scaler'])
        start_epoch = resumed['epoch']
        resume_step = resumed.get('next_step', 0)
        resume_losses = resumed.get('epoch_losses', [])
        best = resumed['best_training_loss']
        history = resumed['history'] if 'history' in resumed else json.loads((output/'history.json').read_text())[:start_epoch]
        if 'torch_rng' in resumed:
            torch.set_rng_state(resumed['torch_rng'].cpu())
            random.setstate(resumed['python_rng'])
            np.random.set_state(resumed['numpy_rng'])
            if args.device == 'cuda':
                torch.cuda.set_rng_state_all([state.cpu() for state in resumed['cuda_rng']])
        del resumed

    def save_resume(epoch, next_step, losses):
        tensor_bytes = sum(p.numel()*p.element_size() for p in model.parameters())
        tensor_bytes += sum(v.numel()*v.element_size() for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor))
        if shutil.disk_usage(output).free < tensor_bytes*1.1+64*1024**2:
            raise RuntimeError('Insufficient free disk space for an atomic checkpoint; previous checkpoint retained')
        state = {'model':model.state_dict(), 'config':config, 'epoch':epoch,
                 'next_step':next_step, 'epoch_losses':losses, 'history':history,
                 'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                 'scaler':scaler.state_dict(), 'best_training_loss':best,
                 'torch_rng':torch.get_rng_state(), 'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if args.device == 'cuda' else []}
        torch.save(state, output/'last.partial.pt')
        (output/'last.partial.pt').replace(output/'last.pt')

    for epoch in range(start_epoch, args.epochs):
        model.train()
        dataset.epoch = epoch
        dataset.skip_before = resume_step * args.batch_size if epoch == start_epoch else 0
        losses = resume_losses if epoch == start_epoch else []
        optimizer.zero_grad(set_to_none=True)
        for step, (x, context, target) in enumerate(loader):
            if epoch == start_epoch and step < resume_step:
                continue
            x, context, target = [t.to(args.device) for t in (x, context, target)]
            with torch.autocast(device_type=args.device, enabled=args.device == 'cuda'):
                loss, parts = composite_loss(model(x, context), target, not args.no_topology, not args.no_auxiliary, args.focal_alpha)
            # Correct scaling for a final accumulation group shorter than requested.
            group_start = (step//args.accumulate)*args.accumulate
            group_size = min(args.accumulate, len(loader)-group_start)
            scaler.scale(loss/group_size).backward()
            if (step+1)%args.accumulate == 0 or step+1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            if args.save_optimizer and args.checkpoint_steps and (step+1)%args.checkpoint_steps == 0 and (step+1)%args.accumulate == 0 and step+1 < len(loader):
                save_resume(epoch, step+1, losses)
            if step == 0 or (step+1)%32 == 0:
                print(json.dumps({'epoch':epoch+1, 'step':step+1, 'steps':len(loader),
                                  'running_loss':float(np.mean(losses))}), flush=True)
        scheduler.step()
        train_loss = float(np.mean(losses))
        row = {'epoch':epoch+1, 'loss':train_loss}
        history.append(row)
        print(json.dumps(row), flush=True)
        (output/'history.json').write_text(json.dumps(history, indent=2))
        state = {'model':model.state_dict(), 'config':config, 'epoch':epoch+1,
                 'training_loss':train_loss, 'selection':'training loss only; outer fold unused'}
        if train_loss < best:
            best = train_loss
            torch.save(state, output/'best.partial.pt')
            (output/'best.partial.pt').replace(output/'best.pt')
        if args.save_optimizer:
            save_resume(epoch+1, 0, [])
    model.load_state_dict(torch.load(output/'best.pt', map_location=args.device, weights_only=False)['model'])
    metrics, rows = evaluate(model, val, args.device, args.crop, args.tta, args.oof_output or output/'oof', args.save_oof_limit, args.no_auxiliary)
    pd.DataFrame(rows).to_csv(output/'oof_metrics.csv', index=False)
    (output/'metrics.json').write_text(json.dumps({'scope':'OOF, public self-evaluator v6 convention; hidden server edge cases unverified', **metrics}, indent=2))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/cache/manifest.json')
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--architecture', choices=['unet', 'segformer_b3', 'convnext_unetpp'], default='segformer_b3')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--samples', type=int, default=800)
    p.add_argument('--crop', type=int, default=512)
    p.add_argument('--scale-min', type=float, default=1.)
    p.add_argument('--scale-max', type=float, default=1.)
    p.add_argument('--balance-physical', action='store_true', help='Sample physical observations uniformly, then one annotator')
    p.add_argument('--foreground-probability', type=float, default=.65)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--encoder-lr-scale', type=float, default=1.)
    p.add_argument('--focal-alpha', type=float, default=.75)
    p.add_argument('--init-checkpoint', help='Warm start from a model trained with the same held-out fold')
    p.add_argument('--resume', help='Resume an optimizer checkpoint without changing its experiment budget')
    p.add_argument('--checkpoint-steps', type=int, default=0, help='Save resumable state at accumulation boundaries every N batches')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='runs/segformer_b3/fold0')
    p.add_argument('--oof-output', help='Optional separate volume for validation probability maps')
    p.add_argument('--no-pretrained', action='store_true')
    p.add_argument('--no-topology', action='store_true')
    p.add_argument('--no-auxiliary', action='store_true')
    p.add_argument('--tta', action='store_true')
    p.add_argument('--save-optimizer', action='store_true', help='Include large optimizer states in checkpoints')
    p.add_argument('--save-oof-limit', type=int, default=0, help='Save maps for at most N physical images; score the entire fold. 0 saves all.')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.epochs < 1 or args.samples < 1 or args.accumulate < 1 or args.batch_size < 1:
        raise ValueError('Epochs, samples, accumulation and batch size must be positive')
    if not 0 < args.focal_alpha < 1 or args.encoder_lr_scale <= 0:
        raise ValueError('Focal alpha must be in (0,1) and encoder learning-rate scale must be positive')
    run(args)
