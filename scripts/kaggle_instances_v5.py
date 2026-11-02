"""Finite two-fold object-detection and per-instance segmentation experiment."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(fold):
    import numpy as np
    import pandas as pd
    import torch
    from src.instance_model import (InstanceTrainingData, augmented_detection, build_detector,
        build_refiner, masks_to_boxes, instance_crop, instance_loss)
    torch.set_num_threads(2)
    seed = 542
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    source, = [p.parent for p in Path('/kaggle/input').rglob('complete.json')
               if json.loads(p.read_text())['fold'] == fold]
    annotations, = Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    manifest = json.loads((source/'cache/manifest.json').read_text())
    for record in manifest['records']:
        for key in ('cache', 'image_cache'):
            record[key] = str(source/'cache'/Path(record[key]).name)
        record['image_path'] = str(annotations.parent/'train_images'/Path(record['image_path']).name)
    directory = Path(f'instance_v5_fold{fold}')
    directory.mkdir(exist_ok=True)
    (directory/'manifest.json').write_text(json.dumps(manifest))
    (directory/'folds.csv').write_bytes((source/'cache/folds.csv').read_bytes())
    folds = pd.read_csv(directory/'folds.csv', dtype={'image_id': str}).set_index('image_id').fold
    train = [r for r in manifest['records'] if folds[str(r['id'])] != fold]
    val = [r for r in manifest['records'] if folds[str(r['id'])] == fold]
    if {r['sha256'] for r in train} & {r['sha256'] for r in val}:
        raise ValueError('Physical identity leakage')
    data = InstanceTrainingData(train)
    provenance = {'held_out_fold': fold, 'seed': seed, 'training_views': len(train),
        'training_physical': len(data.physical), 'validation_views': len(val),
        'initialization': 'ImageNet classification weights only; random detector heads; no COCO detection weights',
        'labels': 'Only bounding boxes and individual instance masks derived from approved training segmentation',
        'detector': 'Faster R-CNN ResNet50 FPN, disk1024, thin anchors; 24epochs480samples',
        'refiner': 'ResNet18 U-Net, individual jittered object crops512, 20epochs800samples',
        'selection': 'Fixed final epoch; outer fold unused for training or checkpoint selection'}
    (directory/'provenance.json').write_text(json.dumps(provenance, indent=2))
    detector = build_detector().cuda().train()
    optimizer = torch.optim.AdamW(detector.parameters(), lr=.0002, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    total = 24*480
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: min(1., (step+1)/480)*(.05+.95*.5*(1+np.cos(np.pi*step/total))))
    history = []
    for epoch in range(24):
        losses = []
        for step in range(480):
            image, valid, masks = data.sample(rng)
            boxes = masks_to_boxes(masks)
            composite, boxes = augmented_detection(np.stack([image[1], image[0], image[0]]), boxes, rng)
            target = {'boxes': torch.from_numpy(boxes).cuda(),
                      'labels': torch.ones(len(boxes), dtype=torch.int64, device='cuda')}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                parts = detector([torch.from_numpy(composite).cuda()], [target])
                loss = sum(parts.values())
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite detector loss: {parts}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses))})
        (directory/'detector_history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps({'fold': fold, 'stage': 'detector', **history[-1]}), flush=True)
    torch.save({'model': detector.state_dict(), 'provenance': provenance}, directory/'detector.pt')
    del detector, optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    refiner = build_refiner().cuda().train()
    encoder = list(refiner.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    optimizer = torch.optim.AdamW([{'params': encoder, 'lr': .0001},
        {'params': [p for p in refiner.parameters() if id(p) not in encoder_ids], 'lr': .001}],
        weight_decay=.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=.00001)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    for epoch in range(20):
        losses = []
        for step in range(800):
            image, valid, masks = data.sample(rng)
            if not masks:
                continue
            x, y = instance_crop(image, masks[int(rng.integers(len(masks)))], rng)
            x, y = torch.from_numpy(x)[None].cuda(), torch.from_numpy(y)[None].cuda()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                loss = instance_loss(refiner(x), y)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite refiner loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 10.)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
        scheduler.step()
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses))})
        (directory/'refiner_history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps({'fold': fold, 'stage': 'refiner', **history[-1]}), flush=True)
    torch.save({'model': refiner.state_dict(), 'provenance': provenance}, directory/'refiner.pt')
    hashes = {}
    for name in ('detector.pt', 'refiner.pt'):
        with (directory/name).open('rb') as stream:
            hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (directory/'trained.json').write_text(json.dumps(hashes, indent=2))


def main():
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    sys.path.insert(0, str(root))
    requirements = [s for s in Path('requirements.txt').read_text().splitlines()
                    if not s.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import numpy as np
    import pandas as pd
    import torch
    from src.dataset import load_image_cache
    from src.instance_model import (build_detector, build_refiner, instance_proposals,
                                    decode_proposals, load_truth)
    from src.metrics import label_prediction_scores, aggregate_reference_pq
    from src.error_analysis import instance_error_rows
    from src.submission import encode
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two T4 GPUs required for independent temporal folds')
    started = time.time()
    processes = []
    streams = []
    for fold in (0, 1):
        stream = Path(f'instance_v5_fold{fold}.log').open('w')
        streams.append(stream)
        processes.append(subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(fold)],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(fold)}, stdout=stream, stderr=subprocess.STDOUT))
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        for path in root.glob('instance_v5_fold*.log'):
            print(path.name, path.read_text()[-6000:], flush=True)
        raise RuntimeError(f'Training exit codes: {codes}')
    torch.set_num_threads(2)
    detector, refiner = build_detector(False), build_refiner(False)
    records_by_fold = {}
    for fold in (0, 1):
        directory = Path(f'instance_v5_fold{fold}')
        manifest = json.loads((directory/'manifest.json').read_text())
        folds = pd.read_csv(directory/'folds.csv', dtype={'image_id': str}).set_index('image_id').fold
        records_by_fold[fold] = sorted([r for r in manifest['records'] if folds[str(r['id'])] == fold],
                                      key=lambda r: (r['sha256'], str(r['id'])))
    candidates = [dict(score_threshold=score, mask_threshold=mask, min_area=area)
                  for score, mask, area in itertools.product((.1, .2, .3, .4, .5, .6), (.4, .5, .6), (16, 64))]
    trials = [[] for _ in candidates]
    frozen = None
    for fold in (0, 1):
        directory = Path(f'instance_v5_fold{fold}')
        detector.load_state_dict(torch.load(directory/'detector.pt', map_location='cpu', weights_only=False)['model'])
        refiner.load_state_dict(torch.load(directory/'refiner.pt', map_location='cpu', weights_only=False)['model'])
        detector.cuda().eval()
        refiner.cuda().eval()
        previous, proposals, rows, errors, predictions = None, None, [], [], []
        physical_count = 0
        for position, record in enumerate(records_by_fold[fold]):
            if previous != record['sha256']:
                if fold == 0 and physical_count == 16:
                    break
                image, valid = load_image_cache(record)
                proposals = instance_proposals(detector, refiner, image, 'cuda')
                previous = record['sha256']
                physical_count += 1
            truth = load_truth(record, valid.shape)
            if fold == 0:
                for parameters, scores in zip(candidates, trials):
                    labels, confidence = decode_proposals(proposals, valid, **parameters)
                    scores.append({'image_id': str(record['id']), **label_prediction_scores(truth, labels, confidence)})
            else:
                labels, confidence = decode_proposals(proposals, valid, **frozen['parameters'])
                rows.append({'image_id': str(record['id']), **label_prediction_scores(truth, labels, confidence)})
                errors.extend(instance_error_rows(truth, labels, confidence, image[1], valid, record['id']))
                for label in range(1, int(labels.max())+1):
                    predictions.append({'image_id': str(record['id']), 'instance': label,
                                        'segmentation_rle': encode(labels == label)})
            print(json.dumps({'fold': fold, 'validation_view': position+1, 'physical_images': physical_count}), flush=True)
        if fold == 0:
            table = [{'parameters': parameters, 'pq': aggregate_reference_pq(scores), 'views': len(scores)}
                     for parameters, scores in zip(candidates, trials)]
            frozen = max(table, key=lambda row: row['pq'])
            Path('instance_v5_calibration.json').write_text(json.dumps(table, indent=2))
            Path('instance_v5_frozen_policy.json').write_text(json.dumps(frozen, indent=2))
        else:
            if len(rows) != 215 or len({r['image_id'] for r in rows}) != 215:
                raise ValueError('Expected every complete fold1 annotation view exactly once')
            pd.DataFrame(rows).to_csv('instance_v5_full_fold1.csv', index=False)
            pd.DataFrame(errors).to_csv('instance_v5_errors.csv', index=False)
            pd.DataFrame(predictions).to_csv('instance_v5_oof_instances.csv', index=False)
            truth_errors = pd.DataFrame(errors).query('kind == "truth"')
            truth_errors.groupby('size_bin').agg(instances=('matched', 'size'), hit_rate=('matched', 'mean'),
                skeleton_recall=('skeleton_recall', 'mean')).to_csv('instance_v5_errors_by_size.csv')
            Path('instance_v5_complete.json').write_text(json.dumps({'pq': aggregate_reference_pq(rows),
                'views': len(rows), 'policy': frozen['parameters'], 'seconds': time.time()-started,
                'scope': 'Complete temporal fold1. Fold0-saved-map calibration. One detector and one refiner; '
                         'no D4; fixed training budget; no test inference or automatic submission.'}, indent=2))
        detector.cpu()
        refiner.cpu()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    os.chdir('/kaggle/working/solar')
    sys.path.insert(0, os.getcwd())
    if args.worker is None:
        main()
    else:
        worker(args.worker)
