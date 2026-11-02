"""Verified COCO compressed-count encoding, one row per predicted filament."""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from pycocotools import mask as coco
import torch
from .models import build_model
from .preprocess import normalize_disk
from .infer import tiled_predict, postprocess, masks_from_labels
from .quality import filter_quality


def encode(mask):
    return coco.encode(np.asfortranarray(mask.astype(np.uint8)))['counts'].decode('ascii')


def decode(counts, shape=(2048, 2048)):
    return coco.decode({'size':list(shape), 'counts':counts.encode('ascii')}).astype(bool)


def validate_submission(path, image_shapes, allow_empty_images=True):
    seen, coverage = set(), set()
    occupied = {}
    with open(path, newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['filament_id', 'segmentation_rle']:
            raise ValueError('Expected filament_id,segmentation_rle header')
        for row in reader:
            if None in row or None in row.values():
                raise ValueError('Malformed CSV row')
            key = row['filament_id']
            if key in seen:
                raise ValueError(f'Duplicate filament ID: {key}')
            seen.add(key)
            image_id, separator, suffix = key.rpartition('_')
            if not separator or not suffix or image_id not in image_shapes:
                raise ValueError(f'Unknown image ID or missing suffix: {key}')
            counts = row['segmentation_rle']
            if not counts or counts != counts.strip() or '"' in counts or "'" in counts:
                raise ValueError('Invalid compressed COCO counts')
            mask = decode(counts, image_shapes[image_id])
            if not mask.any():
                raise ValueError('Each row must represent a nonempty filament')
            if encode(mask) != counts:
                raise ValueError('Noncanonical or invalid RLE')
            packed = np.packbits(mask.ravel())
            if image_id in occupied:
                if np.bitwise_and(occupied[image_id], packed).any():
                    raise ValueError(f'Overlapping predicted instances: {image_id}')
                occupied[image_id] |= packed
            else:
                occupied[image_id] = packed
            coverage.add(image_id)
    missing = sorted(set(image_shapes)-coverage)
    if missing and not allow_empty_images:
        raise ValueError(f'Images without rows: {missing}')
    return {'rows':len(seen), 'images_with_predictions':len(coverage), 'images_without_predictions':missing,
            'encoding':'pycocotools compressed COCO counts, Fortran order',
            'overlapping_pixels':0, 'format_errors':0}


def predict(args):
    if not args.checkpoints:
        raise ValueError('Trained checkpoint(s) are required; no fabricated prediction fallback')
    weights = np.asarray(args.weights if args.weights is not None else np.ones(len(args.checkpoints)), dtype=np.float64)
    if len(weights) != len(args.checkpoints) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('Supply one finite nonnegative weight per checkpoint with a positive sum')
    weights /= weights.sum()
    models, configs = [], []
    for path in args.checkpoints:
        state = torch.load(path, map_location=args.device, weights_only=False)
        config = state['config']
        model = build_model(config['architecture'], False).to(args.device)
        model.load_state_dict(state['model'])
        models.append(model.eval())
        configs.append(config)
    preprocessing = configs[0]['preprocess']
    if any(c['preprocess'] != preprocessing for c in configs):
        raise ValueError('Ensemble preprocessing differs')
    params = json.loads(Path(args.params).read_text())['parameters'] if args.params else {}
    quality_model = json.loads(Path(args.quality_model).read_text()) if args.quality_model else None
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths:
        raise ValueError('No test images found')
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError('Duplicate test observation filenames')
    shapes, rows = {}, []
    for position, path in enumerate(paths):
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError(f'Unreadable image: {path}')
        if raw.shape != (2048, 2048) and not args.allow_nonstandard_size:
            raise ValueError('Official test images must be 2048x2048')
        shapes[path.stem] = raw.shape
        norm, valid, _ = normalize_disk(raw, preprocessing['clahe'], preprocessing['limb'])
        image = np.stack([norm, raw.astype(np.float32)/255])
        member_maps = []
        for model, config in zip(models, configs):
            predicted = tiled_predict(model, image, args.device, args.tile, args.tile//4, args.tta,
                                      args.tta_batch_size)
            if config.get('no_auxiliary'):
                predicted[1:] = 0
            member_maps.append(predicted)
        maps = np.average(member_maps, axis=0, weights=weights).astype(np.float32)
        labels = postprocess(maps, valid, **params)
        if quality_model is not None:
            labels = filter_quality(labels, maps, image[1], quality_model)
        for i, mask in enumerate(masks_from_labels(labels), 1):
            rows.append({'filament_id':f'{path.stem}_{i}', 'segmentation_rle':encode(mask)})
        if (position+1)%5 == 0 or position+1 == len(paths):
            print(f'Predicted {position+1}/{len(paths)} test observations; {len(rows)} filament rows', flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'])
        writer.writeheader()
        writer.writerows(rows)
    result = validate_submission(output, shapes)
    output.with_suffix('.validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--checkpoints', nargs='+', required=True)
    p.add_argument('--weights', nargs='+', type=float, help='Probability averaging weights in checkpoint order')
    p.add_argument('--output', default='submission.csv')
    p.add_argument('--params')
    p.add_argument('--quality-model', help='Portable quality regressor fitted on OOF proposals only')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--tile', type=int, default=512)
    p.add_argument('--tta', action='store_true')
    p.add_argument('--tta-batch-size', type=int, default=1)
    p.add_argument('--allow-nonstandard-size', action='store_true', help='Synthetic testing only')
    predict(p.parse_args())
