"""Prepare training-only dark distractors, excluding every annotator's foreground."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.manifest)
    manifest = json.loads(source.read_text())
    folds = pd.read_csv(source.with_name('folds.csv'), dtype={'image_id': str}).set_index('image_id').fold
    grouped = defaultdict(list)
    for record in manifest['records']:
        # No held-out mask, image, or model is used to construct training samples.
        if folds[str(record['id'])] != args.fold:
            grouped[record['sha256']].append(record)
    count = 0
    for position, records in enumerate(grouped.values()):
        with np.load(records[0]['image_cache']) as data:
            stored, valid = data['image'], data['valid'].astype(bool)
            norm = (stored if stored.ndim == 2 else stored[0]).astype(np.float32)/255
        union = np.zeros(valid.shape, bool)
        for record in records:
            with np.load(record['cache']) as data:
                union |= data['mask'] > 0
        excluded = cv2.dilate(union.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
        background = valid & ~excluded
        contrast = cv2.GaussianBlur(norm, (0, 0), 7)-norm
        threshold = max(.025, float(np.quantile(contrast[background], .97))) if background.any() else np.inf
        yy, xx = np.nonzero(background & (contrast >= threshold))
        order = np.argsort(-contrast[yy, xx], kind='stable')
        cells, centers = set(), []
        for index in order:
            y, x = int(yy[index]), int(xx[index])
            cell = (y//64, x//64)
            if cell not in cells:
                centers.append([y, x])
                cells.add(cell)
            if len(centers) == 64:
                break
        for record in records:
            record['dark_negative_centers'] = centers
        count += len(centers)
        if (position+1) % 50 == 0:
            print(f'Prepared {position+1}/{len(grouped)} physical observations', flush=True)
    manifest['dark_negative_sampling'] = {'held_out_fold': args.fold, 'physical_images': len(grouped),
        'centers': count, 'method': 'Local dark contrast outside dilated union of approved training annotations'}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest))
    destination.with_name('folds.csv').write_bytes(source.with_name('folds.csv').read_bytes())
    print(json.dumps(manifest['dark_negative_sampling']))


if __name__ == '__main__':
    main()
