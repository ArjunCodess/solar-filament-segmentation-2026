"""Decode saved test probability maps using a frozen policy, without a GPU."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from src.infer import postprocess
from src.quality import filter_quality
from src.submission import encode, validate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', required=True)
    parser.add_argument('--params', required=True)
    parser.add_argument('--quality-model')
    parser.add_argument('--expected-images', type=int, default=180)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    files = sorted(Path(args.maps).glob('*.npz'))
    if len(files) != args.expected_images:
        raise ValueError(f'Expected {args.expected_images} cached test observations, found {len(files)}')
    policy = json.loads(Path(args.params).read_text())
    quality = json.loads(Path(args.quality_model).read_text()) if args.quality_model else None
    rows, shapes, sources = [], {}, []
    for path in files:
        with np.load(path) as data:
            maps = data['probabilities'].astype(np.float32)
            valid, raw, image_id = data['valid'].astype(bool), data['raw'].astype(np.float32)/255, str(data['image_id'])
        if image_id != path.stem or image_id in shapes or valid.shape != (2048, 2048):
            raise ValueError('Unexpected or duplicate cached test identity/dimensions')
        shapes[image_id] = valid.shape
        labels = postprocess(maps, valid, **policy['parameters'])
        if quality:
            labels = filter_quality(labels, maps, raw, quality)
        for index in range(1, int(labels.max())+1):
            rows.append({'filament_id': f'{image_id}_{index}', 'segmentation_rle': encode(labels == index)})
        sources.append({'image_id': image_id, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'])
        writer.writeheader()
        writer.writerows(rows)
    result = {**validate_submission(output, shapes), 'policy_sha256': hashlib.sha256(Path(args.params).read_bytes()).hexdigest(),
              'sources': sources, 'warning': 'Decoder must have been selected without test labels or public-score tuning'}
    output.with_suffix('.validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
