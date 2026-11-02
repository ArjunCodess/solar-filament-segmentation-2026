"""COCO import, leakage-resistant folds, cached native-resolution crop sampling."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import zipfile
import shutil
import cv2
import numpy as np
import pandas as pd
from .preprocess import normalize_disk, targets, decode_annotation


def observation_date(record):
    match = re.search(r'(20\d{12})', record['file_name'])
    if match:
        return datetime.strptime(match[1], '%Y%m%d%H%M%S')
    return datetime.fromisoformat(record['date_captured'].replace('Z', '+00:00')).replace(tzinfo=None)


def load_coco(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    by_image = defaultdict(list)
    ids = {str(x['id']) for x in data['images']}
    for ann in data['annotations']:
        if str(ann['image_id']) not in ids:
            raise ValueError('Orphan annotation')
        by_image[str(ann['image_id'])].append(ann)
    return data, by_image


def make_folds(records, annotations, n_folds=5, gap_days=3, seed=42):
    """Connected temporal groups: no observations <= gap_days apart cross folds.

    Greedy vector balancing of count, object-area bins, year and sample count.
    A continuous observing run stays together, even across a calendar boundary.
    """
    ordered = sorted(records, key=observation_date)
    groups, current, previous = [], [], None
    for rec in ordered:
        date = observation_date(rec)
        if previous is not None and (date-previous).total_seconds() > gap_days*86400:
            groups.append(current)
            current = []
        current.append(rec)
        previous = date
    if current:
        groups.append(current)
    if len(groups) < n_folds:
        raise ValueError(f'Only {len(groups)} independent temporal groups for {n_folds} folds; do not silently leak dates')
    years = sorted({observation_date(r).year for r in records})
    def features(group):
        vector = np.zeros(2+4+len(years), float)
        # Count physical observations once; average over independent annotator views.
        physical = defaultdict(list)
        for r in group:
            physical[Path(r['file_name']).name].append(r)
        for views in physical.values():
            vector[0] += 1
            vector[6+years.index(observation_date(views[0]).year)] += 1
            for r in views:
                anns = annotations[str(r['id'])]
                vector[1] += len(anns)/len(views)
                areas = [a.get('area', a['bbox'][2]*a['bbox'][3])/(r['width']*r['height']) for a in anns]
                vector[2:6] += np.histogram(areas, [0, 1e-5, 1e-4, 1e-3, np.inf])[0]/len(views)
        return vector
    vectors = np.array([features(g) for g in groups])
    goal = vectors.sum(0)/n_folds
    normalizer = np.maximum(goal, 1)
    rng = np.random.default_rng(seed)
    order = sorted(rng.permutation(len(groups)), key=lambda i:-float(np.linalg.norm(vectors[i]/normalizer)))
    totals = np.zeros((n_folds, vectors.shape[1]))
    assignments = {}
    for rank, i in enumerate(order):
        costs = []
        for fold in range(n_folds):
            trial = totals.copy()
            trial[fold] += vectors[i]
            costs.append(float(np.square((trial-goal)/normalizer).sum()))
        fold = rank if rank < n_folds else int(np.argmin(costs))
        totals[fold] += vectors[i]
        for r in groups[i]:
            assignments[str(r['id'])] = (fold, i)
    return pd.DataFrame([{'image_id':str(r['id']), 'file_name':r['file_name'],
                          'date':observation_date(r).isoformat(), 'fold':assignments[str(r['id'])][0],
                          'temporal_group':assignments[str(r['id'])][1],
                          'filament_count':len(annotations[str(r['id'])])} for r in records])


def prepare(coco_path, images, output, test_images=None, use_spines=False, limb=True, clahe=True,
            compact_images=False, reuse_cache=None, max_new_records=0):
    output, images = Path(output), Path(images)
    output.mkdir(parents=True, exist_ok=True)
    data, by_image = load_coco(coco_path)
    contract = {'annotation_sha256':hashlib.sha256(Path(coco_path).read_bytes()).hexdigest(),
                'limb':limb, 'clahe':clahe, 'use_spines':use_spines, 'compact_images':compact_images}
    state_path = output/'preparation.json'
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        previous.setdefault('compact_images', False)
        if previous != contract:
            raise ValueError('Cache preparation settings changed; use a separate output directory')
    state_path.write_text(json.dumps(contract, indent=2))
    if reuse_cache:
        reuse_cache = Path(reuse_cache)
        previous = json.loads((reuse_cache/'preparation.json').read_text())
        if any(previous.get(k) != contract[k] for k in ('annotation_sha256', 'limb', 'clahe', 'use_spines')):
            raise ValueError('Reusable targets have different annotations or preprocessing')
    test_stems = {p.stem for p in Path(test_images).glob('*')} if test_images else set()
    seen = {}
    manifest = []
    newly_prepared = 0
    for position, rec in enumerate(data['images']):
        path = images/Path(rec['file_name']).name
        if path.stem in test_stems:
            raise ValueError(f'Training/test observation overlap: {path.stem}')
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
        if image.shape != (rec['height'], rec['width']):
            raise ValueError(f'Image size mismatch: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        shared = output/f'image_{digest[:20]}.npz'
        if digest not in seen:
            geometry = None
            if shared.exists():
                try:
                    with np.load(shared) as existing:
                        if 'geometry_json' in existing:
                            geometry = json.loads(str(existing['geometry_json']))
                except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                    geometry = None  # Rebuild an image cache interrupted by an earlier process.
            if geometry is None:
                reusable_image = reuse_cache/shared.name if reuse_cache else None
                if reusable_image and reusable_image.exists():
                    with np.load(reusable_image) as existing:
                        geometry = json.loads(str(existing['geometry_json']))
                        stored = existing['image']
                        normalized = (stored if stored.ndim == 2 else stored[0]).astype(np.float32)/255
                        del stored
                        disk = existing['valid']
                else:
                    normalized, disk, geometry = normalize_disk(image, clahe=clahe, limb=limb)
                shared_temporary = shared.with_suffix('.partial.npz')
                stored_image = np.uint8(np.rint(normalized*255))
                if not compact_images:
                    stored_image = np.stack([stored_image, image])
                np.savez_compressed(shared_temporary, image=stored_image, valid=disk,
                                    geometry_json=json.dumps(geometry))
                shared_temporary.replace(shared)
                del normalized, disk, stored_image
            seen[digest] = geometry
        geometry = seen[digest]
        key = hashlib.sha256(str(rec['id']).encode()).hexdigest()[:20]
        cache = output/f'{key}.npz'
        if not cache.exists() and reuse_cache and (reuse_cache/cache.name).exists():
            shutil.copyfile(reuse_cache/cache.name, cache)
        # Preserve overlapping GT instances losslessly, without a dense N*H*W float tensor.
        if not cache.exists():
            target = targets(by_image[str(rec['id'])], image.shape, use_spines=use_spines)
            objects = [np.packbits(decode_annotation(a, image.shape).ravel()) for a in by_image[str(rec['id'])]]
            packed = np.stack(objects) if objects else np.zeros((0, (image.size+7)//8), np.uint8)
            encoded = {k:np.uint8(np.rint(v*255)) for k, v in target.items() if k != 'instances'}
            temporary = cache.with_suffix('.partial.npz')
            np.savez_compressed(temporary, **encoded, instances=target['instances'].astype(np.uint16),
                                instance_masks_packed=packed)
            temporary.replace(cache)
            del target, objects, packed, encoded
            newly_prepared += 1
        manifest.append({**rec, 'cache':str(cache.resolve()), 'image_cache':str(shared.resolve()),
                         'image_path':str(path.resolve()), 'sha256':digest, 'disk':geometry})
        if (position+1)%25 == 0 or position+1 == len(data['images']):
            print(f'Prepared {position+1}/{len(data["images"])} annotation views; {len(seen)} physical images', flush=True)
        if max_new_records and newly_prepared >= max_new_records:
            print(f'Completed {newly_prepared} new targets; restart to release process memory and resume.', flush=True)
            return manifest
    # Byte-identical files with different filenames must also stay in one fold.
    canonical = {}
    for rec in manifest:
        canonical.setdefault(rec['sha256'], rec['file_name'])
        rec['file_name'] = canonical[rec['sha256']]
    folds = make_folds(manifest, by_image)
    folds.to_csv(output/'folds.csv', index=False)
    (output/'manifest.json').write_text(json.dumps({'records':manifest, 'source':str(Path(coco_path).resolve()),
        'annotation_sha256':hashlib.sha256(Path(coco_path).read_bytes()).hexdigest(),
        'preprocess':{'limb':limb, 'clahe':clahe, 'use_spines':use_spines}}, indent=2), encoding='utf-8')
    return manifest


def load_image_uint8(record):
    with np.load(record['image_cache']) as z:
        stored, valid = z['image'], z['valid']
    if stored.ndim == 2:
        raw = cv2.imread(record['image_path'], cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != stored.shape:
            raise ValueError('Original grayscale observation is missing or changed')
        stored = np.stack([stored, raw])
    return stored, valid


def load_image_cache(record):
    stored, valid = load_image_uint8(record)
    return stored.astype(np.float32)/255, valid


class CropDataset:
    def __init__(self, records, crop=512, samples=1000, seed=42, augment=True,
                 scale_range=(1., 1.), balance_physical=False, foreground_probability=.65,
                 instance_balanced=False, dark_negative_probability=0.):
        self.records, self.crop, self.samples = records, crop, samples
        self.seed, self.epoch, self.augment, self.skip_before = seed, 0, augment, 0
        if (not 0 < scale_range[0] <= scale_range[1] or not 0 <= foreground_probability <= 1
                or not 0 <= dark_negative_probability <= 1-foreground_probability):
            raise ValueError('Invalid crop scales or foreground sampling probability')
        self.scale_range, self.foreground_probability = scale_range, foreground_probability
        self.instance_balanced, self.dark_negative_probability = instance_balanced, dark_negative_probability
        grouped = defaultdict(list)
        if balance_physical:
            for record in records:
                grouped[record['sha256']].append(record)
        self.physical_groups = list(grouped.values())

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        import torch
        if index < self.skip_before:
            empty = torch.empty(0, dtype=torch.float32)
            return empty, empty, empty
        rng = np.random.default_rng(self.seed+self.epoch*self.samples+index)
        if self.physical_groups:
            group = self.physical_groups[int(rng.integers(len(self.physical_groups)))]
            record = group[int(rng.integers(len(group)))]
        else:
            record = self.records[int(rng.integers(len(self.records)))]
        with np.load(record['image_cache']) as z:
            stored, valid = z['image'], z['valid']
        norm = stored if stored.ndim == 2 else stored[0]
        context = (cv2.resize(norm, (128, 128), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255)[None]
        h, w = norm.shape[-2:]
        c = self.crop if self.scale_range == (1., 1.) else max(1, round(self.crop*rng.uniform(*self.scale_range)))
        h, w = max(h, c), max(w, c)
        with np.load(record['cache']) as z:
            mask = z['mask']
            draw = rng.random()
            if draw < self.foreground_probability and mask.any():
                if self.instance_balanced:
                    instances = z['instances']
                    objects = np.unique(instances)
                    objects = objects[objects > 0]
                    chosen = objects[int(rng.integers(len(objects)))]
                    points = np.argwhere(instances == chosen)
                else:
                    points = np.argwhere(mask > 0)
                cy, cx = points[rng.integers(len(points))]
                top = int(np.clip(cy-rng.integers(c), 0, h-c))
                left = int(np.clip(cx-rng.integers(c), 0, w-c))
            elif (draw < self.foreground_probability+self.dark_negative_probability
                  and record.get('dark_negative_centers')):
                centers = record['dark_negative_centers']
                cy, cx = centers[int(rng.integers(len(centers)))]
                top = int(np.clip(cy-rng.integers(c), 0, h-c))
                left = int(np.clip(cx-rng.integers(c), 0, w-c))
            else:
                top, left = int(rng.integers(h-c+1)), int(rng.integers(w-c+1))
            region = np.s_[top:top+c, left:left+c]
            y = np.stack([mask[region], z['spine'][region], z['boundary'][region],
                          z['skeleton'][region], valid[region].astype(np.uint8)*255]).astype(np.float32)/255
        if stored.ndim == 2:
            raw = cv2.imread(record['image_path'], cv2.IMREAD_GRAYSCALE)
            if raw is None or raw.shape != norm.shape:
                raise ValueError('Original grayscale observation is missing or changed')
            x = np.stack([norm[region], raw[region]]).astype(np.float32)/255
            del raw
        else:
            x = stored[:, top:top+c, left:left+c].astype(np.float32)/255
        del stored, norm, valid
        if min(x.shape[-2:]) < c:
            pad = ((0, 0), (0, c-x.shape[-2]), (0, c-x.shape[-1]))
            x, y = np.pad(x, pad), np.pad(y, pad)
        if c != self.crop:
            x = np.stack([cv2.resize(a, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR) for a in x])
            y = np.stack([cv2.resize(a, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST) for a in y])
        if self.augment:
            k, flip = int(rng.integers(4)), bool(rng.integers(2))
            x, y, context = [np.rot90(a, k, axes=(-2, -1)) for a in (x, y, context)]
            if flip:
                x, y, context = [a[..., ::-1] for a in (x, y, context)]
            gain, offset = rng.uniform(.9, 1.1), rng.uniform(-.03, .03)
            x = np.clip(x*gain+offset+rng.normal(0, .008, x.shape), 0, 1)
            context = np.clip(context*gain+offset, 0, 1)
        return tuple(torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in (x, context, y))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--annotations', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--output', default='data/cache')
    p.add_argument('--test-images')
    p.add_argument('--use-annotated-spines', action='store_true', help='Enable only after organizer clarification')
    p.add_argument('--no-limb', action='store_true')
    p.add_argument('--no-clahe', action='store_true')
    p.add_argument('--compact-images', action='store_true')
    p.add_argument('--reuse-cache')
    p.add_argument('--max-new-records', type=int, default=0, help='Bound each preprocessing process; rerun until manifest.json exists')
    a = p.parse_args()
    prepare(a.annotations, a.images, a.output, a.test_images, a.use_annotated_spines, not a.no_limb, not a.no_clahe,
            a.compact_images, a.reuse_cache, a.max_new_records)
