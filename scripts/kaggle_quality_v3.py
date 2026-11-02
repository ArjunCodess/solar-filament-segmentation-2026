"""OOF proposal-quality regression, grouped selection within fold 0 only."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    sys.path.insert(0, str(root))
    requirements = [s for s in Path('requirements.txt').read_text().splitlines()
                    if not s.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements,
                    'scikit-learn==1.7.2'], check=True)
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import GroupKFold
    from src.infer import postprocess
    from src.metrics import aggregate_reference_pq, label_prediction_scores
    from src.quality import FEATURES, quality_features, filter_quality, predict_quality
    sources = {json.loads(p.read_text())['fold']: p.parent
               for p in Path('/kaggle/input').rglob('complete.json')}
    params = {'mode': 'components', 'threshold': .5, 'min_area': 64, 'seed_threshold': .65}
    training, targets, groups, calibration = [], [], [], []
    manifest = json.loads((sources[0]/'cache/manifest.json').read_text())
    assignments = pd.read_csv(sources[0]/'cache/folds.csv', dtype={'image_id': str}).set_index('image_id')
    records = {str(r['id']): r for r in manifest['records']}
    weights = []
    for path in sorted((sources[0]/'runs/full/segformer_b3/fold0/oof').glob('*.npz')):
        other = sources[0]/'runs/full/convnext_unetpp/fold0/oof'/path.name
        with np.load(path) as a, np.load(other) as b:
            if not np.array_equal(a['truth_packed'], b['truth_packed']):
                raise ValueError('Annotation mismatch')
            maps = (a['probabilities'].astype(np.float32)+b['probabilities'].astype(np.float32))/2
            valid, raw, image_id = a['valid'].astype(bool), a['raw'], str(a['image_id'])
            truth = [np.unpackbits(mask, count=valid.size).reshape(valid.shape).astype(bool)
                     for mask in a['truth_packed']]
        labels = postprocess(maps, valid, **params)
        features = quality_features(labels, maps, raw)
        areas = np.bincount(labels.ravel())[1:]
        matrix = []
        for mask in truth:
            overlaps = np.bincount(labels[mask], minlength=len(areas)+1)[1:]
            union = mask.sum()+areas-overlaps
            matrix.append(np.divide(overlaps, union, out=np.zeros(len(areas)), where=union > 0))
        ious = np.asarray(matrix).reshape(len(truth), len(areas))
        best = ious.max(0) if len(truth) else np.zeros(len(areas))
        quality = np.where(best > .5, best, 0.)
        offset = len(training)
        training.extend(features.tolist())
        targets.extend(quality.tolist())
        groups.extend([int(assignments.loc[image_id, 'temporal_group'])]*len(features))
        # Each physical image has total weight independent of its annotation-view count.
        views = sum(r['sha256'] == records[image_id]['sha256'] for r in records.values())
        weights.extend([1/views]*len(features))
        calibration.append({'image_id': image_id, 'labels': labels, 'truth': truth,
                            'start': offset, 'stop': len(training)})
    x, y, weight = np.asarray(training, np.float32), np.asarray(targets), np.asarray(weights)
    if len(set(groups)) < 3:
        raise ValueError('Need three independent temporal groups for quality selection')
    oof = np.zeros(len(y))
    for train, val in GroupKFold(n_splits=min(4, len(set(groups)))).split(x, y, groups):
        estimator = ExtraTreesRegressor(n_estimators=128, max_depth=5, min_samples_leaf=16,
                                       random_state=342, n_jobs=2)
        estimator.fit(x[train], y[train], sample_weight=weight[train])
        oof[val] = estimator.predict(x[val])
    choices = []
    for threshold in (0., .1, .2, .3, .4):
        rows = []
        for entry in calibration:
            keep = oof[entry['start']:entry['stop']] >= threshold
            mapping = np.zeros(len(keep)+1, np.int32)
            mapping[1:][keep] = np.arange(1, keep.sum()+1)
            rows.append(label_prediction_scores(entry['truth'], mapping[entry['labels']]))
        choices.append({'minimum_quality': threshold, 'pq': aggregate_reference_pq(rows)})
    selected = max(choices, key=lambda row: (row['pq'], -row['minimum_quality']))
    estimator.fit(x, y, sample_weight=weight)
    model = {'features': list(FEATURES), 'minimum_quality': selected['minimum_quality'],
             'parameters': params, 'convnext_weight': .5, 'trees': []}
    for tree in estimator.estimators_:
        state = tree.tree_
        model['trees'].append({'left': state.children_left.tolist(), 'right': state.children_right.tolist(),
            'feature': state.feature.tolist(), 'threshold': state.threshold.tolist(),
            'value': state.value[:, 0, 0].tolist()})
    if not np.allclose(predict_quality(x, model), estimator.predict(x), atol=1e-12):
        raise ValueError('Portable quality predictor differs from fitted estimator')
    Path('quality_model.json').write_text(json.dumps(model))
    Path('quality_calibration.json').write_text(json.dumps({'selected': selected, 'trials': choices,
        'temporal_groups': len(set(groups)), 'proposals': len(y),
        'scope': 'Grouped out-of-sample selection within saved fold-0 maps; fitted on fold 0 only'}, indent=2))
    rows = []
    for fold in (1, 2, 3, 4):
        first = sources[fold]/f'runs/full/segformer_b3/fold{fold}/oof'
        for path in sorted(first.glob('*.npz')):
            with np.load(path) as a, np.load(sources[fold]/f'runs/full/convnext_unetpp/fold{fold}/oof'/path.name) as b:
                if not np.array_equal(a['truth_packed'], b['truth_packed']):
                    raise ValueError('Annotation mismatch')
                maps = (a['probabilities'].astype(np.float32)+b['probabilities'].astype(np.float32))/2
                valid, raw, image_id = a['valid'].astype(bool), a['raw'], str(a['image_id'])
                truth = [np.unpackbits(mask, count=valid.size).reshape(valid.shape).astype(bool)
                         for mask in a['truth_packed']]
            proposals = postprocess(maps, valid, **params)
            for name, labels in [('unfiltered', proposals), ('quality', filter_quality(proposals, maps, raw, model)),
                    ('v2', postprocess(maps, valid, mode='components', threshold=.5, min_area=256, seed_threshold=.65))]:
                rows.append({'policy': name, 'fold': fold, 'image_id': image_id,
                             **label_prediction_scores(truth, labels)})
        pd.DataFrame(rows).to_csv('quality_heldout.csv', index=False)
        print(json.dumps({'fold': fold, 'views_scored': len(rows)}), flush=True)
    results = [{'policy': name, 'pq': aggregate_reference_pq([r for r in rows if r['policy'] == name])}
               for name in ('unfiltered', 'quality', 'v2')]
    Path('quality_complete.json').write_text(json.dumps({'results': results,
        'scope': 'Saved folds 1-4 subset only; no D4; not complete-fold or Kaggle scores'}, indent=2))


if __name__ == '__main__':
    main()
