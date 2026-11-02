"""Fold-0-frozen shape decoding search and error audit on existing OOF maps."""
import itertools
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
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import numpy as np
    import pandas as pd
    from src.error_analysis import instance_error_rows
    from src.infer import postprocess
    from src.metrics import aggregate_reference_pq, label_prediction_scores

    sources = {json.loads(p.read_text())['fold']: p.parent
               for p in Path('/kaggle/input').rglob('complete.json')}
    if set(sources) != set(range(5)):
        raise ValueError('Expected original five fold sources')
    policies, trials, errors, heldout = {}, {}, [], []
    search = list(itertools.product((.4, .5, .6, .7), (64, 256), (0, 2, 4), (0., .7)))
    for fold in range(5):
        source = sources[fold]
        first = source/f'runs/full/segformer_b3/fold{fold}/oof'
        second = source/f'runs/full/convnext_unetpp/fold{fold}/oof'
        paths = sorted(first.glob('*.npz'))
        if not paths:
            raise ValueError(f'No saved OOF maps for fold {fold}')
        for position, path in enumerate(paths):
            with np.load(path) as a, np.load(second/path.name) as b:
                if str(a['image_id']) != str(b['image_id']) or not np.array_equal(a['truth_packed'], b['truth_packed']):
                    raise ValueError('Annotation-view mismatch')
                seg, conv = a['probabilities'].astype(np.float32), b['probabilities'].astype(np.float32)
                valid, raw, image_id = a['valid'].astype(bool), a['raw'], str(a['image_id'])
                truth = [np.unpackbits(mask, count=valid.size).reshape(valid.shape).astype(bool)
                         for mask in a['truth_packed']]
            if fold == 0:
                for weight in (.5, 1.):
                    maps = (1-weight)*seg+weight*conv
                    for threshold, area, radius, mean in search:
                        params = {'mode': 'components', 'threshold': threshold, 'min_area': area,
                                  'seed_threshold': min(1., threshold+.15),
                                  'merge_radius': radius, 'mean_threshold': mean}
                        key = json.dumps({'convnext_weight': weight, 'parameters': params}, sort_keys=True)
                        labels = postprocess(maps, valid, **params)
                        trials.setdefault(key, []).append(label_prediction_scores(truth, labels))
            else:
                for name, policy in policies.items():
                    weight = policy['convnext_weight']
                    maps = (1-weight)*seg+weight*conv
                    labels = postprocess(maps, valid, **policy['parameters'])
                    heldout.append({'policy': name, 'fold': fold, 'image_id': image_id,
                                    **label_prediction_scores(truth, labels)})
                    errors.extend({**row, 'fold': fold, 'policy': name} for row in
                                  instance_error_rows(truth, labels, maps[0], raw, valid, image_id))
            print(json.dumps({'fold': fold, 'view': position+1, 'total': len(paths)}), flush=True)
        if fold == 0:
            table = [{**json.loads(key), 'pq': aggregate_reference_pq(rows), 'views': len(rows)}
                     for key, rows in trials.items()]
            # Prefer simpler extraction when scores tie, and freeze before other folds.
            best = max(table, key=lambda row: (row['pq'], -row['parameters']['merge_radius']))
            policies = {'shape_selected': best,
                        'v2_frozen': {'convnext_weight': .5, 'parameters': {
                            'mode': 'components', 'threshold': .5, 'min_area': 256, 'seed_threshold': .65}},
                        'submitted_v1': {'convnext_weight': .5, 'parameters': json.loads(
                            (root/'baseline_policy.json').read_text())['parameters']}}
            Path('shape_calibration.json').write_text(json.dumps({'selected': best, 'trials': table,
                'scope': 'Saved fold-0 OOF maps only; frozen before folds 1-4'}, indent=2))
            Path('shape_policies.json').write_text(json.dumps(policies, indent=2))
        else:
            pd.DataFrame(heldout).to_csv('shape_heldout.csv', index=False)
            pd.DataFrame(errors).to_csv('instance_errors.csv', index=False)
    summaries = []
    for name in policies:
        rows = [row for row in heldout if row['policy'] == name]
        summaries.append({'policy': name, 'pq': aggregate_reference_pq(rows), 'views': len(rows)})
    frame = pd.DataFrame(errors)
    truth = frame[frame.kind == 'truth']
    groups = truth.groupby(['policy', 'size_bin']).agg(instances=('matched', 'size'),
        hit_rate=('matched', 'mean'), mean_best_iou=('best_iou', 'mean'),
        skeleton_recall=('skeleton_recall', 'mean')).reset_index()
    groups.to_csv('errors_by_size.csv', index=False)
    Path('shape_complete.json').write_text(json.dumps({'results': summaries,
        'scope': 'Saved OOF subset folds 1-4, no D4; not full-fold or leaderboard scores'}, indent=2))


if __name__ == '__main__':
    main()
