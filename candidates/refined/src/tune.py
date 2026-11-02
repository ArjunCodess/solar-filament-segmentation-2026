"""OOF-only parameter search. Keep a separate evaluation fold for unbiased scores."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from .infer import postprocess, masks_from_labels
from .metrics import label_prediction_scores, grid_miou_diagnostic, aggregate_reference_pq


def tune(paths, metric='pq', wide=False):
    if not paths:
        raise ValueError('No held-out probability maps')
    arrays = []
    for path in paths:
        with np.load(path) as z:
            item = {k:z[k] for k in ('probabilities', 'truth', 'valid')}
            item['truth_masks'] = ([np.unpackbits(a, count=item['truth'].size).reshape(item['truth'].shape).astype(bool)
                                    for a in z['truth_packed']] if 'truth_packed' in z else masks_from_labels(item['truth']))
            arrays.append(item)
    trials = []
    candidates = (itertools.product((.4, .5, .6, .7, .8, .9), (.5, 1.1), (.5,), (16, 64, 256), (0,))
                  if wide else itertools.product((.35, .45, .55), (.3, .5), (.5, .7), (4, 12), (0, 1)))
    for threshold, spine_threshold, boundary_threshold, min_area, bridge in candidates:
        params = dict(threshold=threshold, spine_threshold=spine_threshold,
                      boundary_threshold=boundary_threshold, min_area=min_area, bridge=bridge)
        scores = []
        pq_rows = []
        for z in arrays:
            pred = postprocess(z['probabilities'], z['valid'], **params)
            if metric == 'pq':
                pq_rows.append(label_prediction_scores(z['truth_masks'], pred))
            else:
                scores.append(grid_miou_diagnostic(z['truth'] > 0, pred > 0))
        trials.append({'parameters':params, 'score':aggregate_reference_pq(pq_rows) if metric == 'pq' else float(np.mean(scores))})
    best = max(trials, key=lambda t:t['score'])
    return {**best, 'metric':metric, 'scope':'tuning score, not an unbiased performance estimate', 'trials':trials}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--oof', nargs='+', required=True)
    p.add_argument('--metric', choices=['pq', 'grid_miou_diagnostic'], default='pq')
    p.add_argument('--output', default='runs/postprocess.json')
    p.add_argument('--wide', action='store_true', help='Test higher confidence/area cutoffs and unsplit connected components')
    a = p.parse_args()
    result = tune([p for folder in a.oof for p in Path(folder).glob('*.npz')], a.metric, a.wide)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(result, indent=2))
