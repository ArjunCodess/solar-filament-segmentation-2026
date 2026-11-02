"""Recompute frozen-policy PQ and paired temporal-block uncertainty."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    directory = root/'outputs/kaggle/quality-full-v3/solar'
    rows = pd.concat([pd.read_csv(directory/f'quality_full_fold{fold}.csv')
                      for fold in range(1, 5)], ignore_index=True)
    folds = pd.read_csv(root/'outputs/kaggle/refine-v2/solar/fold0_cache/folds.csv')
    expected = set(folds.loc[folds.fold != 0, 'image_id'])
    totals = {}
    blocks = {}
    columns = ['pq_iou_sum', 'pq_tp', 'pq_fp', 'pq_fn']
    for policy, frame in rows.groupby('policy'):
        if len(frame) != 905 or frame.image_id.duplicated().any() or set(frame.image_id) != expected:
            raise ValueError(f'Incomplete or duplicate held-out identities: {policy}')
        merged = frame.merge(folds[['image_id', 'fold', 'temporal_group']],
                             on=['image_id', 'fold'], validate='one_to_one')
        values = merged[columns].sum().to_numpy()
        totals[policy] = float(values[0]/(values[1]+.5*(values[2]+values[3])))
        blocks[policy] = merged.groupby(['fold', 'temporal_group'])[columns].sum().sort_index()
    index = blocks['baseline'].index
    if any(not frame.index.equals(index) for frame in blocks.values()):
        raise ValueError('Policies do not share temporal blocks')
    rng = np.random.default_rng(20261102)
    samples = {policy: [] for policy in blocks}
    fold_ids = index.get_level_values('fold').to_numpy()
    for _ in range(10000):
        selected = np.concatenate([rng.choice(np.flatnonzero(fold_ids == fold),
            size=np.sum(fold_ids == fold), replace=True) for fold in range(1, 5)])
        for policy, frame in blocks.items():
            values = frame.to_numpy()[selected].sum(axis=0)
            samples[policy].append(values[0]/(values[1]+.5*(values[2]+values[3])))
    report = {'views_per_policy': 905, 'temporal_blocks': len(index), 'pq': totals,
        'bootstrap': {'draws': 10000, 'seed': 20261102,
            'method': 'Paired temporal-group cluster resampling within each held-out fold; '
                      'annotator views remain together. Fixed trained models and policies.',
            'limitation': 'Does not estimate retraining, policy-selection, or hidden-server uncertainty.',
            'pq_95_percentile': {p: np.quantile(v, [.025, .975]).tolist() for p, v in samples.items()},
            'delta_vs_baseline_95_percentile': {p: np.quantile(
                np.array(v)-np.array(samples['baseline']), [.025, .975]).tolist()
                for p, v in samples.items() if p != 'baseline'}}}
    recorded = json.loads((directory/'quality_full_validation.json').read_text())
    for policy, pq in totals.items():
        if abs(recorded['results'][policy]['pq']-pq) > 1e-12:
            raise ValueError('Downloaded summary does not match per-view counts')
    (directory/'independent_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
