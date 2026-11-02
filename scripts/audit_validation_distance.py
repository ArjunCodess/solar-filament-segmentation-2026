"""Describe existing OOF performance by temporal distance, without retraining."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.metrics import aggregate_reference_pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', required=True)
    parser.add_argument('--metrics', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--exclude-fold', type=int, default=0, help='Exclude the postprocessing-development fold')
    args = parser.parse_args()
    folds = pd.read_csv(args.folds, dtype={'image_id': str})
    rows = pd.read_csv(args.metrics, dtype={'image_id': str})
    if not rows.image_id.is_unique or not folds.image_id.is_unique:
        raise ValueError('Expected one row per annotation view')
    folds['timestamp'] = pd.to_datetime(folds.date)
    eligible = set(folds.loc[folds.fold != args.exclude_fold, 'image_id'])
    rows = rows[rows.image_id.isin(eligible)].drop(columns=['fold'], errors='ignore')
    physical = folds.drop_duplicates('file_name')
    distances = {}
    for fold in sorted(folds.fold.unique()):
        train = physical[physical.fold != fold].timestamp.to_numpy()
        for record in physical[physical.fold == fold].itertuples():
            distances[record.file_name] = float(np.min(np.abs(train-record.timestamp.to_datetime64()))/np.timedelta64(1, 'D'))
    merged = rows.merge(folds[['image_id', 'fold', 'file_name']], on='image_id', validate='one_to_one')
    if len(merged) != len(rows):
        raise ValueError('Metric view missing from split manifest')
    merged['nearest_training_days'] = merged.file_name.map(distances)
    merged['distance_bin'] = pd.cut(merged.nearest_training_days, [0, 3, 7, 14, 30, np.inf],
        labels=['<=3 days', '3-7 days', '7-14 days', '14-30 days', '>30 days'], right=True)
    summary = []
    for name, group in merged.groupby('distance_bin', observed=True):
        summary.append({'distance_bin': str(name), 'views': len(group),
            'physical_observations': group.file_name.nunique(),
            'pq': aggregate_reference_pq(group.to_dict('records'))})
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.with_suffix('.csv').write_text(merged.to_csv(index=False))
    destination.write_text(json.dumps({'groups': summary, 'excluded_fold': args.exclude_fold,
        'minimum_distance_days': min(distances.values()),
        'scope': 'Stratification of existing predictions, not a longer-gap retraining experiment; '
                 'image difficulty and observation date can confound differences.'}, indent=2))
    print(destination.read_text())


if __name__ == '__main__':
    main()
