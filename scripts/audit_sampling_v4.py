"""Recompute the paired sampling experiment with instance-weighted morphology gates."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_candidates import compare_paths


def main():
    directory = ROOT/'outputs/kaggle/sampling-v4/solar'
    errors = pd.read_csv(directory/'sampling_v4_instance_errors.csv', dtype={'image_id': str})
    truth = errors[errors.kind == 'truth'].copy()
    truth['small_instances'] = (truth.size_bin == 'small').astype(int)
    truth['small_hits'] = truth['small_instances'] * truth.matched.astype(int)
    for name in ('sampling_control', 'targeted_sampling'):
        rows = pd.read_csv(directory/f'sampling_v4_{name}_full_fold1.csv', dtype={'image_id': str})
        morphology = truth[truth.variant == name].groupby('image_id').agg(
            skeleton_recall_sum=('skeleton_recall', 'sum'), skeleton_instances=('instance', 'size'),
            small_hits=('small_hits', 'sum'), small_instances=('small_instances', 'sum'))
        if set(morphology.index) - set(rows.image_id):
            raise ValueError('Instance diagnostics include an unscored view')
        enriched = rows.merge(morphology, on='image_id', how='left', validate='one_to_one')
        columns = ['skeleton_recall_sum', 'skeleton_instances', 'small_hits', 'small_instances']
        enriched[columns] = enriched[columns].fillna(0)
        enriched.to_csv(directory/f'{name}_audited_views.csv', index=False)
    report = compare_paths(directory/'sampling_control_audited_views.csv',
                           directory/'targeted_sampling_audited_views.csv',
                           directory/'sampling_v4_fold1_cache/folds.csv',
                           directory/'paired_morphology_audit.json', draws=10000)
    print(json.dumps({'decision': report['decision'], 'delta': report['delta'],
                      'gates': report['gates'], 'bootstrap': report['bootstrap']}, indent=2))


if __name__ == '__main__':
    main()
