"""Score untouched OOF maps with fixed, previously selected postprocessing."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .infer import postprocess, masks_from_labels
from .metrics import binary_scores, grid_miou_diagnostic, label_prediction_scores, aggregate_reference_pq


def score(folder, params):
    rows = []
    for path in sorted(Path(folder).glob('*.npz')):
        with np.load(path) as z:
            probabilities, truth, valid = z['probabilities'], z['truth'], z['valid']
            gt = [np.unpackbits(a, count=truth.size).reshape(truth.shape).astype(bool) for a in z['truth_packed']]
            image_id = str(z['image_id'])
        labels = postprocess(probabilities, valid, **params)
        rows.append({'image_id':image_id, **binary_scores(truth > 0, labels > 0),
                     'grid_miou_diagnostic':grid_miou_diagnostic(truth > 0, labels > 0),
                     **label_prediction_scores(gt, labels, probabilities[0])})
    if not rows:
        raise ValueError('No OOF maps to score')
    frame = pd.DataFrame(rows)
    return {'pq':aggregate_reference_pq(rows), 'annotation_views':len(rows),
            'view_mean_diagnostics':frame.select_dtypes('number').mean().to_dict()}, frame


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--oof', required=True)
    p.add_argument('--params', required=True)
    p.add_argument('--output', default='runs/heldout_validation.json')
    a = p.parse_args()
    params = json.loads(Path(a.params).read_text())['parameters']
    summary, rows = score(a.oof, params)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2))
    rows.to_csv(output.with_suffix('.csv'), index=False)
    print(json.dumps(summary, indent=2))
