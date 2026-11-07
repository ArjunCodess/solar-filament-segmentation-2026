"""Run attributed upstream decoding on our OOF maps; select on calibration only."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.infer import postprocess
from src.metrics import aggregate_reference_pq, label_prediction_scores
from src.error_analysis import instance_error_rows


def run(calibration, evaluation, upstream, baseline, output):
    sys.path.insert(0, str(upstream))
    from solarseg.postprocess import PostConfig, instances_from_prob
    calibration_paths = sorted(calibration.glob('*.npz'))
    evaluation_paths = sorted(evaluation.glob('*.npz'))
    if not calibration_paths or not evaluation_paths:
        raise ValueError('Both calibration and evaluation require saved OOF maps')
    # Raw native-image hashes keep repeated annotator views together at this boundary.
    identities = []
    for paths in (calibration_paths, evaluation_paths):
        hashes, ids = set(), set()
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                image_id = str(data['image_id'])
                if image_id in ids:
                    raise ValueError('Duplicate annotation-view identity')
                ids.add(image_id)
                hashes.add(hashlib.sha256(data['raw'].tobytes()).hexdigest())
        identities.append((hashes, ids))
    if identities[0][0] & identities[1][0] or identities[0][1] & identities[1][1]:
        raise ValueError('Calibration and evaluation physical observations overlap')
    configs = [PostConfig(low_threshold=low, threshold=high, min_area=area,
                          merge_dist=distance, max_instances=1000000)
               for low, high, area, distance in itertools.product(
                   (.4, .5), (.8, .95), (64, 256), (0, 10))]
    trials = [[] for _ in configs]
    for path in calibration_paths:
        with np.load(path, allow_pickle=False) as data:
            valid = data['valid'].astype(bool)
            truth = [np.unpackbits(m, count=valid.size).reshape(valid.shape).astype(bool)
                     for m in data['truth_packed']]
            for config, rows in zip(configs, trials):
                labels = instances_from_prob(data['probabilities'][0].astype(np.float32), valid, config)
                rows.append(label_prediction_scores(truth, labels))
    scores = [aggregate_reference_pq(rows) for rows in trials]
    selected = int(np.argmax(scores))
    output.mkdir(parents=True, exist_ok=True)
    policy = {'upstream': 'FarnooshMemari/solar-filament-segmentation, run3, MIT',
              'upstream_source_sha256': hashlib.sha256((upstream/'solarseg/postprocess.py').read_bytes()).hexdigest(),
              'parameters': vars(configs[selected]), 'calibration_pq': scores[selected],
              'calibration_views': len(calibration_paths), 'grid_size': len(configs),
              'adaptation': 'No per-image instance cap; no foreground closing or hole filling'}
    (output/'frozen_policy.json').write_text(json.dumps(policy, indent=2))
    rows, errors = [], []
    for path in evaluation_paths:
        with np.load(path, allow_pickle=False) as data:
            valid = data['valid'].astype(bool)
            truth = [np.unpackbits(m, count=valid.size).reshape(valid.shape).astype(bool)
                     for m in data['truth_packed']]
            maps = data['probabilities'].astype(np.float32)
            predictions = {'baseline': postprocess(maps, valid, **baseline),
                           'public_decoder': instances_from_prob(maps[0], valid, configs[selected])}
            for name, labels in predictions.items():
                image_id = str(data['image_id'])
                rows.append({'policy': name, 'image_id': image_id,
                             **label_prediction_scores(truth, labels)})
                errors.extend({'policy': name, **row} for row in instance_error_rows(
                    truth, labels, maps[0], data['raw'], valid, image_id))
    pd.DataFrame(rows).to_csv(output/'per_view.csv', index=False)
    pd.DataFrame(errors).to_csv(output/'instance_errors.csv', index=False)
    report = {'pq': {name: aggregate_reference_pq([r for r in rows if r['policy'] == name])
                     for name in predictions}, 'evaluation_views': len(evaluation_paths),
              'scope': 'Saved-map screen only; not complete-fold validation or a submission',
              'test_inference': False, 'promotion': False}
    (output/'complete.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--baseline-policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    policy = json.loads(args.baseline_policy.read_text())
    run(args.calibration, args.evaluation, args.upstream, policy['parameters'], args.output)
