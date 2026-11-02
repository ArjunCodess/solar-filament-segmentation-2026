"""Choose extraction and backbone weights on fold 0; freeze before other folds."""
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    sys.path.insert(0, str(root))
    requirements = [line for line in Path('requirements.txt').read_text().splitlines()
                    if not line.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import numpy as np
    import pandas as pd
    import torch
    from src.calibration import component_trials
    from src.metrics import aggregate_reference_pq, label_prediction_scores, binary_scores
    from src.infer import postprocess
    torch.set_num_threads(2)
    sources = {json.loads(p.read_text())['fold']: p.parent
               for p in Path('/kaggle/input').rglob('complete.json')}
    if set(sources) != set(range(5)):
        raise ValueError('Five completed fold kernels are required')
    trials = {}
    calibration_ids = []
    for fold in range(5):
        first = sources[fold] / f'runs/full/segformer_b3/fold{fold}/oof'
        second = sources[fold] / f'runs/full/convnext_unetpp/fold{fold}/oof'
        rows = []
        for position, path in enumerate(sorted(first.glob('*.npz'))):
            with np.load(path) as a, np.load(second/path.name) as b:
                if str(a['image_id']) != str(b['image_id']) or not np.array_equal(a['truth_packed'], b['truth_packed']):
                    raise ValueError('OOF annotation views differ')
                segformer = a['probabilities'].astype(np.float32)
                convnext = b['probabilities'].astype(np.float32)
                valid = a['valid'].astype(bool)
                truth = [np.unpackbits(mask, count=valid.size).reshape(valid.shape).astype(bool)
                         for mask in a['truth_packed']]
                image_id = str(a['image_id'])
            if fold == 0:
                calibration_ids.append(image_id)
                for weight in (0., .25, .5, .75, 1.):
                    for row in component_trials((1-weight)*segformer[0]+weight*convnext[0], valid, truth):
                        key = json.dumps({'convnext_weight': weight, 'parameters': row['parameters']}, sort_keys=True)
                        trials.setdefault(key, []).append(row)
            else:
                for name, weight, params in policies:
                    maps = (1-weight)*segformer+weight*convnext
                    labels = postprocess(maps, valid, **params)
                    score = label_prediction_scores(truth, labels)
                    binary = binary_scores(np.logical_or.reduce(truth) if truth else np.zeros_like(valid), labels > 0)
                    rows.append({'policy': name, 'fold': fold, 'image_id': image_id,
                                 **score, **binary})
            print(json.dumps({'stage': 'calibration' if fold == 0 else 'heldout_subset',
                              'fold': fold, 'view': position+1, 'image_id': image_id}), flush=True)
        if fold == 0:
            if not calibration_ids:
                raise ValueError('No fold-0 calibration maps')
            table = [{**json.loads(key), 'pq': aggregate_reference_pq(values),
                      'dice': float(np.mean([r['dice'] for r in values]))} for key, values in trials.items()]
            best = max(table, key=lambda row: (row['pq'], row['dice']))
            equal = max((r for r in table if r['convnext_weight'] == .5), key=lambda row: (row['pq'], row['dice']))
            selected = {**best, 'scope': 'Selected on saved fold-0 OOF maps only; validation without D4',
                        'calibration_image_ids': calibration_ids, 'trials': table}
            Path('calibration_v2.json').write_text(json.dumps(selected, indent=2))
            policies = [('selected_v2', best['convnext_weight'], best['parameters']),
                        ('equal_components', .5, equal['parameters']),
                        ('equal_default', .5, {}), ('convnext_default', 1., {})]
            Path('frozen_policies_v2.json').write_text(json.dumps(policies, indent=2))
            all_rows = []
            print(json.dumps({'stage': 'parameters_frozen', **best}), flush=True)
        else:
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv('heldout_subset_v2.csv', index=False)
    summaries = []
    for policy in sorted({row['policy'] for row in all_rows}):
        subset = [row for row in all_rows if row['policy'] == policy]
        summaries.append({'policy': policy, 'pq': aggregate_reference_pq(subset),
                          'dice_view_mean': float(np.mean([r['dice'] for r in subset])),
                          'views': len(subset), 'scope': 'Saved OOF subset only, folds 1-4; no D4'})
    pd.DataFrame(summaries).to_csv('heldout_summary_v2.csv', index=False)
    Path('calibration_complete.json').write_text(json.dumps({'selected': best, 'heldout': summaries}, indent=2))


if __name__ == '__main__':
    main()
