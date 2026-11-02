"""Paired continuation: area-biased sampling versus instance-balanced foreground and training-only dark negatives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(fold):
    source, = [p.parent for p in Path('/kaggle/input').rglob('complete.json')
               if json.loads(p.read_text())['fold'] == fold]
    refined, = Path('/kaggle/input').rglob('refinement_complete.json')
    checkpoint = refined.parent/f'runs/refine_v2/convnext_unetpp/fold{fold}/best.pt'
    annotations, = Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    manifest = json.loads((source/'cache/manifest.json').read_text())
    for record in manifest['records']:
        for key in ('cache', 'image_cache'):
            record[key] = str(source/'cache'/Path(record[key]).name)
        record['image_path'] = str(annotations.parent/'train_images'/Path(record['image_path']).name)
    cache = Path(f'sampling_v4_fold{fold}_cache')
    cache.mkdir(exist_ok=True)
    (cache/'manifest.json').write_text(json.dumps(manifest))
    (cache/'folds.csv').write_bytes((source/'cache/folds.csv').read_bytes())
    with checkpoint.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    Path(f'sampling_v4_fold{fold}_provenance.json').write_text(json.dumps({
        'fold': fold, 'checkpoint_sha256': digest, 'checkpoint': str(checkpoint),
        'selection': 'Same initialization, seed, sample count, loss, LR, crop scale, physical balancing, and inference; only foreground instance sampling and dark-negative mining differ',
        'calibration': 'Fold 0 only; fold 1 used for comparison, folds 2-4 not examined'}, indent=2))
    mined = cache/'mined/manifest.json'
    subprocess.run([sys.executable, '-u', '-m', 'scripts.prepare_dark_negatives',
        '--manifest', str(cache/'manifest.json'), '--fold', str(fold), '--output', str(mined)], check=True)
    for variant in ('sampling_control', 'targeted_sampling'):
        command = [sys.executable, '-u', '-m', 'src.train', '--manifest', str(mined),
            '--architecture', 'convnext_unetpp', '--fold', str(fold), '--epochs', '8',
            '--samples', '800', '--crop', '640', '--scale-min', '1', '--scale-max', '1',
            '--balance-physical', '--batch-size', '1', '--accumulate', '2', '--threads', '2',
            '--workers', '0', '--device', 'cuda', '--lr', '.00003', '--encoder-lr-scale', '.1',
            '--focal-alpha', '.25', '--seed', '342', '--no-pretrained',
            '--init-checkpoint', str(checkpoint), '--save-oof-limit', '16',
            '--output', f'runs/sampling_v4/{variant}/fold{fold}']
        if variant == 'targeted_sampling':
            command += ['--instance-balanced', '--dark-negative-probability', '.2']
        with Path(f'sampling_v4_{variant}_fold{fold}.log').open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    sys.path.insert(0, str(root))
    requirements = [s for s in Path('requirements.txt').read_text().splitlines()
                    if not s.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import numpy as np
    import pandas as pd
    import torch
    from src.calibration import component_trials
    from src.dataset import load_image_cache
    from src.error_analysis import instance_error_rows
    from src.infer import postprocess, tiled_predict
    from src.metrics import aggregate_reference_pq, label_prediction_scores
    from src.models import build_model
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two T4 GPUs required by the paired sampling experiment')
    started = time.time()
    processes = [subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(fold)],
                  env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(fold)}) for fold in (0, 1)]
    codes = [process.wait() for process in processes]
    if any(codes):
        for path in sorted(root.glob('sampling_v4_*.log')):
            print(path.name, path.read_text()[-6000:], flush=True)
        raise RuntimeError(f'Paired training exit codes: {codes}')
    torch.set_num_threads(2)
    policies = {}
    # Calibrate both variants before any frozen fold-1 result is computed.
    for variant in ('sampling_control', 'targeted_sampling'):
        trials = {}
        for path in sorted(Path(f'runs/sampling_v4/{variant}/fold0/oof').glob('*.npz')):
            with np.load(path) as data:
                valid = data['valid'].astype(bool)
                truth = [np.unpackbits(mask, count=valid.size).reshape(valid.shape).astype(bool)
                         for mask in data['truth_packed']]
                for row in component_trials(data['probabilities'][0].astype(np.float32), valid, truth):
                    key = json.dumps(row['parameters'], sort_keys=True)
                    trials.setdefault(key, []).append(row)
        if not trials:
            raise ValueError('No calibration maps')
        table = [{'parameters': json.loads(key), 'pq': aggregate_reference_pq(rows)}
                 for key, rows in trials.items()]
        policies[variant] = max(table, key=lambda row: row['pq'])
        Path(f'sampling_v4_{variant}_calibration.json').write_text(json.dumps(table, indent=2))
    Path('sampling_v4_frozen_policies.json').write_text(json.dumps(policies, indent=2))
    manifest = json.loads(Path('sampling_v4_fold1_cache/manifest.json').read_text())
    folds = pd.read_csv('sampling_v4_fold1_cache/folds.csv', dtype={'image_id': str}).set_index('image_id').fold
    records = sorted([r for r in manifest['records'] if folds[str(r['id'])] == 1],
                     key=lambda r: (r['sha256'], str(r['id'])))
    results, errors = {}, []
    for variant in ('sampling_control', 'targeted_sampling'):
        state = torch.load(f'runs/sampling_v4/{variant}/fold1/best.pt', map_location='cpu', weights_only=False)
        model = build_model('convnext_unetpp', False)
        model.load_state_dict(state['model'])
        model.cuda().eval()
        del state
        rows, previous, maps = [], None, None
        for position, record in enumerate(records):
            if record['sha256'] != previous:
                image, valid = load_image_cache(record)
                maps = tiled_predict(model, image, 'cuda', 640, 128, False)
                labels = postprocess(maps, valid, **policies[variant]['parameters'])
                previous = record['sha256']
            with np.load(record['cache']) as data:
                truth = [np.unpackbits(mask, count=valid.size).reshape(valid.shape).astype(bool)
                         for mask in data['instance_masks_packed']]
            rows.append({'image_id': str(record['id']), **label_prediction_scores(truth, labels)})
            errors.extend({**row, 'variant': variant} for row in instance_error_rows(
                truth, labels, maps[0], image[1], valid, record['id']))
            if (position+1) % 20 == 0:
                print(json.dumps({'variant': variant, 'frozen_validation_views': position+1}), flush=True)
        pd.DataFrame(rows).to_csv(f'sampling_v4_{variant}_full_fold1.csv', index=False)
        results[variant] = {'pq': aggregate_reference_pq(rows), 'views': len(rows),
                            'parameters': policies[variant]['parameters']}
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(errors).to_csv('sampling_v4_instance_errors.csv', index=False)
    Path('sampling_v4_complete.json').write_text(json.dumps({'results': results, 'seconds': time.time()-started,
        'scope': 'Full fold 1, no D4, 640 tiles; fold-0-frozen extraction. Both variants use physical balancing, fixed-scale 640 crops and unchanged topology/auxiliary losses. The targeted variant samples instances uniformly instead of foreground pixels and assigns 20 percent of crops to training-only dark distractors. No automatic test inference or submission.'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.worker is None:
        main()
    else:
        os.chdir('/kaggle/working/solar')
        sys.path.insert(0, os.getcwd())
        worker(args.worker)
