"""Warm-start two ConvNeXt folds with larger native crops on separate Kaggle GPUs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    sys.path.insert(0, str(root))
    requirements = [line for line in Path('requirements.txt').read_text().splitlines()
                    if not line.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two Kaggle T4 GPUs are required for this fixed two-fold experiment')
    sources = {json.loads(p.read_text())['fold']: p.parent
               for p in Path('/kaggle/input').rglob('complete.json')}
    if not {0, 1}.issubset(sources):
        raise ValueError('Completed source folds 0 and 1 are required')
    annotations, = Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    images = annotations.parent / 'train_images'
    started = time.time()
    processes, streams, provenance = [], [], []
    for fold in (0, 1):
        source = sources[fold]
        manifest = json.loads((source/'cache/manifest.json').read_text())
        for record in manifest['records']:
            for key in ('cache', 'image_cache'):
                record[key] = str(source/'cache'/Path(record[key]).name)
            record['image_path'] = str(images/Path(record['image_path']).name)
        cache = root/f'fold{fold}_cache'
        cache.mkdir(exist_ok=True)
        (cache/'manifest.json').write_text(json.dumps(manifest))
        (cache/'folds.csv').write_bytes((source/'cache/folds.csv').read_bytes())
        checkpoint = source/f'runs/full/convnext_unetpp/fold{fold}/best.pt'
        digest = hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest()
        provenance.append({'fold': fold, 'source_checkpoint': str(checkpoint), 'sha256': digest})
        output = root/f'runs/refine_v2/convnext_unetpp/fold{fold}'
        command = [sys.executable, '-u', '-m', 'src.train', '--manifest', str(cache/'manifest.json'),
                   '--architecture', 'convnext_unetpp', '--fold', str(fold), '--epochs', '12',
                   '--samples', '600', '--crop', '640', '--batch-size', '1', '--accumulate', '2',
                   '--threads', '2', '--workers', '0', '--device', 'cuda', '--lr', '0.00003',
                   '--encoder-lr-scale', '0.1', '--focal-alpha', '0.25', '--seed', '142',
                   '--no-pretrained', '--init-checkpoint', str(checkpoint),
                   '--save-oof-limit', '16', '--output', str(output)]
        stream = (root/f'refine_fold{fold}.log').open('w')
        streams.append(stream)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(fold)}
        processes.append(subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT))
    Path('refinement_provenance.json').write_text(json.dumps(provenance, indent=2))
    with Path('environment.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    codes = [process.wait() for process in processes]
    for stream in streams:
        stream.close()
    if any(codes):
        for fold in (0, 1):
            print((root/f'refine_fold{fold}.log').read_text()[-10000:], flush=True)
        raise RuntimeError(f'Refinement fold exit codes: {codes}')
    Path('refinement_complete.json').write_text(json.dumps({
        'folds': [0, 1], 'epochs_added': 12, 'samples_per_epoch': 600, 'crop': 640,
        'seed': 142, 'seconds': time.time()-started,
        'scope': 'New larger-crop training; fold-0 calibration and fold-1 evaluation still required'}, indent=2))


if __name__ == '__main__':
    main()
