"""Kaggle GPU entry point. The competition is the only annotation source."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    fold = int(os.environ['SOLAR_FOLD'])
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    started = time.time()
    requirements = [line for line in Path('requirements.txt').read_text().splitlines()
                    if not line.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    with Path('environment.txt').open('w') as stream:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This experiment requires a Kaggle GPU accelerator')
    print(json.dumps({'gpu': torch.cuda.get_device_name(), 'fold': fold}), flush=True)
    annotations, = Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    release = annotations.parent.parent
    images = annotations.parent / 'train_images'
    test_images = release / 'test/test_images'
    # Bound preprocessing process memory, retaining each completed atomic target.
    manifest = root / 'cache/manifest.json'
    while not manifest.exists():
        subprocess.run([sys.executable, '-u', '-m', 'src.dataset', '--annotations', str(annotations),
                        '--images', str(images), '--test-images', str(test_images),
                        '--output', str(root / 'cache'), '--compact-images', '--max-new-records', '100'], check=True)
    experiments = [('segformer_b3', 'full'), ('convnext_unetpp', 'full')]
    if fold == 0:
        experiments += [('unet', 'baseline'), ('segformer_b3', 'no_topology'),
                        ('segformer_b3', 'no_auxiliary')]
    plan = {'fold': fold, 'epochs': 16, 'samples_per_epoch': 800, 'crop': 512,
            'seed': 42, 'experiments': experiments, 'selection': 'training loss only',
            'validation_tta': False, 'final_inference_tta': 'D4 in final inference kernel',
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__}
    Path('experiment.json').write_text(json.dumps(plan, indent=2))
    for architecture, variant in experiments:
        output = root / 'runs' / variant / architecture / f'fold{fold}'
        command = [sys.executable, '-u', '-m', 'src.train', '--manifest', str(manifest),
                   '--architecture', architecture, '--fold', str(fold), '--epochs', '16',
                   '--samples', '800', '--crop', '512', '--batch-size', '1', '--accumulate', '2',
                   '--threads', '2', '--device', 'cuda', '--lr', '0.0001',
                   '--encoder-lr-scale', '0.1', '--focal-alpha', '0.25',
                   '--save-oof-limit', '16', '--output', str(output)]
        if variant == 'baseline':
            command += ['--no-topology', '--no-auxiliary']
        elif variant != 'full':
            command += ['--' + variant.replace('_', '-')]
        subprocess.run(command, check=True)
    Path('complete.json').write_text(json.dumps({'fold': fold, 'seconds': time.time() - started,
                                               'experiments': experiments}, indent=2))


if __name__ == '__main__':
    main()
