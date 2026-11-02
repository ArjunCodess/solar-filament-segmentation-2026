"""Single refined ConvNeXt inference and self-contained source/weight packaging."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main():
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    sys.path.insert(0, str(root))
    requirements = [s for s in Path('requirements.txt').read_text().splitlines()
                    if not s.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import cv2
    import torch
    from src.submission import validate_submission
    torch.set_num_threads(2)
    refined, = Path('/kaggle/input').rglob('refinement_complete.json')
    frozen, = Path('/kaggle/input').rglob('refinement_frozen_policies.json')
    report, = Path('/kaggle/input').rglob('refinement_full_check_complete.json')
    evidence = json.loads(report.read_text())
    if evidence['results']['refined']['pq'] <= evidence['results']['original']['pq']:
        raise ValueError('Refinement did not improve complete fold 1')
    checkpoint = refined.parent/'runs/refine_v2/convnext_unetpp/fold1/best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['config']['fold'] != 1 or state['config']['architecture'] != 'convnext_unetpp':
        raise ValueError('Unexpected portable model identity')
    policy = json.loads(frozen.read_text())['refined']
    test, = [p for p in Path('/kaggle/input').rglob('test_images') if p.is_dir()]
    paths = sorted(test.glob('*.jpeg'))
    if len(paths) != 180:
        raise ValueError('Expected all official test images')
    release = root/'portable_refined'
    release.mkdir(exist_ok=True)
    shutil.copytree(root/'src', release/'src', ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    (release/'weights').mkdir(exist_ok=True)
    # Persist only inference state, retaining the original full-precision model tensors.
    torch.save({'model': state['model'], 'config': state['config']}, release/'weights/model.pt')
    del state
    with (release/'weights/model.pt').open('rb') as stream:
        weight_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    (release/'postprocess.json').write_text(json.dumps(policy, indent=2))
    import importlib.metadata
    names = ('numpy', 'scipy', 'opencv-python-headless', 'scikit-image', 'pandas',
             'matplotlib', 'torch', 'torchvision', 'timm', 'segmentation-models-pytorch', 'pycocotools', 'pytest')
    versions = {name: importlib.metadata.version(name) for name in names}
    (release/'requirements.txt').write_text('\n'.join(f'{name}=={version}' for name, version in versions.items())+'\n')
    shutil.copyfile(root/'requirements.txt', release/'requirements-local-reference.txt')
    result = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], text=True, capture_output=True, check=True)
    (release/'environment-kaggle.txt').write_text(result.stdout)
    started = time.perf_counter()
    command = [sys.executable, '-u', '-m', 'src.submission', '--images', str(test),
        '--checkpoints', 'weights/model.pt', '--params', 'postprocess.json', '--device', 'cuda',
        '--tile', '512', '--tta', '--tta-batch-size', '4', '--output', 'submission.csv',
        '--maps-output', str(root/'efficient_test_maps')]
    subprocess.run(command, cwd=release, check=True)
    elapsed = time.perf_counter()-started
    validation = validate_submission(release/'submission.csv',
                                    {p.stem: cv2.imread(str(p), 0).shape for p in paths})
    provenance = {'checkpoint_sha256': weight_hash, 'source_checkpoint': str(checkpoint),
        'source_fold': 1, 'architecture': 'convnext_unetpp', 'members': 1, 'tta': 'D4',
        'tile': 512, 'tta_batch_size': 4, 'inference_seconds': elapsed,
        'seconds_per_observation': elapsed/len(paths), 'validation': validation,
        'validation_scope': 'Full fold 1 without D4, policy fitted on fold 0; separate earlier kernel',
        'reference_pq_without_d4': evidence['results']['refined']['pq'],
        'portability_scope': 'Executed in a fresh Kaggle session from the packaged local weights and source',
        'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'submission_sha256': hashlib.sha256((release/'submission.csv').read_bytes()).hexdigest()}
    (release/'provenance.json').write_text(json.dumps(provenance, indent=2))
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'cells': [
        {'cell_type': 'markdown', 'metadata': {}, 'source': [
            '# Reproduce the single-model refined candidate\n',
            'Run from this directory after installing requirements.txt. Supply the official competition test_images directory. '
            'Weights are included in weights/model.pt. No training or private Kaggle output is required.\n']},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
            'from pathlib import Path\nimport hashlib, json\n',
            "provenance = json.loads(Path('provenance.json').read_text())\n",
            "with Path('weights/model.pt').open('rb') as stream:\n",
            "    assert hashlib.file_digest(stream, 'sha256').hexdigest() == provenance['checkpoint_sha256']\n"]},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
            'import os, subprocess, sys\n',
            "test_images = Path(os.environ['SOLAR_TEST_IMAGES'])\n",
            "subprocess.run([sys.executable, '-m', 'src.submission', '--images', str(test_images),\n",
            "    '--checkpoints', 'weights/model.pt', '--params', 'postprocess.json', '--device', 'cuda',\n",
            "    '--tile', '512', '--tta', '--tta-batch-size', '4', '--output', 'reproduced.csv'], check=True)\n"]},
        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': [
            'import cv2\nfrom src.submission import validate_submission\n',
            "shapes = {p.stem: cv2.imread(str(p), 0).shape for p in test_images.glob('*.jpeg')}\n",
            "validate_submission('reproduced.csv', shapes)\n"]}]}
    (release/'pipeline.ipynb').write_text(json.dumps(notebook, indent=1))
    (release/'README.md').write_text(
        '# Single-model refined candidate\n\n'
        'Install requirements.txt in Python 3.12 with a compatible CUDA PyTorch build. '
        'Set SOLAR_TEST_IMAGES to the official test_images directory and run pipeline.ipynb. '
        'The notebook loads weights/model.pt and requires no retraining or private outputs.\n\n'
        'This is a candidate, not a claim of a winning solution. provenance.json records validation scope, '
        'measured inference time, hardware, and hashes. The submission has not been assigned a Kaggle score '
        'at package creation. See the project report for training and validation methodology.\n')
    with zipfile.ZipFile(root/'portable_refined.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(release.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.write(path, path.relative_to(release))
    Path('efficient_complete.json').write_text(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
