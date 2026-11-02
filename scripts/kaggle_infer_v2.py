"""Score and infer the fold-0-frozen weighted ensemble; no test-label inputs."""
import csv
import hashlib
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
    import cv2
    import pandas as pd
    import torch
    from src.models import build_model
    from src.train import evaluate
    from src.metrics import aggregate_reference_pq
    from src.submission import validate_submission
    torch.set_num_threads(2)
    sources = {json.loads(p.read_text())['fold']: p.parent
               for p in Path('/kaggle/input').rglob('complete.json')}
    if set(sources) != set(range(5)):
        raise ValueError('Five completed source folds are required')
    calibration, = Path('/kaggle/input').rglob('calibration_v2.json')
    policy = json.loads(calibration.read_text())
    Path('postprocess.json').write_text(json.dumps(policy, indent=2))
    weight = policy['convnext_weight']
    architectures = [(name, value) for name, value in
                     [('segformer_b3', 1-weight), ('convnext_unetpp', weight)] if value > 0]
    checkpoints = [(sources[fold]/f'runs/full/{name}/fold{fold}/best.pt', value)
                   for fold in range(5) for name, value in architectures]
    provenance = []
    for path, value in checkpoints:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        provenance.append({'path': str(path), 'weight': value, 'sha256': digest})
    Path('inference_provenance.json').write_text(json.dumps({
        'calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
        'members': provenance, 'tta': 'D4', 'tta_batch_size': 4}, indent=2))
    annotations, = Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    images = annotations.parent/'train_images'
    test = annotations.parent.parent/'test/test_images'

    class WeightedEnsemble(torch.nn.Module):
        def __init__(self, members, weights):
            super().__init__()
            self.members = torch.nn.ModuleList(members)
            self.weights = weights

        def forward(self, x, context):
            probability = sum(w*m(x, context).float().sigmoid()
                              for m, w in zip(self.members, self.weights))/sum(self.weights)
            return torch.logit(probability.clamp(1e-6, 1-1e-6))

    all_rows = []
    for fold in (1, 2, 3, 4):
        source = sources[fold]
        manifest = json.loads((source/'cache/manifest.json').read_text())
        assignments = pd.read_csv(source/'cache/folds.csv', dtype={'image_id': str}).set_index('image_id').fold
        records = [record for record in manifest['records'] if assignments[str(record['id'])] == fold]
        for record in records:
            for key in ('cache', 'image_cache'):
                record[key] = str(source/'cache'/Path(record[key]).name)
            record['image_path'] = str(images/Path(record['image_path']).name)
        members, weights = [], []
        for name, value in architectures:
            state = torch.load(source/f'runs/full/{name}/fold{fold}/best.pt', map_location='cpu', weights_only=False)
            if state['config']['fold'] != fold:
                raise ValueError('Checkpoint fold mismatch')
            model = build_model(name, False)
            model.load_state_dict(state['model'])
            members.append(model.cuda().eval())
            weights.append(value)
            del state
        model = WeightedEnsemble(members, weights).cuda().eval()
        metrics, rows = evaluate(model, records, 'cuda', 512, False,
                                 root/f'oof_v2/fold{fold}', save_limit=1,
                                 parameters=policy['parameters'])
        all_rows.extend([{**row, 'fold': fold} for row in rows])
        pd.DataFrame(all_rows).to_csv('full_heldout_v2.csv', index=False)
        print(json.dumps({'fold': fold, 'metrics': metrics}), flush=True)
        del model, members
        torch.cuda.empty_cache()
    Path('full_heldout_v2.json').write_text(json.dumps({
        'pq': aggregate_reference_pq(all_rows), 'views': len(all_rows),
        'scope': 'Full folds 1-4; fold-0-frozen policy; no validation D4'}, indent=2))
    paths = sorted(test.glob('*.jpeg'))
    if len(paths) != 180:
        raise ValueError('Expected all 180 official test observations')
    count = min(2, torch.cuda.device_count())
    if not count:
        raise RuntimeError('Kaggle GPU required')
    processes = []
    for rank in range(count):
        shard = root/f'test_shard{rank}'
        shard.mkdir(exist_ok=True)
        for path in paths[rank::count]:
            (shard/path.name).symlink_to(path)
        command = [sys.executable, '-u', '-m', 'src.submission', '--images', str(shard),
                   '--checkpoints', *[str(path) for path, _ in checkpoints],
                   '--weights', *[str(value) for _, value in checkpoints],
                   '--params', str(root/'postprocess.json'), '--device', f'cuda:{rank}',
                   '--tta', '--tta-batch-size', '4', '--output', str(root/f'shard{rank}.csv')]
        processes.append(subprocess.Popen(command))
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f'Inference shard exit codes: {codes}')
    rows = []
    for rank in range(count):
        with Path(f'shard{rank}.csv').open(newline='') as stream:
            rows.extend(csv.DictReader(stream))
    with Path('submission.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row['filament_id']))
    shapes = {path.stem: cv2.imread(str(path), 0).shape for path in paths}
    validation = validate_submission('submission.csv', shapes)
    Path('submission.validation.json').write_text(json.dumps(validation, indent=2))
    Path('inference_v2_complete.json').write_text(json.dumps(validation, indent=2))


if __name__ == '__main__':
    main()
