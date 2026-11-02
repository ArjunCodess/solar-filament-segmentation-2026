"""Complete-fold quality-filter validation, then gated test inference."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(rank):
    import numpy as np
    import pandas as pd
    import torch
    from src.dataset import load_image_cache
    from src.error_analysis import instance_error_rows
    from src.infer import tiled_predict, postprocess
    from src.metrics import label_prediction_scores
    from src.models import build_model
    from src.quality import filter_quality
    torch.set_num_threads(2)
    sources = {json.loads(p.read_text())['fold']: p.parent
               for p in Path('/kaggle/input').rglob('complete.json')}
    quality_file, = Path('/kaggle/input').rglob('quality_model.json')
    quality = json.loads(quality_file.read_text())
    shape_file, = Path('/kaggle/input').rglob('shape_policies.json')
    shape = json.loads(shape_file.read_text())['shape_selected']
    if shape['convnext_weight'] != .5:
        raise ValueError('This experiment expects the selected equal-weight shape policy')
    annotations, = Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')

    class Ensemble(torch.nn.Module):
        def __init__(self, members):
            super().__init__()
            self.members = torch.nn.ModuleList(members)

        def forward(self, x, context):
            probability = sum(m(x, context).float().sigmoid() for m in self.members)/len(self.members)
            return torch.logit(probability.clamp(1e-6, 1-1e-6))

    for fold in ((1, 3) if rank == 0 else (2, 4)):
        source = sources[fold]
        manifest = json.loads((source/'cache/manifest.json').read_text())
        assignments = pd.read_csv(source/'cache/folds.csv', dtype={'image_id': str}).set_index('image_id').fold
        records = sorted([r for r in manifest['records'] if assignments[str(r['id'])] == fold],
                         key=lambda r: (r['sha256'], str(r['id'])))
        for record in records:
            for key in ('cache', 'image_cache'):
                record[key] = str(source/'cache'/Path(record[key]).name)
            record['image_path'] = str(annotations.parent/'train_images'/Path(record['image_path']).name)
        members = []
        for architecture in ('segformer_b3', 'convnext_unetpp'):
            state = torch.load(source/f'runs/full/{architecture}/fold{fold}/best.pt', map_location='cpu', weights_only=False)
            if state['config']['fold'] != fold:
                raise ValueError('Checkpoint fold mismatch')
            model = build_model(architecture, False)
            model.load_state_dict(state['model'])
            members.append(model.cuda().eval())
            del state
        model = Ensemble(members).cuda().eval()
        previous, maps, rows, errors = None, None, [], []
        for position, record in enumerate(records):
            if record['sha256'] != previous:
                image, valid = load_image_cache(record)
                maps = tiled_predict(model, image, 'cuda', 512, 128, False)
                baseline = postprocess(maps, valid, mode='components', threshold=.5, seed_threshold=.65, min_area=256)
                proposals = postprocess(maps, valid, **quality['parameters'])
                filtered = filter_quality(proposals, maps, image[1], quality)
                grouped = postprocess(maps, valid, **shape['parameters'])
                previous = record['sha256']
            with np.load(record['cache']) as data:
                truth = [np.unpackbits(mask, count=valid.size).reshape(valid.shape).astype(bool)
                         for mask in data['instance_masks_packed']]
            for name, labels in [('baseline', baseline), ('quality', filtered), ('shape', grouped)]:
                rows.append({'policy': name, 'fold': fold, 'image_id': str(record['id']),
                             **label_prediction_scores(truth, labels)})
                if name != 'shape':
                    errors.extend({**row, 'policy': name, 'fold': fold} for row in instance_error_rows(
                        truth, labels, maps[0], image[1], valid, record['id']))
            if position % 30 == 0:
                pd.DataFrame(rows).to_csv(f'quality_full_fold{fold}.csv', index=False)
                print(json.dumps({'fold': fold, 'views': position+1}), flush=True)
        pd.DataFrame(rows).to_csv(f'quality_full_fold{fold}.csv', index=False)
        pd.DataFrame(errors).to_csv(f'quality_errors_fold{fold}.csv', index=False)
        del model, members
        torch.cuda.empty_cache()


def main():
    root = Path('/kaggle/working/solar')
    os.chdir(root)
    sys.path.insert(0, str(root))
    requirements = [s for s in Path('requirements.txt').read_text().splitlines()
                    if not s.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import cv2
    import pandas as pd
    import torch
    from src.metrics import aggregate_reference_pq
    from src.submission import validate_submission
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Two GPUs required for this fixed validation/inference budget')
    started = time.time()
    processes = [subprocess.Popen([sys.executable, '-u', __file__, '--worker', str(rank)],
                  env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(rank)}) for rank in (0, 1)]
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f'Validation exit codes: {codes}')
    rows = pd.concat([pd.read_csv(f'quality_full_fold{fold}.csv', dtype={'image_id': str})
                      for fold in (1, 2, 3, 4)], ignore_index=True)
    errors = pd.concat([pd.read_csv(f'quality_errors_fold{fold}.csv') for fold in (1, 2, 3, 4)], ignore_index=True)
    summaries = {}
    for name, frame in rows.groupby('policy'):
        if len(frame) != 905 or not frame.image_id.is_unique:
            raise ValueError('Expected complete unique held-out views')
        summaries[name] = {'pq': aggregate_reference_pq(frame.to_dict('records')), 'views': len(frame)}
    truth = errors[errors.kind == 'truth']
    strata = truth.groupby(['policy', 'size_bin']).agg(instances=('matched', 'size'),
        hit_rate=('matched', 'mean'), skeleton_recall=('skeleton_recall', 'mean')).reset_index()
    strata.to_csv('quality_full_errors_by_size.csv', index=False)
    recall = truth.groupby('policy').skeleton_recall.mean()
    small = truth[truth.size_bin == 'small'].groupby('policy').matched.mean()
    promoted = (summaries['quality']['pq'] >= summaries['baseline']['pq']+.005 and
                recall['quality'] >= recall['baseline']-.01 and
                small['quality'] >= small['baseline']-.02)
    report = {'results': summaries, 'promoted': bool(promoted),
        'gate': 'PQ gain >= .005; mean skeleton recall loss <= .01; small-instance hit-rate loss <= .02',
        'scope': 'Complete folds 1-4, no D4; every policy fitted on fold 0 before this check',
        'seconds_validation': time.time()-started}
    Path('quality_full_validation.json').write_text(json.dumps(report, indent=2))
    if not promoted:
        Path('quality_full_complete.json').write_text(json.dumps(report, indent=2))
        return
    quality_file, = Path('/kaggle/input').rglob('quality_model.json')
    quality = json.loads(quality_file.read_text())
    Path('quality_policy.json').write_text(json.dumps({'parameters': quality['parameters']}, indent=2))
    sources = {json.loads(p.read_text())['fold']: p.parent
               for p in Path('/kaggle/input').rglob('complete.json')}
    checkpoints = [sources[fold]/f'runs/full/{architecture}/fold{fold}/best.pt'
                   for fold in range(5) for architecture in ('segformer_b3', 'convnext_unetpp')]
    annotations, = Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json')
    paths = sorted((annotations.parent.parent/'test/test_images').glob('*.jpeg'))
    if len(paths) != 180:
        raise ValueError('Unexpected test set')
    processes = []
    for rank in (0, 1):
        shard = root/f'quality_test_shard{rank}'
        shard.mkdir(exist_ok=True)
        for path in paths[rank::2]:
            (shard/path.name).symlink_to(path)
        command = [sys.executable, '-u', '-m', 'src.submission', '--images', str(shard),
            '--checkpoints', *map(str, checkpoints), '--params', 'quality_policy.json',
            '--quality-model', str(quality_file), '--device', f'cuda:{rank}', '--tta',
            '--tta-batch-size', '4', '--output', f'quality_shard{rank}.csv',
            '--maps-output', str(root/f'quality_test_maps{rank}')]
        processes.append(subprocess.Popen(command))
    codes = [p.wait() for p in processes]
    if any(codes):
        raise RuntimeError(f'Inference exit codes: {codes}')
    predictions = []
    for rank in (0, 1):
        with Path(f'quality_shard{rank}.csv').open(newline='') as stream:
            predictions.extend(csv.DictReader(stream))
    with Path('submission.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'])
        writer.writeheader()
        writer.writerows(sorted(predictions, key=lambda row: row['filament_id']))
    report['submission'] = validate_submission('submission.csv', {p.stem: cv2.imread(str(p), 0).shape for p in paths})
    report['seconds_total'] = time.time()-started
    Path('quality_full_complete.json').write_text(json.dumps(report, indent=2))


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
