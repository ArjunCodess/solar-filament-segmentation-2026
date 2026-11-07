"""Complete fold-1 validation of the submitted D4 recipe and continuation control."""
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
    requirements = [s for s in Path('requirements.txt').read_text().splitlines()
                    if not s.startswith(('torch==', 'torchvision=='))]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', *requirements], check=True)
    import numpy as np
    import pandas as pd
    import torch
    from src.dataset import load_image_cache
    from src.error_analysis import instance_error_rows
    from src.infer import postprocess, tiled_predict
    from src.metrics import aggregate_reference_pq, label_prediction_scores
    from src.models import build_model

    torch.set_num_threads(2)
    original, = Path('/kaggle/input').rglob('complete.json')
    if json.loads(original.read_text())['fold'] != 1:
        raise ValueError('Expected original fold 1 source')
    refined, = Path('/kaggle/input').rglob('refinement_complete.json')
    policies, = Path('/kaggle/input').rglob('refinement_frozen_policies.json')
    sampling, = Path('/kaggle/input').rglob('sampling_v4_complete.json')
    sampling_policies = json.loads((sampling.parent/'sampling_v4_frozen_policies.json').read_text())
    manifest = json.loads((original.parent/'cache/manifest.json').read_text())
    folds = pd.read_csv(original.parent/'cache/folds.csv', dtype={'image_id': str})
    folds.to_csv('exact_validation_folds.csv', index=False)
    heldout = set(folds.loc[folds.fold == 1, 'image_id'])
    records = sorted([r for r in manifest['records'] if str(r['id']) in heldout],
                     key=lambda r: (r['sha256'], str(r['id'])))
    if len(records) != 215:
        raise ValueError('Expected complete 215-view fold 1')
    for record in records:
        for key in ('cache', 'image_cache'):
            record[key] = str(original.parent/'cache'/Path(record[key]).name)
    recipes = [
        ('refined_no_d4', refined.parent/'runs/refine_v2/convnext_unetpp/fold1/best.pt',
         json.loads(policies.read_text())['refined']['parameters'], 512, False),
        ('refined_d4', refined.parent/'runs/refine_v2/convnext_unetpp/fold1/best.pt',
         json.loads(policies.read_text())['refined']['parameters'], 512, True),
        ('control_d4', sampling.parent/'runs/sampling_v4/sampling_control/fold1/best.pt',
         sampling_policies['sampling_control']['parameters'], 640, True)]
    started, results = time.time(), {}
    for name, checkpoint, policy, tile, tta in recipes:
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model = build_model('convnext_unetpp', False)
        model.load_state_dict(state['model'], strict=True)
        del state
        model.cuda().eval()
        previous, rows, errors, saved = None, [], [], set()
        for index, record in enumerate(records):
            if previous != record['sha256']:
                image, valid = load_image_cache(record)
                maps = tiled_predict(model, image, 'cuda', tile, 128, tta, tta_batch_size=4)
                labels = postprocess(maps, valid, **policy)
                previous = record['sha256']
            with np.load(record['cache']) as cache:
                packed = cache['instance_masks_packed']
                truth = [np.unpackbits(m, count=valid.size).reshape(valid.shape).astype(bool)
                         for m in packed]
            rows.append({'image_id': str(record['id']), **label_prediction_scores(truth, labels)})
            errors.extend(instance_error_rows(truth, labels, maps[0], image[1], valid, record['id']))
            if record['sha256'] in saved or len(saved) < 4:
                saved.add(record['sha256'])
                folder = Path(name)
                folder.mkdir(exist_ok=True)
                np.savez_compressed(folder/f'{index}.npz', probabilities=maps.astype(np.float16),
                    valid=valid, raw=image[1], image_id=str(record['id']),
                    truth_packed=packed)
            if (index+1) % 20 == 0:
                print(json.dumps({'recipe': name, 'views': index+1}), flush=True)
        pd.DataFrame(rows).to_csv(f'{name}_views.csv', index=False)
        pd.DataFrame(errors).to_csv(f'{name}_errors.csv', index=False)
        results[name] = {'pq': aggregate_reference_pq(rows), 'views': len(rows),
                         'parameters': policy, 'tile': tile, 'd4': tta}
        del model
        torch.cuda.empty_cache()
    Path('exact_validation_v6_complete.json').write_text(json.dumps({
        'results': results, 'seconds': time.time()-started,
        'scope': 'Complete development fold1; fixed fold0 policies. No test inference or submission.'}, indent=2))


if __name__ == '__main__':
    main()
