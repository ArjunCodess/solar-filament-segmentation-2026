"""Generate figures and ablation tables only from actual saved OOF predictions."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from .metrics import aggregate_reference_pq


def qualitative(baseline_path, proposed_path, output, proposed_title='Spine-guided probability'):
    with np.load(baseline_path) as z:
        baseline, baseline_id = z['probabilities'][0], str(z['image_id'])
    with np.load(proposed_path) as z:
        proposed, raw, truth, proposed_id = z['probabilities'][0], z['raw'], z['truth'] > 0, str(z['image_id'])
    if baseline.shape != proposed.shape or baseline_id != proposed_id:
        raise ValueError('Comparison must use the same held-out image and annotator view')
    ys, xs = np.where(truth)
    if not len(ys):
        raise ValueError('A positive GT region is required for the barb detail figure')
    # Center the detail around a GT pixel; no undocumented selection of a favorable prediction.
    cy, cx = ys[len(ys)//2], xs[len(xs)//2]
    half = min(96, min(truth.shape)//2)
    y0, x0 = max(0, int(cy)-half), max(0, int(cx)-half)
    region = np.s_[y0:min(truth.shape[0], y0+2*half), x0:min(truth.shape[1], x0+2*half)]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
    for col, (image, title) in enumerate([(raw, 'Raw H-alpha'), (truth, 'Ground truth'),
                                         (baseline, 'U-Net probability'), (proposed, proposed_title)]):
        for row in range(2):
            display = image if row == 0 else image[region]
            if row == 0:
                display = cv2.resize(display.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
            display = np.uint8(np.clip(display, 0, 1)*255)
            axes[row, col].imshow(display, cmap='gray', vmin=0, vmax=255,
                                   interpolation='nearest')
            axes[row, col].axis('off')
        axes[0, col].set_title(title, fontsize=11)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def ablation(run_folders, output):
    rows = []
    for folder in run_folders:
        folder = Path(folder)
        for path in sorted(folder.glob('**/oof_metrics.csv')):
            config = json.loads(path.with_name('config.json').read_text())
            frame = pd.read_csv(path)
            if 'synthetic_smoke' in str(path):
                raise ValueError('Synthetic smoke metrics cannot enter the competition ablation table')
            rows.append({'architecture':config['architecture'], 'fold':config['fold'],
                         'topology':not config['no_topology'], 'auxiliary':not config['no_auxiliary'],
                         'limb':config['preprocess']['limb'], 'clahe':config['preprocess']['clahe'],
                         **frame.select_dtypes('number').drop(columns=['image_id'], errors='ignore').mean().to_dict(),
                         'pq':aggregate_reference_pq(frame.to_dict('records'))})
    if not rows:
        raise ValueError('No real OOF results. A measured ablation table cannot be generated yet.')
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    Path(output).with_suffix('.tex').write_text(result.to_latex(index=False, float_format='%.4f'))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('qualitative')
    q.add_argument('--baseline', required=True)
    q.add_argument('--proposed', required=True)
    q.add_argument('--output', default='paper/qualitative.png')
    a = sub.add_parser('ablation')
    a.add_argument('--runs', nargs='+', required=True)
    a.add_argument('--output', default='paper/ablation.csv')
    args = p.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.command == 'qualitative':
        qualitative(args.baseline, args.proposed, args.output)
    else:
        ablation(args.runs, args.output)
