"""Exercise import -> folds -> train -> OOF -> tune -> encode on synthetic data.

All artifacts go under runs/synthetic_smoke, never the competition data folder.
"""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.dataset import prepare
from src.train import parser, run
from src.submission import predict
from src.tune import tune
from argparse import Namespace


def main():
    folder = ROOT/'runs/synthetic_smoke'
    images = folder/'images'
    images.mkdir(parents=True, exist_ok=True)
    records, annotations = [], []
    rng = np.random.default_rng(42)
    y, x = np.indices((128, 128))
    radius = np.sqrt((x-64)**2+(y-64)**2)
    for i in range(10):
        name = f'201{1+i%9}{1+i//9:02d}01000000Bh.jpeg'
        image = np.where(radius < 54, 110+95*np.sqrt(np.maximum(1-(radius/54)**2, 0)), 0)
        image += rng.normal(0, 2, image.shape)
        top = 44+i%5
        image[top:top+4, 35:90] *= .45
        image[top-6:top+4, 55:57] *= .5
        cv2.imwrite(str(images/name), np.uint8(np.clip(image, 0, 255)))
        records.append({'id':str(i), 'file_name':name, 'height':128, 'width':128})
        annotations.append({'id':str(i), 'image_id':str(i), 'segmentation':[[35, top, 90, top, 90, top+4, 57, top+4,
                            57, top-6, 55, top-6, 55, top+4, 35, top+4]], 'bbox':[35, top-6, 55, 10], 'area':230})
    source = folder/'annotations.json'
    source.write_text(json.dumps({'images':records, 'annotations':annotations}))
    prepare(source, images, folder/'cache')
    args = parser().parse_args(['--manifest', str(folder/'cache/manifest.json'), '--architecture', 'unet',
        '--output', str(folder/'model'), '--epochs', '1', '--samples', '8', '--crop', '64', '--batch-size', '2',
        '--device', 'cpu', '--threads', '2', '--no-pretrained', '--tta'])
    run(args)
    result = tune(list((folder/'model/oof').glob('*.npz')))
    (folder/'postprocess.json').write_text(json.dumps(result, indent=2))
    predict(Namespace(checkpoints=[str(folder/'model/best.pt')], weights=None, images=str(images), output=str(folder/'submission.csv'),
                      device='cpu', tile=64, tta=True, tta_batch_size=1,
                      params=str(folder/'postprocess.json'), allow_nonstandard_size=True))
    (folder/'NOTICE.txt').write_text('SYNTHETIC ENGINEERING SMOKE TEST ONLY. No MAGFiLO data, no competition performance claim.\n')


if __name__ == '__main__':
    main()
