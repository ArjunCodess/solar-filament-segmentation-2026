"""Untrained multiscale dark-ridge baseline for real-image inference.

No labels or neural weights are used. Defaults are unvalidated on MAGFiLO.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from .preprocess import normalize_disk
from .submission import encode, validate_submission


def segment(image, min_area=40, min_length=25, min_eccentricity=.85):
    normalized, valid, disk = normalize_disk(image, clahe=False)
    y, x = np.indices(image.shape)
    # The untrained baseline excludes the limb ring where residuals are strongest.
    interior = valid & (((x-disk['xc'])**2+(y-disk['yc'])**2) < (.95*disk['radius'])**2)
    response = np.zeros(image.shape, np.float32)
    for width in (9, 17, 33, 65):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
        response = np.maximum(response, cv2.morphologyEx(normalized, cv2.MORPH_BLACKHAT, kernel))
    response[~interior] = 0
    low, high = np.quantile(response[interior], [.98, .995])
    high, low = max(float(high), .06), max(float(low), .03)
    seeds = (response >= high)&interior
    support = (response >= low)&interior
    grown = ndi.binary_propagation(seeds, structure=np.ones((3, 3)), mask=support)
    components, _ = ndi.label(grown, np.ones((3, 3)))
    labels = np.zeros(image.shape, np.uint16)
    instances = []
    for region in regionprops(components):
        if region.area < min_area or region.axis_major_length < min_length or region.eccentricity < min_eccentricity:
            continue
        if len(instances) >= 65535:
            raise ValueError('Too many instances for uint16 output')
        index = len(instances)+1
        coords = region.coords
        labels[coords[:, 0], coords[:, 1]] = index
        instances.append({'instance':index, 'area':int(region.area), 'eccentricity':float(region.eccentricity),
                          'major_axis_length':float(region.axis_major_length),
                          'mean_ridge_response':float(response[coords[:, 0], coords[:, 1]].mean())})
    return labels, normalized, response, {'disk':disk, 'low_threshold':low, 'high_threshold':high, 'instances':instances}


def run(images, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg'))
    if not paths:
        raise ValueError('No input JPEG observations')
    rows, shapes, details = [], {}, []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Unreadable observation: {path}')
        labels, normalized, response, report = segment(image)
        shapes[path.stem] = image.shape
        report['image_id'] = path.stem
        details.append(report)
        cv2.imwrite(str(output/f'{path.stem}_instances.png'), labels)
        cv2.imwrite(str(output/f'{path.stem}_ridge_response.png'), np.uint16(np.clip(response, 0, 1)*65535))
        for item in report['instances']:
            i = item['instance']
            rows.append({'filament_id':f'{path.stem}_{i}', 'segmentation_rle':encode(labels == i)})
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        for ax, displayed, title in zip(axes, [image/255, normalized, image/255],
                                        ['Raw GONG H-alpha', 'Radial normalization', 'Classical candidates, unvalidated']):
            ax.imshow(displayed, cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        overlay = np.zeros((*labels.shape, 4), np.float32)
        overlay[labels > 0] = [1, .25, .1, .65]
        axes[2].imshow(overlay)
        fig.suptitle('Unlabeled real-image demonstration. No trained-model or segmentation accuracy claim.', fontsize=10)
        fig.savefig(output/f'{path.stem}_comparison.png', dpi=160)
        plt.close(fig)
    csv_path = output/'classical_demo_predictions.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['filament_id', 'segmentation_rle'])
        writer.writeheader()
        writer.writerows(rows)
    validation = validate_submission(csv_path, shapes)
    result = {'scope':'Untrained classical baseline on unlabeled public observations; not a competition submission',
              'format_validation':validation, 'images':details}
    (output/'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'observations':len(paths), 'candidate_instances':len(rows), 'format_errors':validation['format_errors'],
                      'output':str(output.resolve()), 'scope':result['scope']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--images', required=True)
    p.add_argument('--output', default='runs/classical_real_demo')
    a = p.parse_args()
    run(a.images, a.output)
