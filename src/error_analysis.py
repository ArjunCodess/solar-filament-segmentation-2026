"""Per-instance OOF diagnostics; every annotator remains a separate target view."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def instance_error_rows(truth_masks, labels, probability, raw, valid, image_id):
    areas = np.bincount(labels.ravel())[1:]
    intersections = np.array([np.bincount(labels[mask], minlength=len(areas)+1)[1:]
                              for mask in truth_masks], dtype=float).reshape(len(truth_masks), len(areas))
    truth_areas = np.array([mask.sum() for mask in truth_masks])
    unions = truth_areas[:, None]+areas-intersections
    ious = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    disk_y, disk_x = ndi.center_of_mass(valid)
    radius = np.sqrt(valid.sum()/np.pi)
    rows = []
    for index, mask in enumerate(truth_masks):
        yy, xx = np.nonzero(mask)
        best = float(ious[index].max()) if len(areas) else 0.
        skeleton = skeletonize(mask)
        ring = ndi.binary_dilation(mask, iterations=5) & ~mask & valid
        contrast = float(raw[ring].mean()-raw[mask].mean()) if ring.any() else 0.
        rows.append({'image_id': str(image_id), 'kind': 'truth', 'instance': index+1,
                     'area': int(mask.sum()), 'best_iou': best, 'matched': best > .5,
                     'size_bin': 'small' if mask.sum() < 256 else 'medium' if mask.sum() < 2048 else 'large',
                     'radial_fraction': float(np.hypot(yy.mean()-disk_y, xx.mean()-disk_x)/max(radius, 1)),
                     'contrast': contrast, 'mean_probability': float(probability[mask].mean()),
                     'skeleton_recall': float((labels[skeleton] > 0).mean()) if skeleton.any() else 1.,
                     'fragments': int((intersections[index] >= .1*truth_areas[index]).sum())})
    for index, area in enumerate(areas):
        mask = labels == index+1
        best = float(ious[:, index].max()) if len(truth_masks) else 0.
        rows.append({'image_id': str(image_id), 'kind': 'prediction', 'instance': index+1,
                     'area': int(area), 'best_iou': best, 'matched': best > .5,
                     'mean_probability': float(probability[mask].mean()),
                     'merged_truths': int((intersections[:, index] >= .1*np.maximum(truth_areas, 1)).sum())})
    return rows
