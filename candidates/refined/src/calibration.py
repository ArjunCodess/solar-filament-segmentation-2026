"""Calibrate component extraction on saved OOF maps without repeated disk-sized scans."""
import itertools
import numpy as np
from scipy import ndimage as ndi


def component_trials(probability, valid, truth_masks, thresholds=(.4, .5, .6, .7, .8),
                     areas=(16, 64, 256), seed_gaps=(0., .15)):
    """Return exact public-PQ sufficient statistics for each hysteresis setting.

    Count intersections before filtering components. Filtering removes columns from
    that same IoU matrix, so neither dense instance stacks nor watershed reruns are
    needed. Independent overlapping annotation masks remain separate rows.
    """
    truth_indices = [np.flatnonzero(mask) for mask in truth_masks]
    truth_union = np.logical_or.reduce(truth_masks) if truth_masks else np.zeros_like(valid)
    rows = []
    for threshold, gap in itertools.product(thresholds, seed_gaps):
        seed_threshold = min(1., round(threshold+gap, 4))
        labels, count = ndi.label((probability >= threshold) & valid, np.ones((3, 3)))
        flat = labels.ravel()
        sizes = np.bincount(flat, minlength=count+1)[1:]
        seeds = np.bincount(labels[probability >= seed_threshold], minlength=count+1)[1:]
        intersections = np.asarray([np.bincount(flat[index], minlength=count+1)[1:]
                                    for index in truth_indices]).reshape(len(truth_indices), count)
        union = np.asarray([len(index) for index in truth_indices])[:, None]+sizes-intersections
        matrix = np.divide(intersections, union, out=np.zeros_like(union, dtype=float), where=union > 0)
        union_intersection = np.bincount(labels[truth_union], minlength=count+1)[1:]
        for min_area in areas:
            keep = (sizes >= min_area) & (seeds >= 1)
            selected = matrix[:, keep]
            hit = selected > .5
            intersection = int(union_intersection[keep].sum())
            foreground = int(sizes[keep].sum())
            truth_area = int(truth_union.sum())
            rows.append({'parameters': {'mode': 'components', 'threshold': threshold,
                                        'seed_threshold': seed_threshold, 'min_area': min_area},
                         'pq_iou_sum': float(selected[hit].sum()), 'pq_tp': int(hit.sum()),
                         'pq_fp': int((~hit.any(axis=0)).sum()),
                         'pq_fn': int((~hit.any(axis=1)).sum()),
                         'dice': 2*intersection/(foreground+truth_area) if foreground+truth_area else 1.,
                         'predicted_instances': int(keep.sum())})
    return rows
