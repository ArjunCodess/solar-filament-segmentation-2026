"""Explicit local conventions. Grid MIoU is a diagnostic, not the Kaggle scorer."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def binary_scores(truth, prediction):
    g, p = np.asarray(truth, bool), np.asarray(prediction, bool)
    if g.shape != p.shape:
        raise ValueError('Shape mismatch')
    tp, fp, fn = int((g&p).sum()), int((~g&p).sum()), int((g&~p).sum())
    return {'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.,
            'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
            'precision':tp/(tp+fp) if tp+fp else float(not g.any()),
            'recall':tp/(tp+fn) if tp+fn else 1.}


def grid_miou_diagnostic(truth, prediction, grids=(1, 2, 4, 8, 16)):
    """Mean IoU of nonempty cells per grid, then mean over grids, at native pixels.

    Cells empty in both masks are excluded, rather than rewarding background.
    This definition is provisional and must never be reported as official MIoU.
    """
    g, p = np.asarray(truth, bool), np.asarray(prediction, bool)
    if g.shape != p.shape:
        raise ValueError('Shape mismatch')
    results = []
    for grid in grids:
        if grid < 1 or grid > min(g.shape):
            raise ValueError('Invalid grid resolution')
        values = []
        ys, xs = np.linspace(0, g.shape[0], grid+1, dtype=int), np.linspace(0, g.shape[1], grid+1, dtype=int)
        for y0, y1 in zip(ys[:-1], ys[1:]):
            for x0, x1 in zip(xs[:-1], xs[1:]):
                a, b = g[y0:y1, x0:x1], p[y0:y1, x0:x1]
                if (a|b).any():
                    values.append(float((a&b).sum()/(a|b).sum()))
        results.append(float(np.mean(values)) if values else 1.)
    return float(np.mean(results))


def pairwise_iou(truth, prediction):
    """Mask lists support overlapping independent annotations."""
    g, p = list(truth), list(prediction)
    result = np.zeros((len(g), len(p)))
    for i, a in enumerate(g):
        for j, b in enumerate(p):
            result[i, j] = binary_scores(a, b)['iou']
    return result


def matching(matrix, threshold=.5):
    if not matrix.size:
        return []
    eligible = matrix > threshold  # PQ uses a strict threshold.
    rows, cols = linear_sum_assignment(-(eligible.astype(float)*(min(matrix.shape)+1)+matrix*eligible))
    return [(int(i), int(j)) for i, j in zip(rows, cols) if eligible[i, j]]


def instance_scores(truth, prediction, confidence=None, thresholds=(.5, .75)):
    g, p = list(truth), list(prediction)
    return instance_scores_from_iou(pairwise_iou(g, p), confidence, thresholds)


def label_prediction_scores(truth_masks, labels, probability=None):
    """IoU via component counts; no dense prediction-mask stack or pairwise disk scans."""
    truth_masks = list(truth_masks)
    count = int(labels.max())
    areas = np.bincount(labels.ravel(), minlength=count+1)[1:]
    matrix = np.zeros((len(truth_masks), count), np.float64)
    for i, truth in enumerate(truth_masks):
        overlap = np.bincount(labels[np.asarray(truth, bool)], minlength=count+1)[1:]
        union = int(np.sum(truth))+areas-overlap
        matrix[i] = np.divide(overlap, union, out=np.zeros(count), where=union > 0)
    confidence = None
    if probability is not None:
        sums = np.bincount(labels.ravel(), weights=probability.ravel(), minlength=count+1)[1:]
        confidence = np.divide(sums, areas, out=np.zeros(count), where=areas > 0)
    return instance_scores_from_iou(matrix, confidence)


def instance_scores_from_iou(matrix, confidence=None, thresholds=(.5, .75)):
    matrix = np.asarray(matrix)
    g, p = range(matrix.shape[0]), range(matrix.shape[1])
    pairs = matching(matrix)
    tp, fp, fn = len(pairs), len(p)-len(pairs), len(g)-len(pairs)
    denominator = tp+.5*fp+.5*fn
    hit = matrix > .5
    reference_tp = int(hit.sum())
    reference_fp = int((~hit.any(axis=0)).sum())
    reference_fn = int((~hit.any(axis=1)).sum())
    reference_sum = float(matrix[hit].sum())
    reference_den = reference_tp+.5*reference_fp+.5*reference_fn
    result = {'pq':reference_sum/reference_den if reference_den else 0.,
              'pq_iou_sum':reference_sum, 'pq_tp':reference_tp, 'pq_fp':reference_fp, 'pq_fn':reference_fn,
              'standard_pq':sum(matrix[i, j] for i, j in pairs)/denominator if denominator else 1.,
              'tp':tp, 'fp':fp, 'fn':fn, 'hit_rate':tp/len(g) if g else 1.,
              'miss_rate':fn/len(g) if g else 0.,
              'overlap_hit_rate':float((matrix > 0).any(1).mean()) if g else 1.}
    # AP is score-ranked precision/recall integration; it is not TP/(TP+FP+FN).
    scores = np.ones(len(p)) if confidence is None else np.asarray(confidence)
    if len(scores) != len(p):
        raise ValueError('One confidence per instance required')
    order = np.argsort(-scores, kind='stable')
    for threshold in thresholds:
        used, flags = set(), []
        for j in order:
            candidates = [(matrix[i, j], i) for i in range(len(g)) if i not in used and matrix[i, j] >= threshold]
            if candidates:
                _, i = max(candidates)
                used.add(i)
                flags.append(1)
            else:
                flags.append(0)
        if not g:
            ap = float(not p)
        elif not flags:
            ap = 0.
        else:
            hits = np.cumsum(flags)
            precision, recall = hits/np.arange(1, len(flags)+1), hits/len(g)
            ap = np.mean([max(precision[recall >= r], default=0.) for r in np.linspace(0, 1, 101)])
        result[f'ap@{threshold:.2f}'] = float(ap)
    return result


def aggregate_reference_pq(rows):
    """Micro aggregate over annotator-image entries, matching notebook version 6."""
    numerator = sum(r['pq_iou_sum'] for r in rows)
    denominator = sum(r['pq_tp']+.5*r['pq_fp']+.5*r['pq_fn'] for r in rows)
    return numerator/denominator if denominator else 0.
