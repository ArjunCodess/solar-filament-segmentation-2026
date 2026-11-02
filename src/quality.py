"""Image-only instance-quality features and portable regression-tree inference."""
import numpy as np
from scipy import ndimage as ndi

FEATURES = ('log_area', 'mean_probability', 'std_probability', 'q10_probability',
            'q90_probability', 'extent', 'aspect_ratio', 'local_contrast',
            'mean_boundary', 'mean_spine')


def quality_features(labels, maps, raw):
    features = []
    for index, slices in enumerate(ndi.find_objects(labels), 1):
        if slices is None:
            raise ValueError('Instance labels must be contiguous')
        ys, xs = slices
        region = (slice(max(0, ys.start-4), min(labels.shape[0], ys.stop+4)),
                  slice(max(0, xs.start-4), min(labels.shape[1], xs.stop+4)))
        mask = labels[region] == index
        p = maps[0][region][mask]
        ring = ndi.binary_dilation(mask, iterations=3) & (labels[region] == 0)
        contrast = float(raw[region][ring].mean()-raw[region][mask].mean()) if ring.any() else 0.
        height, width = ys.stop-ys.start, xs.stop-xs.start
        features.append([np.log1p(mask.sum()), p.mean(), p.std(), *np.quantile(p, [.1, .9]),
                         mask.sum()/(height*width), max(height, width)/max(1, min(height, width)),
                         contrast, maps[2][region][mask].mean(), maps[1][region][mask].mean()])
    return np.asarray(features, np.float32).reshape(-1, len(FEATURES))


def predict_quality(features, model):
    if model['features'] != list(FEATURES):
        raise ValueError('Quality feature schema mismatch')
    predictions = np.zeros(len(features), np.float64)
    for tree in model['trees']:
        for row, values in enumerate(features):
            node = 0
            while tree['left'][node] != -1:
                node = tree['left'][node] if values[tree['feature'][node]] <= tree['threshold'][node] else tree['right'][node]
            predictions[row] += tree['value'][node]
    return predictions/len(model['trees'])


def filter_quality(labels, maps, raw, model):
    predictions = predict_quality(quality_features(labels, maps, raw), model)
    keep = predictions >= model['minimum_quality']
    mapping = np.zeros(len(keep)+1, np.int32)
    mapping[1:][keep] = np.arange(1, int(keep.sum())+1)
    return mapping[labels]
