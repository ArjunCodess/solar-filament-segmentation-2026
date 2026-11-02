"""Native-resolution, overlap-blended D4 inference and spine/boundary instances."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import torch
import torch.nn.functional as F


def d4(x, k, flip):
    x = torch.rot90(x, k, (-2, -1))
    return x.flip(-1) if flip else x


def undo_d4(x, k, flip):
    return torch.rot90(x.flip(-1) if flip else x, -k, (-2, -1))


@torch.inference_mode()
def tiled_predict(model, image, device='cpu', tile=512, overlap=128, tta=True, tta_batch_size=1):
    if tile % 32 or not 0 <= overlap < tile:
        raise ValueError('Tile must be a multiple of 32; overlap must be in [0,tile)')
    if tta_batch_size < 1:
        raise ValueError('TTA batch size must be positive')
    model.eval()
    x = torch.as_tensor(image, dtype=torch.float32)
    if x.ndim != 3 or x.shape[0] != 2:
        raise ValueError('Input must be [2,H,W] normalized and raw grayscale')
    _, original_h, original_w = x.shape
    context = F.interpolate(x[:1][None], (128, 128), mode='bilinear', align_corners=False).to(device)
    x = F.pad(x, (0, max(0, tile-original_w), 0, max(0, tile-original_h)))
    h, w = x.shape[-2:]
    def starts(length):
        return sorted(set(list(range(0, max(length-tile+1, 1), tile-overlap))+[length-tile]))
    result = np.zeros((3, h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    window = np.maximum(np.outer(np.hanning(tile), np.hanning(tile)), .01).astype(np.float32)
    transforms = [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]
    for top in starts(h):
        for left in starts(w):
            patch = x[:, top:top+tile, left:left+tile][None].to(device)
            predictions = []
            for start in range(0, len(transforms), tta_batch_size):
                batch = transforms[start:start+tta_batch_size]
                patches = torch.cat([d4(patch, k, flip) for k, flip in batch])
                contexts = torch.cat([d4(context, k, flip) for k, flip in batch])
                with torch.autocast(device_type=str(device).split(':')[0], enabled=str(device).startswith('cuda')):
                    logits = model(patches, contexts).float().sigmoid()
                predictions.extend(undo_d4(logits[i:i+1], k, flip) for i, (k, flip) in enumerate(batch))
            predicted = torch.stack(predictions).mean(0)[0].cpu().numpy()
            result[:, top:top+tile, left:left+tile] += predicted*window
            weight[top:top+tile, left:left+tile] += window
    return (result/weight[None])[:, :original_h, :original_w]


def postprocess(probabilities, valid=None, threshold=.5, spine_threshold=.4,
                boundary_threshold=.6, min_area=8, bridge=0, mode='watershed',
                seed_threshold=None, seed_min_area=1, merge_radius=0, mean_threshold=0.):
    if mode not in {'watershed', 'components'}:
        raise ValueError('Unknown instance extraction mode')
    if seed_threshold is not None and not threshold <= seed_threshold <= 1:
        raise ValueError('Seed threshold must be between foreground threshold and one')
    if merge_radius < 0 or int(merge_radius) != merge_radius or not 0 <= mean_threshold <= 1:
        raise ValueError('Merge radius must be a nonnegative integer; mean threshold must be in [0,1]')
    p, spine, boundary = probabilities
    foreground = p >= threshold
    if valid is not None:
        foreground &= valid
    if bridge:
        # Add only low-confidence pixels supported by the predicted spine.
        closed = ndi.binary_closing(foreground, iterations=bridge)
        foreground |= closed & (spine >= spine_threshold) & (p >= threshold*.7)
        if valid is not None:
            foreground &= valid
    grouping = foreground
    if merge_radius:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*merge_radius+1, 2*merge_radius+1))
        grouping = cv2.morphologyEx(foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
        # Closing groups nearby fragments but must never invent foreground pixels.
        grouping |= foreground
        if valid is not None:
            grouping &= valid
    components, count = ndi.label(grouping, np.ones((3, 3)))
    components[~foreground] = 0
    sizes = np.bincount(components.ravel())
    sizes = np.pad(sizes, (0, max(0, count+1-len(sizes))))
    retained = sizes >= min_area
    retained[0] = False
    if mean_threshold:
        sums = np.bincount(components.ravel(), weights=p.ravel(), minlength=count+1)
        retained &= np.divide(sums, sizes, out=np.zeros_like(sums), where=sizes > 0) >= mean_threshold
    if seed_threshold is not None:
        seed_counts = np.bincount(components[(p >= seed_threshold) & foreground], minlength=count+1)
        retained &= seed_counts >= seed_min_area
    foreground &= retained[components]
    if mode == 'components':
        # Keep faint connected branches only when their component has a confident seed.
        mapping = np.zeros(count+1, np.int32)
        mapping[retained] = np.arange(1, int(retained.sum())+1)
        return mapping[components]
    cores = foreground & (spine >= spine_threshold) & (boundary < boundary_threshold)
    markers, _ = ndi.label(cores, np.ones((3, 3)))
    seeded = np.zeros(count+1, bool)
    seeded[np.unique(components[markers > 0])] = True
    missing = np.flatnonzero(retained & ~seeded)
    fallback = np.zeros(count+1, np.int32)
    fallback[missing] = np.arange(len(missing))+int(markers.max())+1
    markers = np.where(markers > 0, markers, fallback[components])
    if not foreground.any():
        return np.zeros(p.shape, np.int32)
    distances = ndi.distance_transform_edt(foreground)
    labels = watershed(boundary-distances/max(float(distances.max()), 1),
                       markers, mask=foreground)
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    mapping = np.zeros(len(areas), np.int32)
    mapping[keep] = np.arange(1, len(keep)+1)
    return mapping[labels]


def masks_from_labels(labels):
    return [labels == i for i in np.unique(labels) if i]
