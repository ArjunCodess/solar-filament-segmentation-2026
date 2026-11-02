"""Disk geometry, robust radial correction, and native-resolution targets."""
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, distance_transform_edt
from skimage.morphology import skeletonize


@dataclass
class Disk:
    xc: float
    yc: float
    radius: float
    residual: float


def detect_disk(image):
    image = np.asarray(image, np.float32)
    if image.ndim != 2 or not np.isfinite(image).all() or image.max() <= image.min():
        raise ValueError('Expected a finite, nonconstant grayscale image')
    scaled = np.uint8(np.clip((image-image.min())/(image.max()-image.min())*255, 0, 255))
    _, binary = cv2.threshold(cv2.GaussianBlur(scaled, (9, 9), 0), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    # Reject image frame points, then iteratively fit the limb circle.
    h, w = image.shape
    pts = contour[(contour[:, 0] > 1)&(contour[:, 0] < w-2)&(contour[:, 1] > 1)&(contour[:, 1] < h-2)]
    for _ in range(4):
        if len(pts) < 20:
            raise ValueError('Insufficient solar limb points')
        a = np.c_[2*pts, np.ones(len(pts))]
        xc, yc, c = np.linalg.lstsq(a, (pts**2).sum(1), rcond=None)[0]
        radius = np.sqrt(max(c+xc*xc+yc*yc, 0))
        residual = np.abs(np.linalg.norm(pts-[xc, yc], axis=1)-radius)
        keep = residual <= max(2, np.median(residual)*3)
        pts = pts[keep]
    if not min(h, w)*.2 < radius < max(h, w)*.7:
        raise ValueError(f'Implausible disk radius: {radius}')
    return Disk(float(xc), float(yc), float(radius), float(np.median(residual)))


def normalize_disk(image, clahe=True, limb=True):
    image = np.asarray(image, np.float32)
    disk = detect_disk(image)
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    radius = (np.sqrt((x-disk.xc)**2+(y-disk.yc)**2)/disk.radius).astype(np.float32)
    valid = radius <= .995
    bins = np.minimum((radius*511).astype(np.int32), 511)
    # One sort instead of scanning a four-million-pixel image 512 times.
    rb, values = bins[valid], image[valid]
    order = np.argsort(rb)
    edges = np.searchsorted(rb[order], np.arange(513))
    values = values[order]
    profile = np.array([np.median(values[edges[i]:edges[i+1]]) if edges[i+1] > edges[i] else np.nan for i in range(512)])
    good = np.flatnonzero(np.isfinite(profile))
    profile = gaussian_filter1d(np.interp(np.arange(512), good, profile[good]), 5).astype(np.float32)
    del rb, values, order, edges
    corrected = image/np.maximum(profile[bins], max(float(np.median(profile))*.1, 1e-6)) if limb else image.copy()
    lo, hi = np.percentile(corrected[valid], [1, 99.5])
    corrected = np.uint8(np.clip((corrected-lo)/max(hi-lo, 1e-6)*255, 0, 255))
    if clahe:
        corrected = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16)).apply(corrected)
    corrected[~valid] = 0
    return corrected.astype(np.float32)/255, valid, asdict(disk)


def decode_annotation(annotation, shape):
    from pycocotools import mask as coco
    seg = annotation['segmentation']
    h, w = shape
    if isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle['counts'], list):
            rle = coco.frPyObjects(rle, h, w)
        elif isinstance(rle['counts'], str):
            rle['counts'] = rle['counts'].encode('ascii')
    else:
        if not seg:
            return np.zeros(shape, bool)
        if isinstance(seg[0], (int, float)):
            seg = [seg]
        rle = coco.merge(coco.frPyObjects(seg, h, w))
    result = coco.decode(rle)
    if result.ndim == 3:
        result = result.any(2)
    if result.shape != tuple(shape):
        raise ValueError('Annotation shape differs from image')
    return result.astype(bool)


def targets(annotations, shape, use_spines=False, sigma=2.):
    mask = np.zeros(shape, bool)
    spine = np.zeros(shape, np.uint8)
    boundary = np.zeros(shape, np.uint8)
    instances = np.zeros(shape, np.int32)
    for k, ann in enumerate(annotations, 1):
        obj = decode_annotation(ann, shape)
        mask |= obj
        instances[obj] = k
        # Work within each object's padded box; full-disk skeletonization per object
        # otherwise scans millions of background pixels for every filament.
        bx, by, bw, bh = cv2.boundingRect(obj.astype(np.uint8))
        roi = np.s_[max(0, by-1):min(shape[0], by+bh+1), max(0, bx-1):min(shape[1], bx+bw+1)]
        local = obj[roi].astype(np.uint8)
        boundary[roi] |= cv2.morphologyEx(local, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        # Mask-derived topology is the safe default under the current metadata rule.
        skel = np.zeros(shape, np.uint8)
        skel[roi] = skeletonize(local > 0).astype(np.uint8)
        if use_spines and ann.get('spine'):
            pts = np.asarray(ann['spine'], np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.polylines(skel, [pts], False, 1, 1)
        spine |= skel
    # OpenCV's exact Euclidean transform returns float32 without SciPy's full
    # coordinate grids, which can exhaust system commit on 2048-pixel disks.
    heat = cv2.distanceTransform(1-spine, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) if spine.any() else np.full(shape, np.inf, np.float32)
    np.square(heat, out=heat)
    heat *= -1/(2*sigma*sigma)
    np.exp(heat, out=heat)
    return {'mask':mask.astype(np.float32), 'spine':heat.astype(np.float32),
            'skeleton':spine.astype(np.float32), 'boundary':boundary.astype(np.float32), 'instances':instances}
