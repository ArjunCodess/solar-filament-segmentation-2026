"""Box proposals plus object-specific segmentation, using approved masks as labels."""
from collections import defaultdict
from functools import lru_cache

import cv2
import numpy as np
import torch
from torch import nn

from .dataset import load_image_cache


def build_detector(pretrained=True, image_size=1024):
    from torchvision.models import ResNet50_Weights
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    # ImageNet backbone only. No COCO detection weights or external solar labels.
    anchors = AnchorGenerator(tuple((size,) for size in (8, 16, 32, 64, 128)),
                              ((.125, .25, .5, 1., 2., 4., 8.),)*5)
    return fasterrcnn_resnet50_fpn(weights=None,
        weights_backbone=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None,
        num_classes=2, min_size=image_size, max_size=image_size,
        rpn_anchor_generator=anchors, box_detections_per_img=40,
        box_score_thresh=.05, box_nms_thresh=.4,
        rpn_pre_nms_top_n_train=2000, rpn_post_nms_top_n_train=1000,
        rpn_pre_nms_top_n_test=1000, rpn_post_nms_top_n_test=500)


def build_refiner(pretrained=True, conditioned=False, in_channels=None):
    import segmentation_models_pytorch as smp
    if in_channels is None:
        in_channels = 3 if conditioned else 2
    return smp.Unet(encoder_name='resnet18', encoder_weights='imagenet' if pretrained else None,
                    in_channels=in_channels, classes=1)


def masks_to_boxes(masks):
    boxes = []
    for mask in masks:
        yy, xx = np.nonzero(mask)
        if len(xx):
            boxes.append([xx.min(), yy.min(), xx.max()+1, yy.max()+1])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def load_truth(record, shape):
    with np.load(record['cache']) as data:
        return [np.unpackbits(mask, count=int(np.prod(shape))).reshape(shape).astype(bool)
                for mask in data['instance_masks_packed']]


def box_window(box, shape, margin=.15):
    x0, y0, x1, y1 = box
    width, height = max(16., x1-x0), max(16., y1-y0)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    return (max(0, int(np.floor(cx-width*(.5+margin)))),
            max(0, int(np.floor(cy-height*(.5+margin)))),
            min(shape[1], int(np.ceil(cx+width*(.5+margin)))),
            min(shape[0], int(np.ceil(cy+height*(.5+margin)))))


def refine_input(image, window, size=512, box=None, seed=None, conditioned=None):
    x0, y0, x1, y1 = window
    patch = cv2.resize(image[:, y0:y1, x0:x1].transpose(1, 2, 0),
                       (size, size), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
    out = (patch-.5)/.25
    use_condition = (box is not None or seed is not None) if conditioned is None else bool(conditioned)
    if not use_condition:
        return np.ascontiguousarray(out, dtype=np.float32)
    if box is None and seed is None:
        cx, cy = .5*(x0+x1), .5*(y0+y1)
        bw, bh = (x1-x0)/1.3, (y1-y0)/1.3
        box = (cx-.5*bw, cy-.5*bh, cx+.5*bw, cy+.5*bh)
    cues = []
    if box is not None:
        bx0, by0, bx1, by1 = [float(v) for v in box]
        u = x0+(np.arange(size, dtype=np.float32)+.5)*max(1, x1-x0)/size
        v = y0+(np.arange(size, dtype=np.float32)+.5)*max(1, y1-y0)/size
        in_box = ((u[None, :] >= bx0) & (u[None, :] <= bx1)
                  & (v[:, None] >= by0) & (v[:, None] <= by1)).astype(np.float32)
        cx, cy = .5*(bx0+bx1), .5*(by0+by1)
        rx, ry = max(.5, .5*(bx1-bx0)), max(.5, .5*(by1-by0))
        gauss = np.exp(-.5*(((u[None, :]-cx)/rx)**2+((v[:, None]-cy)/ry)**2)).astype(np.float32)
        cues.append(.5*in_box+.5*gauss)
    if seed is not None:
        seed_arr = np.asarray(seed, dtype=np.float32)
        if seed_arr.shape == image.shape[1:]:
            seed_arr = seed_arr[y0:y1, x0:x1]
        if seed_arr.shape != (size, size):
            seed_arr = cv2.resize(seed_arr, (size, size), interpolation=cv2.INTER_LINEAR)
        cues.append(np.clip(seed_arr, 0., 1.))
    cue = np.mean(cues, axis=0, dtype=np.float32)[None]
    return np.ascontiguousarray(np.concatenate([out, cue], axis=0), dtype=np.float32)


class InstanceTrainingData:
    """Records must already be split by temporal fold before construction."""
    def __init__(self, records):
        self.records = records
        grouped = defaultdict(list)
        for i, record in enumerate(records):
            grouped[record['sha256']].append(i)
        self.physical = list(grouped.values())

    @lru_cache(maxsize=4)
    def image(self, index):
        # Cache by a canonical view per physical observation.
        return load_image_cache(self.records[index])

    def sample(self, rng):
        group = self.physical[int(rng.integers(len(self.physical)))]
        index = int(rng.choice(group))
        image, valid = self.image(group[0])
        masks = load_truth(self.records[index], valid.shape)
        return image, valid, masks


def augmented_detection(image, boxes, rng):
    """Transform image and exclusive-coordinate boxes together."""
    h, w = image.shape[1:]
    if rng.random() < .5:
        image = image[:, :, ::-1]
        boxes = boxes[:, [2, 1, 0, 3]]*np.array([-1, 1, -1, 1], np.float32)
        boxes[:, [0, 2]] += w
    if rng.random() < .5:
        image = image[:, ::-1, :]
        boxes = boxes[:, [0, 3, 2, 1]]*np.array([1, -1, 1, -1], np.float32)
        boxes[:, [1, 3]] += h
    if rng.random() < .5:
        image = image.transpose(0, 2, 1)
        boxes = boxes[:, [1, 0, 3, 2]]
    return np.ascontiguousarray(image), np.ascontiguousarray(boxes)


def instance_crop(image, mask, rng, size=512, conditioned=False, seed=None):
    box = masks_to_boxes([mask])[0]
    width, height = box[2]-box[0], box[3]-box[1]
    box += np.array([rng.uniform(-.1, .1)*width, rng.uniform(-.1, .1)*height,
                     rng.uniform(-.1, .1)*width, rng.uniform(-.1, .1)*height], np.float32)
    window = box_window(box, mask.shape, margin=float(rng.uniform(.1, .3)))
    x0, y0, x1, y1 = window
    use_cond = bool(conditioned) or seed is not None
    x = refine_input(image, window, size, box=box if use_cond else None,
                     seed=seed if use_cond else None, conditioned=use_cond)
    y = cv2.resize(mask[y0:y1, x0:x1].astype(np.uint8), (size, size),
                   interpolation=cv2.INTER_NEAREST).astype(np.float32)[None]
    k = int(rng.integers(4))
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
    if rng.random() < .5:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def instance_loss(logits, target):
    probability = logits.sigmoid()
    dice = 1-(2*(probability*target).sum((1, 2, 3))+1)/(
        probability.sum((1, 2, 3))+target.sum((1, 2, 3))+1)
    return .5*dice.mean()+.5*nn.functional.binary_cross_entropy_with_logits(logits, target)


@torch.inference_mode()
def instance_proposals(detector, refiner, image, device, size=512, conditioned=None):
    if conditioned is None:
        params = list(refiner.parameters())
        conditioned = bool(params and params[0].ndim == 4 and params[0].shape[1] > 2)
    composite = np.stack([image[1], image[0], image[0]])
    output = detector([torch.from_numpy(composite).to(device)])[0]
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    proposals = []
    for box, score in zip(boxes, scores):
        window = box_window(box, image.shape[1:])
        x0, y0, x1, y1 = window
        patch = torch.from_numpy(refine_input(
            image, window, size, box=box if conditioned else None,
            conditioned=bool(conditioned)))[None].to(device)
        probability = refiner(patch).sigmoid()[0, 0].cpu().numpy()
        probability = cv2.resize(probability, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
        proposals.append({'window': window, 'score': float(score), 'probability': probability})
    return proposals


def decode_proposals(proposals, valid, score_threshold=.3, mask_threshold=.5, min_area=16):
    """One box remains one object, even when its mask has disconnected pieces."""
    labels = np.zeros(valid.shape, np.int32)
    confidence = np.zeros(valid.shape, np.float32)
    identifier = 0
    for proposal in proposals:
        if proposal['score'] < score_threshold:
            continue
        x0, y0, x1, y1 = proposal['window']
        prob = proposal['probability']
        mask = (prob >= mask_threshold) & valid[y0:y1, x0:x1]
        if mask.sum() < min_area:
            continue
        identifier += 1
        strength = prob*proposal['score']
        take = mask & (strength > confidence[y0:y1, x0:x1])
        labels[y0:y1, x0:x1][take] = identifier
        confidence[y0:y1, x0:x1][take] = strength[take]
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep > 0]
    mapping = np.zeros(len(areas), np.int32)
    mapping[keep] = np.arange(1, len(keep)+1)
    return mapping[labels], confidence
