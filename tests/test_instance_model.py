import numpy as np
import torch

from src.instance_model import (augmented_detection, build_detector, build_refiner,
    decode_proposals, instance_crop, instance_loss, masks_to_boxes)


def test_instance_identity_survives_disconnected_pixels_and_overlap():
    probability = np.zeros((32, 32), np.float32)
    probability[3:7, 3:8] = .9
    probability[20:25, 20:26] = .9
    other = np.zeros_like(probability)
    other[4:8, 4:9] = .8
    proposals = [{'window': (0, 0, 32, 32), 'score': .9, 'probability': probability},
                 {'window': (0, 0, 32, 32), 'score': .5, 'probability': other}]
    labels, confidence = decode_proposals(proposals, np.ones((32, 32), bool), min_area=1)
    assert labels[4, 4] == labels[22, 22] == 1
    assert labels[7, 8] == 2
    assert confidence[4, 4] > confidence[7, 8]
    rejected, _ = decode_proposals(proposals, np.ones((32, 32), bool), score_threshold=.95)
    assert not rejected.any()


def test_detection_augmentation_keeps_boxes_and_image_aligned():
    image = np.zeros((3, 32, 48), np.float32)
    image[:, 4:13, 7:26] = 1
    boxes = masks_to_boxes([image[0] > 0])
    for seed in range(30):
        transformed, target = augmented_detection(image.copy(), boxes.copy(), np.random.default_rng(seed))
        np.testing.assert_array_equal(target, masks_to_boxes([transformed[0] > 0]))
        assert transformed.flags.c_contiguous


def test_object_crop_targets_one_instance_and_loss_has_finite_gradient():
    mask = np.zeros((64, 64), bool)
    mask[20:24, 15:45] = True
    image = np.stack([mask.astype(np.float32)]*2)
    x, y = instance_crop(image, mask, np.random.default_rng(9), size=64)
    assert x.shape == (2, 64, 64) and y.shape == (1, 64, 64) and y.sum() > 0
    logits = torch.zeros(1, 1, 64, 64, requires_grad=True)
    loss = instance_loss(logits, torch.from_numpy(y)[None])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_actual_detector_and_refiner_forward_without_weight_downloads():
    torch.set_num_threads(2)
    detector = build_detector(False, image_size=64).eval()
    refiner = build_refiner(False).eval()
    with torch.inference_mode():
        predictions = detector([torch.zeros(3, 64, 64)])[0]
        mask = refiner(torch.zeros(1, 2, 64, 64))
    assert predictions['boxes'].shape[1] == 4
    assert torch.isfinite(predictions['scores']).all()
    assert mask.shape == (1, 1, 64, 64) and torch.isfinite(mask).all()
    detector.train()
    losses = detector([torch.zeros(3, 64, 64)], [{'boxes': torch.tensor([[4., 5., 25., 12.]]),
                                                'labels': torch.ones(1, dtype=torch.int64)}])
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert detector.roi_heads.box_predictor.cls_score.weight.grad is not None
