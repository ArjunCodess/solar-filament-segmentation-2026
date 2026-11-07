import numpy as np
import torch

from src.instance_model import (augmented_detection, box_window, build_detector, build_refiner,
    decode_proposals, instance_crop, instance_loss, instance_proposals, masks_to_boxes, refine_input)


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


def test_conditioned_crop_and_refine_input_distinguish_neighboring_instances():
    mask_a = np.zeros((64, 64), bool)
    mask_b = np.zeros((64, 64), bool)
    mask_a[26:32, 20:25] = True
    mask_b[26:32, 28:33] = True
    image = np.stack([(mask_a | mask_b).astype(np.float32)]*2)
    box_a, box_b = masks_to_boxes([mask_a, mask_b])
    window = box_window(box_a, mask_a.shape, margin=.25)
    unconditioned = refine_input(image, window, size=64)
    assert unconditioned.shape == (2, 64, 64)
    cond_a = refine_input(image, window, size=64, box=box_a)
    cond_b = refine_input(image, window, size=64, box=box_b)
    seed_a = refine_input(image, window, size=64, seed=mask_a)
    seed_b = refine_input(image, window, size=64, seed=mask_b)
    assert cond_a.shape == cond_b.shape == seed_a.shape == seed_b.shape == (3, 64, 64)
    np.testing.assert_array_equal(cond_a[:2], unconditioned)
    np.testing.assert_array_equal(cond_b[:2], unconditioned)
    assert not np.allclose(cond_a[2], cond_b[2])
    assert not np.allclose(seed_a[2], seed_b[2])
    for seed in range(8):
        x_a, y_a = instance_crop(image, mask_a, np.random.default_rng(seed), size=64, conditioned=True)
        x_b, y_b = instance_crop(image, mask_b, np.random.default_rng(seed), size=64, conditioned=True)
        assert x_a.shape == x_b.shape == (3, 64, 64)
        for x, y in ((x_a, y_a), (x_b, y_b)):
            target_pixels = y[0] > .5
            neighbor_pixels = (x[0] > 0) & (~target_pixels)
            assert target_pixels.any() and neighbor_pixels.any()
            assert float(x[2, target_pixels].mean()) > float(x[2, neighbor_pixels].mean())+.4


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


def test_conditioned_refiner_forward_backward_and_proposals():
    torch.set_num_threads(2)
    legacy_refiner = build_refiner(False).eval()
    conditioned_refiner = build_refiner(False, conditioned=True).train()
    assert legacy_refiner.encoder.conv1.in_channels == 2
    assert conditioned_refiner.encoder.conv1.in_channels == 3
    mask_a = np.zeros((64, 64), bool)
    mask_b = np.zeros((64, 64), bool)
    mask_a[18:24, 16:22] = True
    mask_b[18:24, 25:31] = True
    image = np.stack([(mask_a | mask_b).astype(np.float32)]*2)
    x, y = instance_crop(image, mask_a, np.random.default_rng(3), size=64, conditioned=True)
    logits = conditioned_refiner(torch.from_numpy(x)[None])
    loss = instance_loss(logits, torch.from_numpy(y)[None])
    loss.backward()
    grad = conditioned_refiner.encoder.conv1.weight.grad
    assert logits.shape == (1, 1, 64, 64) and torch.isfinite(loss)
    assert grad is not None and torch.isfinite(grad).all()
    assert float(grad[:, 2].abs().sum()) > 0
    conditioned_refiner.eval()

    class FixedDetector(torch.nn.Module):
        def forward(self, inputs):
            return [{'boxes': torch.tensor([[16., 18., 22., 24.], [25., 18., 31., 24.]]),
                     'scores': torch.tensor([.9, .8])}]

    detector = FixedDetector()
    legacy_props = instance_proposals(detector, legacy_refiner, image, 'cpu', size=64)
    cond_props = instance_proposals(detector, conditioned_refiner, image, 'cpu', size=64)
    assert len(legacy_props) == len(cond_props) == 2
    for prop in cond_props:
        w = prop['window']
        assert prop['probability'].shape == (w[3]-w[1], w[2]-w[0])
        assert np.isfinite(prop['probability']).all()
