import csv
import cv2
import numpy as np
import pytest
import torch

from src.dataset import CropDataset
from src.error_analysis import instance_error_rows
from src.infer import postprocess
from src.submission import encode, validate_submission
from src.quality import FEATURES, predict_quality, quality_features, filter_quality


def test_overlap_contract_rejects_distinct_ids(tmp_path):
    mask = np.zeros((16, 16), bool)
    mask[3:7, 4:10] = True
    path = tmp_path/'overlap.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerows([('image_1', encode(mask)), ('image_2', encode(mask))])
    with pytest.raises(ValueError, match='Overlapping'):
        validate_submission(path, {'image': mask.shape})


def test_fragment_grouping_keeps_pixel_support_and_far_instances():
    maps = np.zeros((3, 64, 64), np.float32)
    maps[0, 20:26, 5:20] = .9
    maps[0, 20:26, 22:38] = .9
    maps[0, 45:50, 50:58] = .9
    plain = postprocess(maps, mode='components', min_area=1)
    grouped = postprocess(maps, mode='components', min_area=1, merge_radius=2)
    assert plain.max() == 3 and grouped.max() == 2
    assert np.array_equal(grouped > 0, plain > 0)
    assert grouped[21, 10] == grouped[21, 30] != grouped[47, 53]


def test_component_mean_filter_removes_uncertain_object():
    maps = np.zeros((3, 32, 32), np.float32)
    maps[0, 3:8, 4:10] = .55
    maps[0, 20:25, 20:28] = .9
    labels = postprocess(maps, mode='components', min_area=1, mean_threshold=.7)
    assert labels.max() == 1 and labels[5, 6] == 0 and labels[22, 23] == 1


@pytest.mark.parametrize('scale', [.5, 1.5])
def test_scaled_crops_preserve_alignment_and_binary_labels(tmp_path, scale):
    image = np.zeros((96, 96), np.uint8)
    image[20:70, 30:60] = 255
    raw = tmp_path/'raw.png'
    cv2.imwrite(str(raw), image)
    image_cache, targets = tmp_path/'image.npz', tmp_path/'targets.npz'
    np.savez(image_cache, image=image, valid=np.ones_like(image, bool))
    np.savez(targets, mask=image, spine=image, boundary=image, skeleton=image)
    records = [{'image_cache': str(image_cache), 'cache': str(targets),
                'image_path': str(raw), 'sha256': 'same'}]
    dataset = CropDataset(records, crop=64, samples=1, augment=False,
                          scale_range=(scale, scale), balance_physical=True)
    x, context, y = dataset[0]
    assert x.shape == (2, 64, 64) and y.shape == (5, 64, 64)
    assert set(torch.unique(y).tolist()) <= {0., 1.}
    assert ((x[0] > .5) == (y[0] > .5)).float().mean() > .95
    assert torch.equal(x[0], x[1]) and context.shape == (1, 128, 128)


def test_error_audit_distinguishes_fragmentation_and_false_positive():
    truth = np.zeros((48, 48), bool)
    truth[20:26, 5:35] = True
    labels = np.zeros_like(truth, np.int32)
    labels[20:26, 5:15] = 1
    labels[20:26, 20:35] = 2
    labels[3:5, 40:43] = 3
    rows = instance_error_rows([truth], labels, (labels > 0).astype(float),
                               np.ones_like(truth, float), np.ones_like(truth), 'test')
    assert rows[0]['fragments'] == 2 and not rows[0]['matched']
    assert rows[-1]['best_iou'] == 0 and not rows[-1]['matched']


def test_portable_quality_model_rejects_low_confidence_without_changing_pixels():
    maps = np.zeros((3, 32, 32), np.float32)
    maps[0, 3:8, 3:8] = .55
    maps[0, 20:25, 20:25] = .95
    labels = postprocess(maps, mode='components', min_area=1)
    features = quality_features(labels, maps, np.ones((32, 32), np.float32))
    model = {'features': list(FEATURES), 'minimum_quality': .5, 'trees': [
        {'left': [1, -1, -1], 'right': [2, -1, -1], 'feature': [1, -2, -2],
         'threshold': [.7, -2, -2], 'value': [.5, .1, .8]}]}
    assert np.allclose(predict_quality(features, model), [.1, .8])
    filtered = filter_quality(labels, maps, np.ones((32, 32)), model)
    assert np.array_equal(filtered > 0, labels == 2)
    assert filter_quality(np.zeros_like(labels), maps, np.ones((32, 32)), model).max() == 0
