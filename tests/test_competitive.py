import csv
import json
import cv2
import numpy as np
import pytest
import torch

from src.dataset import CropDataset
from src.error_analysis import instance_error_rows
from src.infer import postprocess
from src.submission import encode, validate_submission
from src.submission import predict
from src.models import build_model
from types import SimpleNamespace
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


def test_prediction_cache_matches_csv_image_identity(tmp_path):
    torch.set_num_threads(2)
    yy, xx = np.indices((128, 128))
    radius = np.hypot(xx-64, yy-64)
    raw = np.uint8(np.where(radius < 54, 140+70*np.sqrt(np.maximum(1-(radius/54)**2, 0)), 0))
    images = tmp_path/'images'
    images.mkdir()
    cv2.imwrite(str(images/'observation.png'), raw)
    checkpoint = tmp_path/'tiny.pt'
    torch.save({'model': build_model('unet', False).state_dict(), 'config': {
        'architecture': 'unet', 'preprocess': {'clahe': True, 'limb': True}, 'no_auxiliary': True}}, checkpoint)
    args = SimpleNamespace(checkpoints=[checkpoint], weights=None, device='cpu', params=None,
        quality_model=None, maps_output=tmp_path/'maps', images=images, allow_nonstandard_size=True,
        tile=64, tta=False, tta_batch_size=1, output=tmp_path/'submission.csv')
    predict(args)
    with np.load(args.maps_output/'observation.npz') as cache:
        assert str(cache['image_id']) == 'observation'
        assert cache['probabilities'].shape == (3, 128, 128)
        assert np.array_equal(cache['raw'], raw) and np.isfinite(cache['probabilities']).all()
    assert validate_submission(args.output, {'observation': raw.shape})['overlapping_pixels'] == 0


def test_instance_balancing_reaches_small_objects(tmp_path):
    image = np.zeros((96, 96), np.uint8)
    image[8:38, 8:38] = 80
    image[75:79, 75:79] = 220
    instances = np.zeros_like(image, np.uint16)
    instances[8:38, 8:38] = 1
    instances[75:79, 75:79] = 2
    image_cache, target = tmp_path/'image.npz', tmp_path/'target.npz'
    np.savez(image_cache, image=np.stack([image, image]), valid=np.ones_like(image, bool))
    mask = np.uint8(instances > 0)*255
    np.savez(target, mask=mask, spine=mask, boundary=mask, skeleton=mask, instances=instances)
    records = [{'image_cache': str(image_cache), 'cache': str(target)}]
    dataset = CropDataset(records, crop=16, samples=32, augment=False,
                          foreground_probability=1., instance_balanced=True)
    small = sum(float(dataset[index][0].max()) > .6 for index in range(32))
    assert 8 <= small <= 24


def test_dark_negative_sampler_crops_the_requested_distractor(tmp_path):
    image = np.zeros((96, 96), np.uint8)
    image[70, 70] = 255
    zero = np.zeros_like(image)
    image_cache, target = tmp_path/'image.npz', tmp_path/'target.npz'
    np.savez(image_cache, image=np.stack([image, image]), valid=np.ones_like(image, bool))
    np.savez(target, mask=zero, spine=zero, boundary=zero, skeleton=zero)
    dataset = CropDataset([{'image_cache': str(image_cache), 'cache': str(target),
                           'dark_negative_centers': [[70, 70]]}], crop=16, samples=4,
                          augment=False, foreground_probability=0., dark_negative_probability=1.)
    for index in range(4):
        x, _, y = dataset[index]
        assert x.max() == 1 and not y[:4].any()


def test_dark_negative_preparation_excludes_all_annotators_and_heldout(tmp_path, monkeypatch):
    from scripts.prepare_dark_negatives import main
    image = np.full((96, 96), 180, np.uint8)
    image[48:54, 48:54] = 0
    image[75:81, 10:16] = 0
    image_cache = tmp_path/'image.npz'
    np.savez(image_cache, image=image, valid=np.ones_like(image, bool))
    records = []
    for index, region in enumerate(((slice(10, 16), slice(10, 16)), (slice(48, 54), slice(48, 54)))):
        mask = np.zeros_like(image)
        mask[region] = 255
        cache = tmp_path/f'view{index}.npz'
        np.savez(cache, mask=mask)
        records.append({'id': str(index), 'sha256': 'training', 'cache': str(cache), 'image_cache': str(image_cache)})
    # These nonexistent held-out paths cause a failure if the miner reads validation data.
    records.append({'id': 'heldout', 'sha256': 'heldout', 'cache': 'missing-target.npz', 'image_cache': 'missing-image.npz'})
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'records': records}))
    manifest.with_name('folds.csv').write_text('image_id,fold\n0,1\n1,1\nheldout,0\n')
    output = tmp_path/'derived/manifest.json'
    monkeypatch.setattr('sys.argv', ['prepare_dark_negatives', '--manifest', str(manifest), '--fold', '0', '--output', str(output)])
    main()
    prepared = json.loads(output.read_text())
    assert 'dark_negative_centers' not in prepared['records'][-1]
    centers = prepared['records'][0]['dark_negative_centers']
    assert centers and centers == prepared['records'][1]['dark_negative_centers']
    assert all(not (38 <= y < 64 and 38 <= x < 64) for y, x in centers)
