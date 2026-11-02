import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from src.dataset import make_folds, observation_date
from src.preprocess import detect_disk, normalize_disk, targets
from src.metrics import binary_scores, grid_miou_diagnostic, instance_scores
from src.infer import d4, undo_d4, tiled_predict, postprocess
from src.losses import composite_loss, cldice_loss
from src.models import build_model
from src.submission import encode, decode, validate_submission


@pytest.fixture
def solar():
    y, x = np.indices((128, 128))
    r = np.sqrt((x-63)**2+(y-65)**2)
    return np.uint8(np.where(r < 52, 100+100*np.sqrt(np.maximum(1-(r/52)**2, 0)), 0))


def test_disk_and_radial_normalization(solar):
    disk = detect_disk(solar)
    assert abs(disk.xc-63) < 2 and abs(disk.yc-65) < 2 and abs(disk.radius-52) < 3
    norm, valid, _ = normalize_disk(solar)
    assert np.isfinite(norm).all() and norm.min() >= 0 and norm.max() <= 1
    assert not norm[~valid].any()
    with pytest.raises(ValueError):
        detect_disk(np.zeros((32, 32)))


def test_native_targets_and_rle():
    ann = {'segmentation':[[10, 10, 50, 10, 50, 14, 10, 14]], 'spine':[10, 12, 49, 12]}
    result = targets([ann], (64, 64), use_spines=True)
    assert result['skeleton'].sum() > 20 and result['boundary'].sum() > 0
    assert np.array_equal(decode(encode(result['mask']), (64, 64)), result['mask'])
    compressed = {'segmentation':{'size':[64, 64], 'counts':encode(result['mask'])}}
    assert np.array_equal(targets([compressed], (64, 64))['mask'], result['mask'])


def test_metrics_penalize_fragmentation_and_extra_objects():
    g = np.zeros((32, 32), bool)
    g[4:8, 4:28] = True
    assert binary_scores(g, g)['iou'] == 1
    assert grid_miou_diagnostic(g, g) == 1
    assert instance_scores([g], [g])['pq'] == 1
    extra = np.zeros_like(g)
    extra[20:22, 20:22] = True
    assert instance_scores([g], [g, extra], [.9, .1])['pq'] < 1
    left = g.copy()
    left[:, 16:] = False
    right = g & ~left
    assert instance_scores([g], [left, right])['pq'] == 0  # exactly 0.5 does not match
    assert instance_scores([g], [extra, g], [.9, .1])['ap@0.50'] == pytest.approx(.5)
    assert binary_scores(np.zeros_like(g), np.zeros_like(g))['iou'] == 1


def test_temporal_and_annotator_leakage():
    records, annotations = [], {}
    for group in range(15):
        for day in (0, 1):
            date = datetime(2012+group%10, 1, 1)+timedelta(days=group*10+day)
            for view in range(2):
                key = f'{group}-{day}-{view}'
                records.append({'id':key, 'file_name':date.strftime('%Y%m%d%H%M%S')+'Bh.jpeg', 'width':64, 'height':64})
                annotations[key] = [{'area':30, 'bbox':[1, 1, 10, 3]}]
    folds = make_folds(records, annotations)
    assert folds.fold.nunique() == 5
    assert folds.groupby('file_name').fold.nunique().max() == 1
    for a in records:
        for b in records:
            if abs((observation_date(a)-observation_date(b)).total_seconds()) <= 3*86400:
                f = folds.set_index('image_id').fold
                assert f[str(a['id'])] == f[str(b['id'])]


def test_d4_inverse_and_tiling():
    x = torch.arange(2*35*47).reshape(1, 2, 35, 47)
    for k in range(4):
        for flip in (False, True):
            assert torch.equal(x, undo_d4(d4(x, k, flip), k, flip))
    class Identity(torch.nn.Module):
        def forward(self, x, context):
            return x[:, :1].expand(-1, 3, -1, -1)
    image = np.random.default_rng(1).random((2, 71, 89), dtype=np.float32)
    for batch_size in (1, 4, 8):
        pred = tiled_predict(Identity(), image, tile=64, overlap=16, tta=True, tta_batch_size=batch_size)
        assert np.allclose(pred[0], torch.sigmoid(torch.tensor(image[0])).numpy(), atol=1e-6)


def test_loss_finite_with_topology_gradients():
    torch.set_num_threads(2)
    logits = torch.zeros((2, 3, 32, 32), requires_grad=True)
    y = torch.zeros((2, 5, 32, 32))
    y[:, 4] = 1
    y[0, 0, 15:18, 3:29] = 1
    y[0, 3, 16, 3:29] = 1
    loss, parts = composite_loss(logits, y)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 16].abs().sum() > 0
    assert all(np.isfinite(v) for v in parts.values())


@pytest.mark.parametrize('architecture', ['unet', 'segformer_b3', 'convnext_unetpp'])
def test_model_contract(architecture):
    torch.set_num_threads(2)
    model = build_model(architecture, pretrained=False).eval()
    with torch.no_grad():
        result = model(torch.rand(1, 2, 64, 64), torch.rand(1, 1, 128, 128))
    assert result.shape == (1, 3, 64, 64) and torch.isfinite(result).all()


def test_submission_roundtrip_and_rejection(tmp_path):
    mask = np.zeros((32, 32), bool)
    mask[3:7, 5:25] = 1
    path = tmp_path/'submission.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filament_id', 'segmentation_rle'])
        writer.writerow(['20150125172714Mh_1', encode(mask)])
    assert validate_submission(path, {'20150125172714Mh':(32, 32)})['format_errors'] == 0
    with path.open('a', newline='') as f:
        csv.writer(f).writerow(['20150125172714Mh_1', encode(mask)])
    with pytest.raises(ValueError, match='Duplicate'):
        validate_submission(path, {'20150125172714Mh':(32, 32)})


def test_postprocess_boundary_separation():
    maps = np.zeros((3, 64, 64), np.float32)
    maps[0, 20:28, 5:59] = .9
    maps[1, 24, 8:25] = .9
    maps[1, 24, 40:56] = .9
    maps[2, 20:28, 31:33] = 1
    result = postprocess(maps, min_area=4)
    assert result.max() == 2 and (result > 0).sum() == (maps[0] > .5).sum()


def test_crop_dataset_compact_cache_shapes(solar, tmp_path):
    from src.dataset import CropDataset, prepare
    images = tmp_path/'images'
    images.mkdir()
    records = []
    for i in range(5):
        name = f'201{1+i}0101000000Bh.jpeg'
        img = solar.copy()
        img[60+i, 60] = 50
        cv2.imwrite(str(images/name), img)
        records.append({'id':str(i), 'file_name':name, 'height':128, 'width':128})
    anns = [{'id':str(i), 'image_id':str(i), 'segmentation':[[20, 20, 40, 20, 40, 24, 20, 24]],
             'bbox':[20, 20, 20, 4], 'area':80} for i in range(5)]
    coco = tmp_path/'annotations.json'
    coco.write_text(json.dumps({'images':records, 'annotations':anns}))
    manifest = prepare(coco, images, tmp_path/'cache', compact_images=True)
    x, context, y = CropDataset(manifest, crop=64, samples=2)[0]
    assert x.shape == (2, 64, 64) and context.shape == (1, 128, 128) and y.shape == (5, 64, 64)
    assert x.dtype == torch.float32 and context.dtype == torch.float32 and y.dtype == torch.float32
    assert torch.isfinite(x).all() and torch.isfinite(context).all() and torch.isfinite(y).all()


def test_mid_epoch_resume_skips_io_and_preserves_samples(solar, tmp_path):
    from unittest.mock import patch
    from torch.utils.data import DataLoader
    from src.dataset import CropDataset

    image_path = tmp_path/'raw.jpeg'
    cv2.imwrite(str(image_path), solar)
    image_cache, cache = tmp_path/'image.npz', tmp_path/'targets.npz'
    np.savez(image_cache, image=solar, valid=np.ones_like(solar, dtype=bool))
    target = np.zeros_like(solar)
    target[60:65, 35:90] = 255
    np.savez(cache, mask=target, spine=target, boundary=target, skeleton=target)
    records = [{'image_cache':str(image_cache), 'cache':str(cache), 'image_path':str(image_path)}]
    uninterrupted = CropDataset(records, crop=64, samples=6, seed=19)
    resumed = CropDataset(records, crop=64, samples=6, seed=19)
    uninterrupted.epoch = resumed.epoch = 3
    batch_size, resume_step = 2, 2
    resumed.skip_before = resume_step*batch_size

    with patch('src.dataset.np.load', wraps=np.load) as load, patch('src.dataset.cv2.imread', wraps=cv2.imread) as read:
        batches = iter(DataLoader(resumed, batch_size=batch_size, num_workers=0))
        for _ in range(resume_step):
            assert all(tensor.shape == (batch_size, 0) for tensor in next(batches))
        load.assert_not_called()
        read.assert_not_called()
        actual = next(batches)
        assert load.call_count == 2*batch_size
        assert read.call_count == batch_size

    expected = [uninterrupted[index] for index in range(4, 6)]
    for head, tensor in enumerate(actual):
        assert torch.equal(tensor, torch.stack([sample[head] for sample in expected]))
    resumed.epoch += 1
    resumed.skip_before = 0
    assert resumed[0][0].shape == (2, 64, 64)

