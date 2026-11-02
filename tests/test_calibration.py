import numpy as np
import pytest
from src.calibration import component_trials
from src.infer import postprocess
from src.metrics import label_prediction_scores, binary_scores


def test_hysteresis_retains_connected_faint_barbs_and_rejects_unseeded_noise():
    maps = np.zeros((3, 32, 32), np.float32)
    maps[0, 14:18, 5:18] = .9
    maps[0, 15, 18:28] = .45
    maps[0, 3:6, 3:6] = .45
    maps[1, 15, [7, 10, 13]] = .9
    valid = np.ones((32, 32), bool)
    valid[:, 26:] = False
    labels = postprocess(maps, valid, mode='components', threshold=.4,
                         seed_threshold=.8, min_area=4)
    assert labels.max() == 1
    assert labels[15, 25] == 1 and labels[15, 26] == 0
    assert not labels[3:6, 3:6].any()
    with pytest.raises(ValueError, match='Seed threshold'):
        postprocess(maps, threshold=.8, seed_threshold=.4)


def test_component_calibration_matches_independent_pixel_mask_scoring():
    rng = np.random.default_rng(72)
    probability = rng.random((48, 48))
    valid = rng.random((48, 48)) > .1
    truth = [rng.random((48, 48)) > .7 for _ in range(3)]
    truth.append(np.zeros((48, 48), bool))
    for row in component_trials(probability, valid, truth, thresholds=(.4, .8), areas=(1, 8)):
        labels = postprocess(np.stack([probability, probability*0, probability*0]), valid, **row['parameters'])
        score = label_prediction_scores(truth, labels)
        for key in ('pq_iou_sum', 'pq_tp', 'pq_fp', 'pq_fn'):
            assert row[key] == pytest.approx(score[key])
        assert row['dice'] == pytest.approx(binary_scores(np.logical_or.reduce(truth), labels > 0)['dice'])


def test_component_calibration_empty_truth_and_prediction():
    row, = component_trials(np.zeros((8, 8)), np.ones((8, 8), bool), [],
                            thresholds=(.5,), areas=(1,), seed_gaps=(0.,))
    assert row['pq_tp'] == row['pq_fp'] == row['pq_fn'] == 0
    assert row['dice'] == 1
