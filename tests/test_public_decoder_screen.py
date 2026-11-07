import json
from pathlib import Path

import numpy as np
import pytest

from scripts.screen_public_decoder import run


def test_screen_rejects_same_physical_image_under_different_annotator_ids(tmp_path):
    upstream = Path(__file__).resolve().parents[1]/'reference/upstream/farnoosh-run3'
    if not (upstream/'solarseg/postprocess.py').exists():
        pytest.skip('Optional licensed upstream checkout is not installed')
    for directory, image_id in (('calibration', 'annotator_a'), ('evaluation', 'annotator_b')):
        folder = tmp_path/directory
        folder.mkdir()
        np.savez(folder/'map.npz', image_id=image_id, raw=np.zeros((8, 8)))
    with pytest.raises(ValueError, match='physical observations overlap'):
        run(tmp_path/'calibration', tmp_path/'evaluation', upstream, {}, tmp_path/'out')


def test_public_decoder_screen_records_frozen_policy_and_disallows_promotion(tmp_path):
    upstream = Path(__file__).resolve().parents[1]/'reference/upstream/farnoosh-run3'
    if not (upstream/'solarseg/postprocess.py').exists():
        pytest.skip('Optional licensed upstream checkout is not installed')
    for index, directory in enumerate(('calibration', 'evaluation')):
        folder = tmp_path/directory
        folder.mkdir()
        mask = np.zeros((32, 32), bool)
        mask[8:24, 8:24] = True
        probability = np.zeros((3, 32, 32), np.float32)
        probability[0, mask] = .99
        np.savez(folder/'map.npz', image_id=str(index), raw=np.full(mask.shape, index),
                 valid=np.ones_like(mask), probabilities=probability,
                 truth_packed=np.packbits(mask.reshape(1, -1), axis=1))
    output = tmp_path/'out'
    run(tmp_path/'calibration', tmp_path/'evaluation', upstream,
        {'mode': 'components', 'min_area': 64}, output)
    report = json.loads((output/'complete.json').read_text())
    assert report['pq'] == {'baseline': 1., 'public_decoder': 1.}
    assert report['promotion'] is False
    assert json.loads((output/'frozen_policy.json').read_text())['calibration_views'] == 1
