import numpy as np
from src.classical import segment


def test_classical_detects_an_unlabeled_dark_ridge_and_stays_on_disk():
    y, x = np.indices((128, 128))
    radius = np.sqrt((x-64)**2+(y-64)**2)
    image = np.uint8(np.where(radius < 54, 130+80*np.sqrt(np.maximum(1-(radius/54)**2, 0)), 0))
    image[62:66, 30:98] = 25
    labels, normalized, response, report = segment(image)
    assert labels.shape == image.shape
    assert labels[62:66, 35:93].any()
    assert not labels[radius > 54].any()
    assert np.isfinite(response).all() and np.isfinite(normalized).all()
    assert report['high_threshold'] >= report['low_threshold']
