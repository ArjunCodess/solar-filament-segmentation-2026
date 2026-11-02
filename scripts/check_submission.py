"""Validate a saved competition CSV against test-image identities and dimensions."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from src.submission import validate_submission

p = argparse.ArgumentParser()
p.add_argument('submission')
p.add_argument('--images', required=True)
a = p.parse_args()
shapes = {}
for path in Path(a.images).iterdir():
    if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        continue
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape != (2048, 2048):
        raise ValueError(f'Unreadable or nonstandard test image: {path}')
    if path.stem in shapes:
        raise ValueError(f'Duplicate test image: {path.stem}')
    shapes[path.stem] = image.shape
if not shapes:
    raise ValueError('Empty test directory')
print(json.dumps(validate_submission(a.submission, shapes), indent=2))
