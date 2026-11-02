"""Average matching held-out maps without introducing train-fold predictions."""
import argparse
from pathlib import Path
import numpy as np


def combine(folders, output):
    sources = [{p.name:p for p in Path(f).glob('*.npz')} for f in folders]
    if not sources or not sources[0]:
        raise ValueError('No OOF predictions')
    keys = set(sources[0])
    if any(set(source) != keys for source in sources):
        raise ValueError('OOF members must contain the same held-out image/views')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in sorted(keys):
        members = []
        for source in sources:
            with np.load(source[key]) as z:
                members.append({k:z[k] for k in z.files})
        first = members[0]
        for member in members[1:]:
            for field in ('image_id', 'truth', 'truth_packed', 'valid'):
                if not np.array_equal(first[field], member[field]):
                    raise ValueError(f'OOF identity/target mismatch in {key}: {field}')
        first['probabilities'] = np.mean([m['probabilities'] for m in members], axis=0)
        np.savez_compressed(output/key, **first)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--oof', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    combine(a.oof, a.output)
