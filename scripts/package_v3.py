"""Build immutable, hashed payloads for explicit shape and paired-training jobs."""
import base64
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main(jobs):
    files = [*sorted((ROOT/'src').glob('*.py')), ROOT/'requirements.txt',
             ROOT/'scripts/kaggle_shape_search.py', ROOT/'scripts/kaggle_multiscale_v3.py',
             ROOT/'scripts/kaggle_quality_v3.py', ROOT/'scripts/kaggle_efficient_v3.py',
             ROOT/'scripts/kaggle_quality_full_v3.py', ROOT/'scripts/kaggle_sampling_v4.py',
             ROOT/'scripts/prepare_dark_negatives.py']
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
        baseline = json.loads((ROOT/'configs/baseline_postprocess.json').read_text())
        archive.writestr('baseline_policy.json', json.dumps({'parameters': baseline['parameters']}))
    payload = base64.b64encode(buffer.getvalue()).decode('ascii')
    for name, script, gpu, folds, extra in (
        ('shape-search-v3', 'kaggle_shape_search.py', False, range(5), []),
        ('quality-v3', 'kaggle_quality_v3.py', False, range(5), []),
        ('quality-full-v3', 'kaggle_quality_full_v3.py', True, range(5), [
            'arjunvijayprakash/solar-filament-quality-v3',
            'arjunvijayprakash/solar-filament-shape-search-v3']),
        ('efficient-v3', 'kaggle_efficient_v3.py', True, (), [
            'arjunvijayprakash/solar-filament-refine-v2',
            'arjunvijayprakash/solar-filament-refinement-comparison',
            'arjunvijayprakash/solar-filament-refinement-full-check']),
        ('multiscale-v3', 'kaggle_multiscale_v3.py', True, (0, 1),
         ['arjunvijayprakash/solar-filament-refine-v2']),
        ('sampling-v4', 'kaggle_sampling_v4.py', True, (0, 1),
         ['arjunvijayprakash/solar-filament-refine-v2']),
    ):
        if jobs and name not in jobs:
            continue
        directory = ROOT/'kaggle'/name
        directory.mkdir(parents=True, exist_ok=True)
        (directory/'run.py').write_text(
            "import base64, io, zipfile, runpy\nfrom pathlib import Path\n"
            "root=Path('/kaggle/working/solar')\nroot.mkdir(exist_ok=True)\n"
            f"zipfile.ZipFile(io.BytesIO(base64.b64decode('{payload}'))).extractall(root)\n"
            f"runpy.run_path(str(root/'scripts/{script}'),run_name='__main__')\n", encoding='utf-8')
        (directory/'kernel-metadata.json').write_text(json.dumps({
            'id': f'arjunvijayprakash/solar-filament-{name}', 'title': f'Solar filament {name}',
            'code_file': 'run.py', 'language': 'python', 'kernel_type': 'script',
            'is_private': True, 'enable_gpu': gpu, 'enable_internet': True,
            'dataset_sources': [], 'competition_sources': ['filament-segmentation-2026'] if gpu else [],
            'kernel_sources': [f'arjunvijayprakash/solar-filament-fold-{f}' for f in folds]+extra}, indent=2))
        provenance = {
            'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'payload_sha256': hashlib.sha256(buffer.getvalue()).hexdigest()}
        (directory/'source_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'Packaged {len(files)} source files for {jobs or "all v3 jobs"}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', action='append', choices=(
        'shape-search-v3', 'quality-v3', 'multiscale-v3', 'efficient-v3', 'quality-full-v3', 'sampling-v4'))
    main(parser.parse_args().job)
