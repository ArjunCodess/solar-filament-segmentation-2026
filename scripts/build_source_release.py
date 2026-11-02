"""Copy a curated source release; never include local data, credentials, or caches."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT/'release/public'
    destination.mkdir(parents=True, exist_ok=True)
    selected = [*sorted((ROOT/'src').glob('*.py')), *sorted((ROOT/'tests').glob('*.py')),
                ROOT/'requirements.txt', ROOT/'requirements-tested-cpu.txt',
                ROOT/'paper/working_notes.tex', ROOT/'paper/working_notes.pdf',
                ROOT/'paper/kaggle_qualitative.png', ROOT/'docs/competitive_campaign.md',
                ROOT/'docs/method_walkthrough.md',
                ROOT/'reference/self_evaluation_source.py', ROOT/'reference/README.md']
    selected += sorted((ROOT/'tests/fixtures').glob('*.tex'))
    selected += sorted((ROOT/'configs').glob('*.json'))
    selected += [ROOT/'scripts'/name for name in (
        'check_submission.py', 'kaggle_train.py', 'kaggle_calibrate_v2.py', 'kaggle_infer_v2.py',
        'kaggle_refine_v2.py', 'kaggle_shape_search.py', 'kaggle_quality_v3.py',
        'kaggle_multiscale_v3.py', 'kaggle_efficient_v3.py', 'package_v3.py',
        'audit_validation_distance.py', 'smoke.py', 'finalize_local_run.py', 'build_source_release.py')]
    for source in selected:
        target = destination/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (destination/'.gitignore').write_text(
        '__pycache__/\n.pytest_cache/\n.venv/\ndata/\noutputs/\nruns/\n*.partial*\n')
    (destination/'.gitattributes').write_text('* -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n')
    (destination/'README.md').write_text(
        '# Solar filament segmentation, IEEE Big Data Cup 2026\n\n'
        'Research implementation by Arjun Vijay Prakash. The accepted five-fold '
        'submission currently has public PQ 0.33. New experiments are in progress; '
        'this source release is not yet an organizer-ready final submission.\n\n'
        'The pipeline normalizes radial illumination, trains native-resolution '
        'segmentation models with full-disk context, and extracts nonoverlapping '
        'filament instances. Training uses only competition-approved segmentation '
        'masks. Independent annotation views stay in the same temporal fold.\n\n'
        'Install requirements.txt in Python 3.12 with a compatible PyTorch build, '
        'then run `python -m pytest -q`. See docs/method_walkthrough.md and '
        'docs/competitive_campaign.md for methodology, scope, and experiment status.\n\n'
        'Weights and the final reproduction notebook will be included after model '
        'selection. The current source snapshot alone cannot reproduce the submitted '
        'model predictions. The four-page paper is working notes.\n')
    (destination/'source_manifest.json').write_text(json.dumps({
        'status': 'source snapshot; final model and weights pending',
        'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in selected}}, indent=2))
    print(f'Copied {len(selected)} curated files to {destination}')


if __name__ == '__main__':
    main()
