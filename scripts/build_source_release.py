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
                ROOT/'paper/quality_filter_diagnostic.png', ROOT/'paper/organizer_report.pdf',
                ROOT/'paper/organizer/main.tex', ROOT/'paper/organizer/preamble.tex',
                ROOT/'paper/organizer/main.bib',
                ROOT/'docs/method_walkthrough.md',
                ROOT/'reference/self_evaluation_source.py', ROOT/'reference/README.md']
    selected += sorted((ROOT/'reference/organizer-template').glob('*'))
    selected += sorted((ROOT/'tests/fixtures').glob('*.tex'))
    selected += sorted((ROOT/'configs').glob('*.json'))
    selected += [ROOT/'scripts'/name for name in (
        'check_submission.py', 'kaggle_train.py', 'kaggle_calibrate_v2.py', 'kaggle_infer_v2.py',
        'kaggle_refine_v2.py', 'kaggle_shape_search.py', 'kaggle_quality_v3.py',
        'kaggle_multiscale_v3.py', 'kaggle_efficient_v3.py', 'package_v3.py',
        'audit_validation_distance.py', 'smoke.py', 'finalize_local_run.py', 'build_source_release.py',
        'prepare_dark_negatives.py', 'redecode_submission.py', 'kaggle_quality_full_v3.py',
        'kaggle_sampling_v4.py', 'kaggle_instances_v5.py', 'audit_quality_full.py', 'build_organizer_report.py')]
    for source in selected:
        target = destination/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (destination/'.gitignore').write_text(
        '__pycache__/\n.pytest_cache/\n.venv/\ndata/\noutputs/\nruns/\n*.partial*\n')
    (destination/'.gitattributes').write_text('* -text\n*.pt filter=lfs diff=lfs merge=lfs -text\n')
    (destination/'README.md').write_text(
        '# Solar filament segmentation, IEEE Big Data Cup 2026\n\n'
        'Research implementation by Arjun Vijay Prakash. The best accepted candidate '
        'is the refined single ConvNeXt model with D4, Kaggle submission 57906671, '
        'public PQ 0.34. The earlier ten-member ensemble scores 0.33. '
        'A top-three result is not established.\n\n'
        'The pipeline normalizes radial illumination, trains native-resolution '
        'segmentation models with full-disk context, and extracts nonoverlapping '
        'filament instances. Training uses only competition-approved segmentation '
        'masks. Independent annotation views stay in the same temporal fold.\n\n'
        'Install requirements.txt in Python 3.12 with a compatible PyTorch build, '
        'then run `python -m pytest -q`. See docs/method_walkthrough.md and '
        'docs/competitive_campaign.md for methodology, scope, and experiment status.\n\n'
        'To reproduce the 0.34 candidate, change into `candidates/refined`, install '
        'its pinned requirements in a Python 3.12 environment with compatible CUDA, '
        'set `SOLAR_TEST_IMAGES` to the official competition test_images directory, '
        'and execute `pipeline.ipynb`. Download Git LFS weights first using '
        '`git lfs pull`. That folder preserves the exact inference source, full-precision '
        'weights, frozen policy, environment, provenance, and submitted CSV. '
        'The root training source includes newer sampling options; use the candidate '
        'folder for exact historical inference.\n\n'
        'Measured inference: 1852.5 seconds for 180 native-resolution observations '
        'on one Tesla T4, including D4 and map-cache writes. Local reference PQ '
        '0.407945 covers complete fold 1 without D4, not the full five-fold D4 candidate.\n\n'
        'The organizer-format report is `paper/organizer_report.pdf`; original IEEE '
        'working notes remain separately available. A paired sampling experiment '
        'and a separate detector-plus-refiner experiment are running. Team and country '
        'are confirmed; final organizer-form delivery remains pending final candidate selection. No rank or award is guaranteed.\n')
    (destination/'source_manifest.json').write_text(json.dumps({
        'status': 'research release with exact portable 0.34 candidate; final organizer delivery pending',
        'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in selected}}, indent=2))
    print(f'Copied {len(selected)} curated files to {destination}')


if __name__ == '__main__':
    main()
