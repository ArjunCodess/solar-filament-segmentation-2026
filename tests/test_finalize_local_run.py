import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# The legacy pilot formatter must be tested against its own historical template,
# so regenerating the measured competition paper cannot change these fixtures.
NOTES = ROOT / 'tests' / 'fixtures' / 'pilot_working_notes.tex'
spec = importlib.util.spec_from_file_location('finalize_local_run', ROOT / 'scripts' / 'finalize_local_run.py')
finalizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finalizer)

PILOT_ROWS = [
    ['unet', '0', 'False', 'False', 'True', 'True', '0.21', '0.18'],
    ['segformer_b3', '0', 'True', 'True', 'True', 'True', '0.34', '0.27'],
    ['convnext_unetpp', '0', 'True', 'True', 'True', 'True', '0.33', '0.26'],
    ['segformer_b3', '0', 'False', 'True', 'True', 'True', '0.30', '0.25'],
]


@pytest.fixture(autouse=True)
def original_notes_stay_unchanged():
    before = NOTES.read_bytes()
    yield
    assert NOTES.read_bytes() == before


def write_inputs(tmp_path):
    notes = tmp_path / 'paper' / 'working_notes.tex'
    notes.parent.mkdir(parents=True)
    notes.write_bytes(NOTES.read_bytes())
    summary = {
        'training_views': 1154,
        'training_annotations': 8199,
        'training_physical_images': 707,
        'test_images': 180,
        'epochs': 3,
        'samples_per_epoch': 384,
        'fold': 0,
        'scope': 'Bounded single-fold pilot, not full five-fold training or a leaderboard result',
        'saved_oof_physical_images': 6,
    }
    summary_path = tmp_path / 'docs' / 'data_summary.json'
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary), encoding='utf-8')
    pilot_path = tmp_path / 'paper' / 'pilot_results.csv'
    with pilot_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(list(finalizer.PILOT_COLUMNS))
        writer.writerows(PILOT_ROWS)
    validation_path = tmp_path / 'submission.validation.json'
    validation_path.write_text(json.dumps({
        'rows': 12,
        'images_with_predictions': 180,
        'images_without_predictions': [],
        'encoding': 'pycocotools compressed COCO counts, Fortran order',
        'format_errors': 0,
    }), encoding='utf-8')
    return {
        'summary': summary_path,
        'pilot': pilot_path,
        'validation': validation_path,
        'notes': notes,
        'build': False,
    }


def test_dry_fixture_replaces_stale_text_with_measured_pilot(tmp_path):
    paths = write_inputs(tmp_path)
    png = paths['notes'].parent / 'qualitative.png'
    png.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 8)
    finalizer.finalize(**paths)
    finalizer.finalize(**paths)
    text = paths['notes'].read_text(encoding='utf-8')
    assert text.count(r'\begin{table}') == 1
    assert text.count('0.3400') == 1
    assert 'Conventional U-Net & 0.2100 & 0.1800' in text
    assert 'SegFormer-B3, full system & 0.3400 & 0.2700' in text
    assert 'ConvNeXt-Small U-Net++, full & 0.3300 & 0.2600' in text
    assert 'SegFormer-B3, no topology & 0.3000 & 0.2500' in text
    assert r'\includegraphics[width=\linewidth]{qualitative.png}' in text
    assert '707' in text and '1154' in text and '8199' in text and '180' in text
    assert r'\caption{Bounded single-fold pilot on fold 0.}' in text
    assert r'\begin{tabular*}{\columnwidth}' in text
    assert '\n\\balance\n' not in text
    assert 'the current verified suite passes 24 tests' in text
    assert 'all 12 tests' not in text
    assert 'not a five-fold result' in text
    assert 'not a leaderboard score' in text
    assert 'organizer notebook version 6' in text
    assert 'Grid MIoU is a local diagnostic' in text
    assert 'Five-fold validation groups physical observations' in text
    assert 'Planned comparisons' not in text
    assert 'anonymous API requests return 401' not in text
    assert r'\pending' not in text.split(r'\label{tab:ablations}')[1].split(r'\end{table}')[0]
    assert not (paths['notes'].parent / (paths['notes'].name + '.partial')).exists()


def test_missing_qualitative_png_is_omitted(tmp_path):
    paths = write_inputs(tmp_path)
    finalizer.finalize(**paths)
    text = paths['notes'].read_text(encoding='utf-8')
    assert 'qualitative.png is absent' in text
    assert 'includegraphics' not in text


@pytest.mark.parametrize('name', ['summary', 'pilot', 'validation', 'notes'])
def test_missing_input_fails_before_write(tmp_path, name):
    paths = write_inputs(tmp_path)
    notes = paths['notes']
    before = notes.read_bytes()
    paths[name].unlink()
    with pytest.raises(FileNotFoundError, match='absent'):
        finalizer.finalize(**paths)
    if name == 'notes':
        assert not notes.exists()
    else:
        assert notes.read_bytes() == before


def test_cli_skip_build_and_rejects_bad_counts(tmp_path):
    paths = write_inputs(tmp_path)
    finalizer.main([
        '--summary', str(paths['summary']),
        '--pilot', str(paths['pilot']),
        '--validation', str(paths['validation']),
        '--notes', str(paths['notes']),
        '--skip-build',
    ])
    assert 'Bounded single-fold pilot' in paths['notes'].read_text(encoding='utf-8')
    summary = json.loads(paths['summary'].read_text(encoding='utf-8'))
    summary['training_physical_images'] = 20
    paths['summary'].write_text(json.dumps(summary), encoding='utf-8')
    before = paths['notes'].read_bytes()
    with pytest.raises(ValueError, match='707'):
        finalizer.finalize(**paths)
    assert paths['notes'].read_bytes() == before
    summary['training_physical_images'] = 707
    paths['summary'].write_text(json.dumps(summary), encoding='utf-8')
    pilot = paths['pilot'].read_text(encoding='utf-8').replace(',0,', ',1,', 1)
    paths['pilot'].write_text(pilot, encoding='utf-8')
    before = paths['notes'].read_bytes()
    with pytest.raises(ValueError, match='fold-0'):
        finalizer.finalize(**paths)
    assert paths['notes'].read_bytes() == before


def test_mid_document_balance_is_removed(tmp_path):
    paths = write_inputs(tmp_path)
    notes = paths['notes']
    notes.write_text(
        notes.read_text(encoding='utf-8').replace(
            r'\section{Implementation verification and experimental protocol}',
            '\\balance\n\\section{Implementation verification and experimental protocol}',
            1,
        ),
        encoding='utf-8',
    )
    finalizer.finalize(**paths)
    assert '\n\\balance\n' not in notes.read_text(encoding='utf-8')


def test_bad_png_fails_before_write(tmp_path):
    paths = write_inputs(tmp_path)
    before = paths['notes'].read_bytes()
    (paths['notes'].parent / 'qualitative.png').write_bytes(b'not a png')
    with pytest.raises(ValueError, match='not a PNG'):
        finalizer.finalize(**paths)
    assert paths['notes'].read_bytes() == before
