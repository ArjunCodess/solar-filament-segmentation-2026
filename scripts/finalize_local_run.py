"""Insert a measured fold-0 pilot into the working notes.

Reads docs/data_summary.json, paper/pilot_results.csv, and
submission.validation.json. Fails before editing the paper when any required
input is missing or inconsistent. Grid MIoU is a local diagnostic. Reference
PQ is the organizer notebook version 6 all-pairs IoU>0.5 score. This does not
train, download, or claim five-fold or leaderboard performance.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_COUNTS = {
    'training_physical_images': 707,
    'training_views': 1154,
    'training_annotations': 8199,
    'test_images': 180,
}
PILOT_COLUMNS = (
    'architecture', 'fold', 'topology', 'auxiliary', 'limb', 'clahe',
    'pq', 'grid_miou_diagnostic',
)
ARCH_NAMES = {
    'unet': 'Conventional U-Net',
    'segformer_b3': 'SegFormer-B3',
    'convnext_unetpp': 'ConvNeXt-Small U-Net++',
}
STALE = {
    'PILOT_ABSTRACT': (
        'Competition data and trained competition weights remain unavailable, '
        'so no competition accuracy or leaderboard result is claimed.'
    ),
    'PILOT_PROTOCOL': (
        r'The planned five-fold comparison in Table~\ref{tab:ablations} holds fold identities, '
        'seeds, crop size, training budget, and inference settings fixed. Checkpoints are selected '
        'using training loss, keeping outer folds unused until final inference; hyperparameter studies '
        'require separate development folds. Preprocessing ablations require separate caches. '
        'Checkpoints and out-of-fold maps are saved per member and fold. Final reports should include '
        'score distributions, image-level uncertainty, runtimes, memory, and one-to-many/many-to-one counts.'
    ),
    'PILOT_QUALITATIVE': (
        'No such comparison is presented here because real baseline and proposed-model predictions '
        'are unavailable. Invented masks would not demonstrate preservation of solar filament barbs.'
    ),
    'PILOT_ACCESS': (
        'Competition training remains blocked by data access: anonymous API requests return 401 '
        'and no credential was supplied. The organizer email remains unsent.'
    ),
    'PILOT_CLOSE': (
        'Final working notes require genuine held-out measurements, qualitative comparisons, '
        'authors and affiliations, and verified deadlines after data and compute access are resolved.'
    ),
}


def as_bool(value):
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes'}:
        return True
    if text in {'0', 'false', 'no'}:
        return False
    raise ValueError(f'Expected a boolean pilot flag, got {value!r}')


def metric(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'pilot row has no numeric {key}') from error
    if value != value or value in {float('inf'), float('-inf')} or not 0 <= value <= 1:
        raise ValueError(f'{key} must be finite and in [0, 1], got {value}')
    return value


def configuration_label(row):
    architecture = row['architecture'].strip()
    if not architecture:
        raise ValueError('pilot_results.csv has a row without architecture')
    name = ARCH_NAMES.get(architecture, architecture.replace('_', ' '))
    topology, auxiliary = as_bool(row['topology']), as_bool(row['auxiliary'])
    limb, clahe = as_bool(row['limb']), as_bool(row['clahe'])
    if architecture == 'unet' and not topology and not auxiliary and limb and clahe:
        return 'Conventional U-Net'
    if topology and auxiliary and limb and clahe:
        qualifier = 'full' if architecture == 'convnext_unetpp' else 'full system'
        return f'{name}, {qualifier}'
    parts = []
    if not topology:
        parts.append('no topology')
    if not auxiliary:
        parts.append('no auxiliary')
    if not limb:
        parts.append('no radial correction')
    if not clahe:
        parts.append('no CLAHE')
    return f'{name}, {", ".join(parts)}' if parts else name


def latex_text(value):
    return (str(value)
            .replace('\\', r'\textbackslash{}')
            .replace('&', r'\&')
            .replace('%', r'\%')
            .replace('$', r'\$')
            .replace('#', r'\#')
            .replace('_', r'\_')
            .replace('{', r'\{')
            .replace('}', r'\}'))


def pilot_table(rows):
    body = '\n'.join(
        f'{latex_text(configuration_label(row))} & {metric(row, "pq"):.4f} & '
        f'{metric(row, "grid_miou_diagnostic"):.4f}\\\\'
        for row in rows)
    # The caption stays one column wide. The five-fold and leaderboard
    # disclaimer remains in the protocol paragraph, which is allowed to wrap.
    return '\n'.join([
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Bounded single-fold pilot on fold 0.}',
        r'\label{tab:ablations}',
        r'\footnotesize',
        r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc@{}}',
        r'\toprule',
        r'Configuration & Reference PQ & Grid MIoU$^*$\\',
        r'\midrule',
        body,
        r'\bottomrule',
        r'\end{tabular*}',
        r'\vspace{0.4ex}',
        r'\parbox{\columnwidth}{\raggedright $^*$Grid MIoU is a local diagnostic, not an official MIoU. '
        r'Reference PQ is the organizer notebook version 6 all-pairs IoU$>$0.5 score.}',
        r'\end{table}',
    ])


def fresh_text(summary, validation, has_figure):
    counts = (
        f'Official competition data contain {summary["training_physical_images"]} physical training images, '
        f'{summary["training_views"]} annotation views, {summary["training_annotations"]} annotations, '
        f'and {summary["test_images"]} test images.'
    )
    fold = summary['fold']
    if has_figure:
        qualitative = '\n'.join([
            rf'Figure~\ref{{fig:qualitative}} shows one held-out pilot view. '
            'The example illustrates the remaining false positives.',
            r'\begin{figure}[t]',
            r'\centering',
            r'\includegraphics[width=\linewidth]{qualitative.png}',
            r'\caption{One held-out fold-0 pilot view and a native-pixel detail.}',
            r'\label{fig:qualitative}',
            r'\end{figure}',
        ])
    else:
        qualitative = (
            'No qualitative panel is included because qualitative.png is absent. '
            'Invented masks are not substituted.'
        )
    absent = len(validation['images_without_predictions'])
    return {
        'PILOT_ABSTRACT': (
            f'{counts} The measurements in Table~\\ref{{tab:ablations}} are a bounded '
            f'single-fold pilot on fold {fold}; no leaderboard score is claimed.'
        ),
        'PILOT_PROTOCOL': (
            rf'Table~\ref{{tab:ablations}} lists only the measured fold-{fold} pilot '
            f'({summary["epochs"]} epochs, {summary["samples_per_epoch"]} samples per epoch, '
            f'{summary["saved_oof_physical_images"]} saved out-of-fold physical images). '
            'Reference PQ is the organizer notebook version 6 all-pairs score: every ground-truth/prediction '
            'pair with IoU greater than 0.5 is counted, and counts are aggregated before division. '
            'Grid MIoU is a local diagnostic and is not an official MIoU. '
            'Configurations that were not trained are omitted. '
            'This is not a five-fold result and not a leaderboard score.'
        ),
        'PILOT_QUALITATIVE': qualitative,
        'PILOT_ACCESS': (
            f'{counts} '
            f'Local validation of the submission file reports {validation["rows"]} rows, '
            f'{validation["images_with_predictions"]} images with predictions, '
            f'{absent} images without predictions, and {validation["format_errors"]} format errors. '
            'This validates file encoding, not segmentation accuracy.'
        ),
        'PILOT_CLOSE': (
            'Authors, affiliations, and the unfinished planned ablations remain open. '
        ),
    }


def splice(text, marker, stale, fresh):
    begin, end = f'% {marker}', f'% END_{marker}'
    block = f'{begin}\n{fresh}\n{end}'
    start = text.find(begin)
    if start >= 0:
        stop = text.find(end, start)
        if stop < 0:
            raise ValueError(f'Unclosed {marker} marker in working notes')
        return text[:start] + block + text[stop + len(end):]
    found = text.count(stale)
    if found != 1:
        raise ValueError(f'Expected one stale anchor for {marker}, found {found}')
    return text.replace(stale, block, 1)


def replace_table(text, table):
    label = r'\label{tab:ablations}'
    at = text.find(label)
    if at < 0:
        raise ValueError('working_notes.tex has no tab:ablations anchor')
    start = text.rfind(r'\begin{table}', 0, at)
    stop = text.find(r'\end{table}', at)
    if start < 0 or stop < 0:
        raise ValueError('tab:ablations is not inside a table environment')
    stop += len(r'\end{table}')
    block = text[start:stop]
    if 'Planned comparisons' not in block and 'Bounded single-fold pilot' not in block:
        raise ValueError('Refusing to replace a table that is not the planned-results table')
    return text[:start] + table + text[stop:]


def strip_column_balance(text):
    """Drop a mid-document \\balance command.

    The balance package replaces the two-column output routine for every
    following page. Issued before the last column, it stretches those columns
    past the IEEEtran text height and build_paper.py rejects the Overfull vbox.
    """
    return text.replace('\n\\balance\n', '\n')


def render(text, summary, rows, validation, has_figure):
    updated = strip_column_balance(replace_table(text, pilot_table(rows)))
    for marker, fresh in fresh_text(summary, validation, has_figure).items():
        updated = splice(updated, marker, STALE[marker], fresh)
    lowered = updated.lower()
    for phrase in (
        'competition data and trained competition weights remain unavailable',
        'planned comparisons. no competition measurements are available.',
        'the planned five-fold comparison in table',
        'real baseline and proposed-model predictions are unavailable',
        'anonymous api requests return 401',
        'after data and compute access are resolved',
    ):
        if phrase in lowered:
            raise ValueError(f'Stale text remains in working notes: {phrase}')
    for phrase in (
        'bounded single-fold pilot',
        'not a five-fold result',
        'not a leaderboard score',
        'organizer notebook version 6',
        'grid miou is a local diagnostic',
        'all-pairs',
        'not an official miou',
    ):
        if phrase not in lowered:
            raise ValueError(f'Pilot text is missing required wording: {phrase}')
    included = r'\includegraphics[width=\linewidth]{qualitative.png}' in updated
    if has_figure and not included:
        raise ValueError('qualitative.png is present but was not included')
    if not has_figure and 'includegraphics' in updated:
        raise ValueError('qualitative figure was included without qualitative.png')
    return updated


def require_int(data, key, source):
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f'{source} {key} must be an integer')
    return value


def load_summary(path):
    try:
        summary = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'{path.name} is not valid JSON: {error}') from error
    if not isinstance(summary, dict):
        raise ValueError(f'{path.name} must be a JSON object')
    for key, expected in OFFICIAL_COUNTS.items():
        got = require_int(summary, key, path.name)
        if got != expected:
            raise ValueError(f'{path.name} {key} is {got}, expected the acquired official count {expected}')
    if require_int(summary, 'fold', path.name) != 0:
        raise ValueError(f'{path.name} fold is {summary["fold"]}, expected the single pilot fold 0')
    for key in ('epochs', 'samples_per_epoch'):
        if require_int(summary, key, path.name) < 1:
            raise ValueError(f'{path.name} {key} must be positive')
    if require_int(summary, 'saved_oof_physical_images', path.name) < 0:
        raise ValueError(f'{path.name} saved_oof_physical_images must be non-negative')
    scope = str(summary.get('scope', '')).lower()
    if 'single-fold' not in scope and 'single fold' not in scope:
        raise ValueError(f'{path.name} scope must describe a single-fold pilot')
    if 'five-fold' not in scope and 'five fold' not in scope:
        raise ValueError(f'{path.name} scope must say this is not full five-fold training')
    if 'leaderboard' not in scope or 'not' not in scope:
        raise ValueError(f'{path.name} scope must say this is not a leaderboard result')
    return summary


def load_pilot(path):
    text = path.read_text(encoding='utf-8')
    if 'synthetic' in text.lower():
        raise ValueError(f'{path.name} contains synthetic metrics; refusing to update the paper')
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise ValueError(f'{path.name} has no measured rows')
    missing = [column for column in PILOT_COLUMNS if column not in rows[0]]
    if missing:
        raise ValueError(f'{path.name} is missing columns: {", ".join(missing)}')
    folds = {row['fold'].strip() for row in rows}
    if folds != {'0'}:
        raise ValueError(f'{path.name} folds are {sorted(folds)}; only a fold-0 pilot can be written')
    return rows


def load_validation(path, test_images):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as error:
        raise ValueError(f'{path.name} is not valid JSON: {error}') from error
    if not isinstance(data, dict):
        raise ValueError(f'{path.name} must be a JSON object')
    missing_images = data.get('images_without_predictions')
    if not isinstance(missing_images, list):
        raise ValueError(f'{path.name} images_without_predictions must be a list')
    rows = require_int(data, 'rows', path.name)
    covered = require_int(data, 'images_with_predictions', path.name)
    errors = require_int(data, 'format_errors', path.name)
    if min(rows, covered, errors) < 0:
        raise ValueError(f'{path.name} counts must be non-negative')
    if errors != 0:
        raise ValueError(f'{path.name} reports {errors} format errors; refusing to update the paper')
    if covered + len(missing_images) != test_images:
        raise ValueError(
            f'{path.name} covers {covered} images with predictions and {len(missing_images)} without, '
            f'which does not equal {test_images} official test images')
    return data


def png_present(path):
    if not path.is_file():
        return False
    if path.read_bytes()[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError(f'{path} is not a PNG; refusing to include it')
    return True


def invoke_build(root):
    script = Path(root) / 'scripts' / 'build_paper.py'
    if not script.is_file():
        raise FileNotFoundError(f'Required paper build script is absent: {script}')
    completed = subprocess.run([sys.executable, str(script)], cwd=root)
    if completed.returncode:
        raise RuntimeError(f'build_paper.py failed with exit code {completed.returncode}')


def finalize(summary, pilot, validation, notes, build=False, root=None):
    summary, pilot, validation, notes = map(Path, (summary, pilot, validation, notes))
    missing = [path for path in (summary, pilot, validation, notes) if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required pilot inputs are absent: ' + ', '.join(str(path) for path in missing))
    summary_data = load_summary(summary)
    rows = load_pilot(pilot)
    validation_data = load_validation(validation, summary_data['test_images'])
    has_figure = png_present(notes.parent / 'qualitative.png')
    original = notes.read_text(encoding='utf-8')
    updated = render(original, summary_data, rows, validation_data, has_figure)
    if updated != original:
        temporary = notes.with_name(notes.name + '.partial')
        temporary.write_text(updated, encoding='utf-8', newline='\n')
        temporary.replace(notes)
    if build:
        invoke_build(root or notes.resolve().parents[1])
    return notes


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Insert the measured fold-0 pilot into the working notes.')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--summary', type=Path)
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--notes', type=Path)
    parser.add_argument('--skip-build', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = args.root
    try:
        finalize(
            summary=args.summary or root / 'docs' / 'data_summary.json',
            pilot=args.pilot or root / 'paper' / 'pilot_results.csv',
            validation=args.validation or root / 'submission.validation.json',
            notes=args.notes or root / 'paper' / 'working_notes.tex',
            build=not args.skip_build,
            root=root,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
