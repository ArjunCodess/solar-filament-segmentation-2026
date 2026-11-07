"""Compare one candidate's per-view metrics with a baseline.

Reference PQ sums ``pq_iou_sum``, ``pq_tp``, ``pq_fp``, and ``pq_fn`` with
``aggregate_reference_pq``. Paired temporal-block bootstrap runs only when
every aligned view has a ``temporal_group`` from the metrics or a manifest.
Skeleton coverage and small-object hit rate are gate inputs only when weights
or an explicit aggregate are present. Source files are read only.

This does not train, download, submit, or inspect masks for merges and omitted barbs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import aggregate_reference_pq


PQ_COLUMNS = ('pq_iou_sum', 'pq_tp', 'pq_fp', 'pq_fn')
PQ_GAIN_MIN = 0.005
SKELETON_DECLINE_MAX = 0.01
SMALL_HIT_DECLINE_MAX = 0.02
BLOCK_METADATA = (
    'temporal_group', 'fold', 'file_name', 'sha256', 'date', 'station', 'observing_station',
)
STRATA = (
    'contrast_bin', 'size_bin', 'area_bin', 'radial_bin', 'disk_radius_bin',
    'station', 'observing_station', 'date_bin',
)
IDENTITY_COLUMNS = ('fold', 'temporal_group', 'file_name', 'sha256')
SKELETON_NUMERATORS = ('skeleton_covered', 'skeleton_hits', 'skeleton_recall_sum')
SKELETON_VALUES = ('skeleton_coverage', 'skeleton_recall')
SKELETON_WEIGHTS = ('skeleton_count', 'skeleton_pixels', 'skeleton_instances')
SMALL_NUMERATORS = ('small_hits',)
SMALL_VALUES = ('small_hit_rate', 'small_object_hit_rate')
SMALL_WEIGHTS = ('small_instances', 'small_count')
SUMMARY_SKELETON = ('skeleton_coverage', 'skeleton_recall')
SUMMARY_SMALL = ('small_hit_rate', 'small_object_hit_rate')
VIEW_KEYS = ('views', 'rows', 'records')
GATE_RULE = (
    'PQ gain >= 0.005; skeleton coverage loss <= 0.01; small-object hit-rate loss <= 0.02. '
    'A strictly positive paired temporal-block interval is required before numeric promotion. '
    'Visual review remains required. Missing metrics are not evaluable.'
)
BOOTSTRAP_LIMITATION = (
    'Does not estimate retraining, policy-selection, or hidden-server uncertainty. '
    'Intervals condition on the supplied fixed metrics.'
)
VISUAL_REVIEW = (
    'Required before promotion. This comparison does not inspect systematic merges or omitted barbs.'
)


class ComparisonError(ValueError):
    """Raised when a comparison would invent an alignment or a missing metric."""


def _canonical_atom(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    if not np.isfinite(number):
        return text
    if number == round(number):
        return int(round(number))
    return number


def _slot(value, reason=None, note=None):
    return {
        'status': 'not_evaluable' if value is None else 'evaluable',
        'value': value,
        'reason': reason,
        'note': note,
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not np.isfinite(number):
            raise ComparisonError('Refusing to write a non-finite metric')
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _records(payload, path, keys):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in keys:
            if isinstance(payload.get(key), list):
                return payload[key]
        if payload and all(isinstance(item, dict) for item in payload.values()):
            return [{'image_id': key, **item} for key, item in payload.items()]
    raise ComparisonError(f'{path} must be a list of rows or an object with one of {keys}')


def load_metrics(path):
    """Return per-view rows and any explicit scalar morphology aggregates."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path, dtype={'image_id': str}, encoding='utf-8'), {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise ComparisonError(f'{path} is not valid JSON: {exc}') from exc
    summary = {}
    if isinstance(payload, dict):
        for key in (*SUMMARY_SKELETON, *SUMMARY_SMALL):
            if key in payload and not isinstance(payload[key], (list, dict)):
                summary[key] = payload[key]
    frame = pd.DataFrame(_records(payload, path, VIEW_KEYS))
    if 'image_id' in frame.columns:
        frame['image_id'] = frame['image_id'].map(lambda value: str(value).strip())
    return frame, summary


def load_manifest(path):
    """Load split metadata. Manifest records may identify views with ``id``."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        frame = pd.read_csv(path, dtype={'image_id': str}, encoding='utf-8')
    else:
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise ComparisonError(f'{path} is not valid JSON: {exc}') from exc
        frame = pd.DataFrame(_records(payload, path, VIEW_KEYS))
        if 'image_id' not in frame.columns and 'id' in frame.columns:
            frame = frame.rename(columns={'id': 'image_id'})
    if 'image_id' not in frame.columns:
        raise ComparisonError(f'{path} has no image_id or id column')
    frame = frame.copy()
    frame['image_id'] = frame['image_id'].map(lambda value: str(value).strip())
    if frame['image_id'].duplicated().any():
        raise ComparisonError(f'{path} repeats image_id values')
    keep = ['image_id', *[column for column in (*BLOCK_METADATA, *STRATA) if column in frame.columns]]
    return frame.loc[:, keep]


def _prepare_identity(frame, role, policy):
    if 'image_id' not in frame.columns:
        raise ComparisonError(f'{role} has no image_id column')
    frame = frame.copy()
    frame['image_id'] = frame['image_id'].map(lambda value: str(value).strip())
    if frame['image_id'].eq('').any() or frame['image_id'].isna().any():
        raise ComparisonError(f'{role} has a blank image_id')
    selected = None
    column = next((name for name in ('policy', 'variant') if name in frame.columns), None)
    if column is not None:
        labels = frame[column].map(lambda value: str(value).strip())
        choices = sorted(set(labels))
        if policy is None:
            if len(choices) != 1:
                raise ComparisonError(
                    f'{role} contains policies {choices}; pass --{role}-policy')
            policy = choices[0]
        if str(policy) not in choices:
            raise ComparisonError(f'{role} has no policy {policy}; found {choices}')
        frame = frame.loc[labels.eq(str(policy))].drop(columns=[column]).copy()
        selected = str(policy)
    if frame.empty:
        raise ComparisonError(f'{role} has no annotation views')
    if frame['image_id'].duplicated().any():
        duplicated = sorted(set(frame.loc[frame['image_id'].duplicated(), 'image_id']))
        raise ComparisonError(f'{role} repeats image_id values: {duplicated[:20]}')
    return frame.reset_index(drop=True), selected


def _require_same_views(baseline, candidate):
    left, right = set(baseline['image_id']), set(candidate['image_id'])
    if left != right:
        missing = sorted(left - right)
        extra = sorted(right - left)
        raise ComparisonError(
            'Candidate and baseline annotation views differ. '
            f'Only in baseline ({len(missing)}): {missing[:20]}. '
            f'Only in candidate ({len(extra)}): {extra[:20]}.'
        )


def _require_agreement(baseline, candidate, columns):
    for column in columns:
        if column not in baseline.columns or column not in candidate.columns:
            continue
        merged = baseline[['image_id', column]].merge(
            candidate[['image_id', column]], on='image_id', suffixes=('_baseline', '_candidate'))
        mismatched = [
            str(image_id) for image_id, left, right in merged.itertuples(index=False, name=None)
            if _canonical_atom(left) != _canonical_atom(right)
        ]
        if mismatched:
            raise ComparisonError(f'{column} disagrees for image_id values {mismatched[:20]}')


def _attach_manifest(frame, manifest, role):
    if manifest is None:
        return frame.copy()
    merged = frame.merge(manifest, on='image_id', how='left', suffixes=('', '_manifest'), validate='one_to_one')
    for column in [name for name in manifest.columns if name != 'image_id']:
        supplied_name = f'{column}_manifest'
        if column not in frame.columns:
            continue
        mismatched = [
            str(image_id) for image_id, current, supplied in zip(
                merged['image_id'], merged[column], merged[supplied_name])
            if pd.notna(current) and pd.notna(supplied) and _canonical_atom(current) != _canonical_atom(supplied)
        ]
        if mismatched:
            raise ComparisonError(
                f'{role} {column} disagrees with the manifest for image_id values {mismatched[:20]}')
        merged[column] = [
            current if pd.notna(current) else supplied
            for current, supplied in zip(merged[column], merged[supplied_name])
        ]
        merged = merged.drop(columns=[supplied_name])
    return merged


def _share_metadata(baseline, candidate):
    baseline, candidate = baseline.copy(), candidate.copy()
    for column in BLOCK_METADATA:
        left, right = column in baseline.columns, column in candidate.columns
        if left == right:
            continue
        source, target = (baseline, candidate) if left else (candidate, baseline)
        mapped = source.set_index('image_id')[column]
        target[column] = target['image_id'].map(mapped)
    return baseline, candidate


def _require_pq(frame, role):
    missing = [column for column in PQ_COLUMNS if column not in frame.columns]
    if missing:
        raise ComparisonError(
            f'{role} is missing sufficient statistics {missing}; they will not be imputed')
    frame = frame.copy()
    for column in PQ_COLUMNS:
        converted = pd.to_numeric(frame[column], errors='coerce')
        if converted.isna().any() or not np.isfinite(converted.to_numpy(dtype=float)).all() or (converted < 0).any():
            raise ComparisonError(
                f'{role} column {column} must contain finite values >= 0; it will not be imputed')
        frame[column] = converted
    return frame


def _pooled_pq(frame):
    return float(aggregate_reference_pq(frame.loc[:, PQ_COLUMNS].to_dict('records')))


def _mean_view_pq(frame):
    sufficient = np.array([
        aggregate_reference_pq([row]) for row in frame.loc[:, PQ_COLUMNS].to_dict('records')
    ], dtype=float)
    if 'pq' not in frame.columns:
        return float(sufficient.mean()), True
    stored = pd.to_numeric(frame['pq'], errors='coerce')
    if stored.isna().any() or not np.isfinite(stored.to_numpy(dtype=float)).all():
        raise ComparisonError('Stored per-view pq must be finite; it will not be imputed')
    return float(stored.mean()), bool(np.max(np.abs(stored.to_numpy(dtype=float) - sufficient)) <= 1e-6)


def _explicit_summary(summary, keys):
    present = [key for key in keys if summary and key in summary and summary[key] is not None]
    if not present:
        return None
    try:
        values = [float(summary[key]) for key in present]
    except (TypeError, ValueError) as exc:
        raise ComparisonError(f'Explicit aggregate {present} is not numeric') from exc
    if max(values) - min(values) > 1e-6:
        raise ComparisonError(f'Explicit aggregates {present} disagree')
    if not np.isfinite(values[0]):
        raise ComparisonError(f'Explicit aggregate {present[0]} is non-finite')
    return values[0]


def _pooled_rate(frame, numerators, values, weights):
    columns = set(frame.columns)
    specs = []
    for weight in weights:
        if weight not in columns:
            continue
        for numerator in numerators:
            if numerator in columns:
                specs.append((numerator, weight, False))
        for value in values:
            if value in columns:
                specs.append((value, weight, True))
    if not specs:
        bare = [name for name in (*numerators, *values) if name in columns]
        return {'state': 'unweighted' if bare else 'absent'}
    rates, sources = [], []
    for left_name, weight_name, weighted_value in specs:
        left = pd.to_numeric(frame[left_name], errors='coerce')
        weight = pd.to_numeric(frame[weight_name], errors='coerce')
        if left.isna().any() or weight.isna().any():
            raise ComparisonError(f'{left_name} and {weight_name} must be numeric; they will not be imputed')
        total = float(weight.sum())
        if total == 0:
            return {'state': 'zero', 'weight': weight_name}
        if (weight < 0).any():
            raise ComparisonError(f'{weight_name} contains a negative weight')
        if weighted_value:
            rates.append(float((left * weight).sum()) / total)
            sources.append(f'sum({left_name}*{weight_name})/sum({weight_name})')
        else:
            rates.append(float(left.sum()) / total)
            sources.append(f'sum({left_name})/sum({weight_name})')
    if max(rates) - min(rates) > 1e-6:
        raise ComparisonError('Morphology columns imply different pooled rates: ' + ', '.join(sources))
    return {'state': 'value', 'value': rates[0], 'source': sources[0]}


def _resolve_rate(frame, summary, numerators, values, weights, summary_keys):
    pooled = _pooled_rate(frame, numerators, values, weights)
    explicit = _explicit_summary(summary, summary_keys)
    if pooled['state'] == 'value':
        if explicit is not None and abs(pooled['value'] - explicit) > 1e-6:
            raise ComparisonError(
                f"Explicit aggregate {explicit} disagrees with {pooled['source']} = {pooled['value']}")
        return _slot(pooled['value'], note=pooled['source'])
    if pooled['state'] == 'zero':
        return _slot(None, reason=f"{pooled['weight']} sums to zero; the rate was not imputed")
    if pooled['state'] == 'unweighted':
        if explicit is None:
            return _slot(None, reason=(
                'Per-view rate is present without instance or pixel weights. '
                'An unweighted mean of views was not used as the morphology gate.'
            ))
        return _slot(explicit, note='Using the explicit aggregate; unweighted per-view rates were not pooled')
    if explicit is not None:
        return _slot(explicit, note='explicit aggregate')
    return _slot(None, reason='Metric columns are absent and no explicit aggregate was provided')


def _standard_pq(frame):
    if not {'tp', 'fp', 'fn'} <= set(frame.columns):
        return _slot(None, reason='tp, fp, and fn are required to pool standard one-to-one PQ')
    counts = frame.loc[:, ['tp', 'fp', 'fn']].apply(pd.to_numeric, errors='coerce')
    if counts.isna().any().any():
        return _slot(None, reason='tp, fp, or fn is non-numeric')
    denominator = counts['tp'] + 0.5 * counts['fp'] + 0.5 * counts['fn']
    if 'matched_iou_sum' in frame.columns:
        numerator = pd.to_numeric(frame['matched_iou_sum'], errors='coerce')
        note = 'sum(matched_iou_sum)/sum(tp+0.5*fp+0.5*fn)'
    elif 'standard_pq' in frame.columns:
        stored = pd.to_numeric(frame['standard_pq'], errors='coerce')
        if stored.isna().any():
            return _slot(None, reason='standard_pq is non-numeric')
        numerator = stored * denominator
        note = 'Recovered matched-IoU sums from per-view standard_pq and detection counts'
    else:
        return _slot(None, reason='matched_iou_sum or standard_pq is absent')
    if numerator.isna().any():
        return _slot(None, reason='Matched IoU numerator is non-numeric')
    total = float(denominator.sum())
    if total == 0:
        return _slot(0.0, note='Detection denominator sums to zero')
    return _slot(float(numerator.sum() / total), note=note)


def _qualifying_mean_iou(frame):
    matches = float(frame['pq_tp'].sum())
    if matches == 0:
        return _slot(None, reason='No qualifying pairs')
    return _slot(float(frame['pq_iou_sum'].sum()) / matches, note='sum(pq_iou_sum)/sum(pq_tp)')


def _detection_rate(frame, kind):
    if not {'tp', 'fp', 'fn'} <= set(frame.columns):
        return _slot(None, reason='tp, fp, and fn are absent')
    counts = frame.loc[:, ['tp', 'fp', 'fn']].apply(pd.to_numeric, errors='coerce')
    if counts.isna().any().any():
        return _slot(None, reason='tp, fp, or fn is non-numeric')
    tp, fp, fn = (float(counts[column].sum()) for column in ('tp', 'fp', 'fn'))
    denominator = tp + (fp if kind == 'precision' else fn)
    if denominator == 0:
        return _slot(None, reason=f'{kind} denominator is zero; the rate was not imputed')
    return _slot(tp / denominator, note=f'sum(tp)/sum(tp+{"fp" if kind == "precision" else "fn"})')


def _summed_alias(frame, aliases, label):
    present = [name for name in aliases if name in frame.columns]
    if not present:
        return _slot(None, reason=f'{label} columns are absent')
    totals = []
    for name in present:
        converted = pd.to_numeric(frame[name], errors='coerce')
        if converted.isna().any():
            return _slot(None, reason=f'{name} is non-numeric')
        totals.append(float(converted.sum()))
    if max(totals) - min(totals) > 1e-6:
        raise ComparisonError(f'{label} columns {present} disagree')
    return _slot(totals[0], note=f'sum({present[0]})')


def _physical_images(frame):
    for column in ('file_name', 'sha256'):
        if column not in frame.columns or frame[column].isna().any():
            continue
        identities = frame[column].map(lambda value: str(value).strip())
        if identities.eq('').any():
            continue
        return _slot(int(identities.nunique()), note=f'unique {column}')
    return _slot(None, reason='No file_name or sha256 metadata for every aligned view')


def _summarize_side(frame, summary, name, policy):
    mean_pq, stored_matches = _mean_view_pq(frame)
    return {
        'name': name,
        'policy': policy,
        'annotation_views': int(len(frame)),
        'physical_images': _physical_images(frame),
        'pooled_pq': _pooled_pq(frame),
        'mean_view_pq': mean_pq,
        'stored_view_pq_matches_sufficient_stats': stored_matches,
        'standard_pq': _standard_pq(frame),
        'qualifying_pair_mean_iou': _qualifying_mean_iou(frame),
        'precision': _detection_rate(frame, 'precision'),
        'recall': _detection_rate(frame, 'recall'),
        'splits': _summed_alias(frame, ('splits', 'split_count', 'n_splits'), 'split'),
        'merges': _summed_alias(frame, ('merges', 'merge_count', 'n_merges'), 'merge'),
        'skeleton_coverage': _resolve_rate(
            frame, summary, SKELETON_NUMERATORS, SKELETON_VALUES, SKELETON_WEIGHTS, SUMMARY_SKELETON),
        'small_hit_rate': _resolve_rate(
            frame, summary, SMALL_NUMERATORS, SMALL_VALUES, SMALL_WEIGHTS, SUMMARY_SMALL),
    }


def _block_table(frame, keys):
    working = frame.copy()
    for key in keys:
        working[key] = working[key].map(_canonical_atom)
    return working.groupby(keys, sort=True)[list(PQ_COLUMNS)].sum()


def _pq_sums(sums):
    numerator = float(sums[0])
    denominator = float(sums[1] + 0.5 * (sums[2] + sums[3]))
    return numerator / denominator if denominator else 0.0


def paired_block_deltas(baseline_blocks, candidate_blocks, draws, seed):
    """Paired cluster draws. Fold levels are resampled inside each fold."""
    if not baseline_blocks.index.equals(candidate_blocks.index):
        raise ComparisonError('Temporal blocks differ between baseline and candidate')
    if len(baseline_blocks) == 0:
        raise ComparisonError('No temporal blocks')
    if int(draws) < 1:
        raise ComparisonError('Bootstrap draws must be positive')
    base = baseline_blocks.loc[:, PQ_COLUMNS].to_numpy(dtype=float)
    candidate = candidate_blocks.loc[:, PQ_COLUMNS].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    deltas = np.empty(int(draws), dtype=float)
    fold_values = None
    unique_folds = None
    if isinstance(baseline_blocks.index, pd.MultiIndex) and 'fold' in baseline_blocks.index.names:
        fold_values = baseline_blocks.index.get_level_values('fold').to_numpy()
        unique_folds = sorted(set(fold_values.tolist()))
    for draw in range(int(draws)):
        if fold_values is None:
            selected = rng.choice(len(base), size=len(base), replace=True)
        else:
            selected = np.concatenate([
                rng.choice(np.flatnonzero(fold_values == fold), size=int(np.sum(fold_values == fold)), replace=True)
                for fold in unique_folds
            ])
        deltas[draw] = _pq_sums(candidate[selected].sum(axis=0)) - _pq_sums(base[selected].sum(axis=0))
    return deltas


def _bootstrap_unavailable(reason):
    return {
        'status': 'not_evaluable',
        'reason': reason,
        'draws': None,
        'seed': None,
        'temporal_blocks': None,
        'folds': None,
        'method': None,
        'limitation': BOOTSTRAP_LIMITATION,
        'delta_pooled_pq_95_percentile': None,
        'positive_interval': None,
    }


def _bootstrap(baseline, candidate, draws, seed):
    for role, frame in (('baseline', baseline), ('candidate', candidate)):
        if 'temporal_group' not in frame.columns or frame['temporal_group'].isna().any():
            return _bootstrap_unavailable(
                f'{role} has no temporal_group for every aligned view. '
                'Paired temporal-block bootstrap was not run; views were not treated as independent draws.'
            )
    keys = ['fold', 'temporal_group'] if (
        'fold' in baseline.columns and baseline['fold'].notna().all() and candidate['fold'].notna().all()
    ) else ['temporal_group']
    left, right = _block_table(baseline, keys), _block_table(candidate, keys)
    if not left.index.equals(right.index):
        return _bootstrap_unavailable('Baseline and candidate temporal blocks differ')
    deltas = paired_block_deltas(left, right, draws, seed)
    low, high = (float(value) for value in np.quantile(deltas, [0.025, 0.975]))
    within_fold = len(keys) == 2
    return {
        'status': 'evaluable',
        'reason': None,
        'draws': int(draws),
        'seed': int(seed),
        'temporal_blocks': int(len(left)),
        'folds': sorted(set(left.index.get_level_values('fold').tolist())) if within_fold else [],
        'method': (
            'Paired temporal-group cluster resampling within each fold; annotator views in a group stay together. '
            'The same draw is applied to the baseline and the candidate. Fixed supplied metrics.'
            if within_fold else
            'Paired temporal-group cluster resampling in one pool because fold metadata is absent; '
            'annotator views in a group stay together. The same draw is applied to both series. Fixed supplied metrics.'
        ),
        'limitation': BOOTSTRAP_LIMITATION,
        'delta_pooled_pq_95_percentile': [low, high],
        'positive_interval': bool(low > 0.0),
    }


def _strata(baseline, candidate):
    report = {}
    for column in STRATA:
        if column not in baseline.columns or column not in candidate.columns:
            report[column] = {
                'status': 'not_evaluable',
                'reason': 'Column absent on the aligned per-view rows or manifest',
                'levels': None,
            }
            continue
        pairs = baseline[['image_id', column, *PQ_COLUMNS]].merge(
            candidate[['image_id', column, *PQ_COLUMNS]], on='image_id', suffixes=('_baseline', '_candidate'))
        mismatched = [
            str(image_id) for image_id, left, right in zip(
                pairs['image_id'], pairs[f'{column}_baseline'], pairs[f'{column}_candidate'])
            if _canonical_atom(left) != _canonical_atom(right)
        ]
        if mismatched or pairs[f'{column}_baseline'].isna().any():
            report[column] = {
                'status': 'not_evaluable',
                'reason': 'Stratum values disagree or are missing for some aligned views',
                'levels': None,
            }
            continue
        levels = []
        pairs = pairs.copy()
        pairs['_stratum'] = pairs[f'{column}_baseline'].map(_canonical_atom)
        for key, group in pairs.groupby('_stratum', sort=True):
            base_rows = group.rename(columns={f'{name}_baseline': name for name in PQ_COLUMNS})
            cand_rows = group.rename(columns={f'{name}_candidate': name for name in PQ_COLUMNS})
            base_pq = float(aggregate_reference_pq(base_rows.loc[:, PQ_COLUMNS].to_dict('records')))
            cand_pq = float(aggregate_reference_pq(cand_rows.loc[:, PQ_COLUMNS].to_dict('records')))
            levels.append({
                'level': key,
                'annotation_views': int(len(group)),
                'baseline_pooled_pq': base_pq,
                'candidate_pooled_pq': cand_pq,
                'delta_pooled_pq': cand_pq - base_pq,
            })
        report[column] = {'status': 'evaluable', 'reason': None, 'levels': levels}
    return report


def _decline_gate(baseline_slot, candidate_slot, maximum, comparison):
    if baseline_slot['status'] != 'evaluable' or candidate_slot['status'] != 'evaluable':
        reasons = [slot['reason'] for slot in (baseline_slot, candidate_slot) if slot['reason']]
        return {
            'status': 'not_evaluable',
            'max_decline': maximum,
            'baseline': baseline_slot['value'],
            'candidate': candidate_slot['value'],
            'decline': None,
            'comparison': comparison,
            'reason': '; '.join(dict.fromkeys(reasons)) if reasons else 'metric not present',
        }
    decline = baseline_slot['value'] - candidate_slot['value']
    return {
        'status': 'pass' if decline <= maximum else 'fail',
        'max_decline': maximum,
        'baseline': baseline_slot['value'],
        'candidate': candidate_slot['value'],
        'decline': decline,
        'comparison': comparison,
        'reason': None,
    }


def _decide(gates, bootstrap):
    failed = [name for name, gate in gates.items() if gate['status'] == 'fail']
    missing = [name for name, gate in gates.items() if gate['status'] == 'not_evaluable']
    if failed:
        reason = f"Failed gates: {', '.join(failed)}."
        if missing:
            reason += f" Missing gates were not treated as passes: {', '.join(missing)}."
        reason += ' The preserved baseline remains preferred.'
        return 'reject', reason
    if missing:
        return 'not_evaluable', (
            f"Not evaluable gates: {', '.join(missing)}. "
            'Missing morphology was not imputed, so the candidate is not promoted.'
        )
    if bootstrap['status'] != 'evaluable':
        return 'inconclusive', (
            'Numeric morphology gates pass, but the paired temporal-block interval is not evaluable. '
            'Inconclusive evidence favors the preserved baseline. ' + bootstrap['reason']
        )
    low, high = bootstrap['delta_pooled_pq_95_percentile']
    if not bootstrap['positive_interval']:
        return 'inconclusive', (
            f'Numeric morphology gates pass, but the paired 95% interval [{low}, {high}] is not strictly positive. '
            'Inconclusive evidence favors the preserved baseline.'
        )
    return 'numeric_pass_pending_visual_review', (
        'Numeric gates pass and the paired interval is strictly positive. '
        'Promotion still requires visual review for systematic merges and omitted barbs.'
    )


def compare_tables(baseline, candidate, manifest=None, baseline_summary=None, candidate_summary=None,
                   draws=10000, seed=20261102, baseline_name='baseline', candidate_name='candidate',
                   baseline_policy=None, candidate_policy=None):
    """Compare aligned per-view tables. ``manifest`` supplies temporal blocks when present."""
    if int(draws) < 1:
        raise ComparisonError('Bootstrap draws must be positive')
    baseline, baseline_policy = _prepare_identity(baseline, 'baseline', baseline_policy)
    candidate, candidate_policy = _prepare_identity(candidate, 'candidate', candidate_policy)
    _require_same_views(baseline, candidate)
    _require_agreement(baseline, candidate, IDENTITY_COLUMNS)
    baseline = _attach_manifest(baseline, manifest, 'baseline')
    candidate = _attach_manifest(candidate, manifest, 'candidate')
    _require_agreement(baseline, candidate, IDENTITY_COLUMNS)
    baseline, candidate = _share_metadata(baseline, candidate)
    baseline, candidate = _require_pq(baseline, 'baseline'), _require_pq(candidate, 'candidate')
    base_side = _summarize_side(baseline, baseline_summary or {}, baseline_name, baseline_policy)
    cand_side = _summarize_side(candidate, candidate_summary or {}, candidate_name, candidate_policy)
    if (base_side['physical_images']['status'] == 'evaluable'
            and cand_side['physical_images']['status'] == 'evaluable'
            and base_side['physical_images']['value'] != cand_side['physical_images']['value']):
        raise ComparisonError('Baseline and candidate physical-image counts disagree')
    pooled_delta = cand_side['pooled_pq'] - base_side['pooled_pq']
    gates = {
        'pq_gain': {
            'status': 'pass' if pooled_delta >= PQ_GAIN_MIN else 'fail',
            'threshold': PQ_GAIN_MIN,
            'value': pooled_delta,
            'comparison': 'candidate pooled PQ minus baseline pooled PQ >= 0.005',
            'reason': None,
        },
        'skeleton_coverage': _decline_gate(
            base_side['skeleton_coverage'], cand_side['skeleton_coverage'], SKELETON_DECLINE_MAX,
            'baseline skeleton coverage minus candidate skeleton coverage <= 0.01'),
        'small_hit_rate': _decline_gate(
            base_side['small_hit_rate'], cand_side['small_hit_rate'], SMALL_HIT_DECLINE_MAX,
            'baseline small-object hit rate minus candidate small-object hit rate <= 0.02'),
    }
    bootstrap = _bootstrap(baseline, candidate, int(draws), int(seed))
    decision, decision_reason = _decide(gates, bootstrap)
    return {
        'comparison': 'same-view pooled reference PQ',
        'gate_rule': GATE_RULE,
        'visual_review': VISUAL_REVIEW,
        'annotation_views': base_side['annotation_views'],
        'physical_images': base_side['physical_images'],
        'baseline': base_side,
        'candidate': cand_side,
        'delta': {
            'pooled_pq': pooled_delta,
            'mean_view_pq': cand_side['mean_view_pq'] - base_side['mean_view_pq'],
            'aggregation': (
                'Decision PQ sums pq_iou_sum, pq_tp, pq_fp, and pq_fn before division. '
                'mean_view_pq is the unweighted mean of per-view PQ and is not the decision metric.'
            ),
        },
        'bootstrap': bootstrap,
        'strata': _strata(baseline, candidate),
        'gates': gates,
        'decision': decision,
        'decision_reason': decision_reason,
    }


def compare_paths(baseline, candidate, manifest=None, output=None, **kwargs):
    """Load CSV or JSON metrics, compare them, and optionally write the report."""
    baseline, candidate = Path(baseline), Path(candidate)
    manifest_path = None if manifest is None else Path(manifest)
    output_path = None if output is None else Path(output)
    if output_path is not None:
        sources = {baseline.resolve(), candidate.resolve()}
        if manifest_path is not None:
            sources.add(manifest_path.resolve())
        if output_path.resolve() in sources:
            raise ComparisonError('Refusing to overwrite a source metrics or manifest file')
    base_frame, base_summary = load_metrics(baseline)
    cand_frame, cand_summary = load_metrics(candidate)
    manifest_frame = None if manifest_path is None else load_manifest(manifest_path)
    report = compare_tables(
        base_frame, cand_frame, manifest_frame, base_summary, cand_summary, **kwargs)
    report['sources'] = {
        'baseline': str(baseline),
        'candidate': str(candidate),
        'manifest': None if manifest_path is None else str(manifest_path),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_json_ready(report), indent=2) + '\n', encoding='utf-8')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=(
        'Compare same-view candidate per-view metrics against a baseline using pooled reference PQ, '
        'paired temporal-block bootstrap, and morphology gates.'
    ))
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--manifest', help='folds CSV or manifest JSON; temporal_group enables block bootstrap')
    parser.add_argument('--output', required=True)
    parser.add_argument('--draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--baseline-name')
    parser.add_argument('--candidate-name')
    parser.add_argument('--baseline-policy')
    parser.add_argument('--candidate-policy')
    args = parser.parse_args(argv)
    try:
        report = compare_paths(
            args.baseline, args.candidate, args.manifest, args.output,
            draws=args.draws, seed=args.seed,
            baseline_name=args.baseline_name or Path(args.baseline).stem,
            candidate_name=args.candidate_name or Path(args.candidate).stem,
            baseline_policy=args.baseline_policy, candidate_policy=args.candidate_policy,
        )
    except ComparisonError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(_json_ready(report), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
