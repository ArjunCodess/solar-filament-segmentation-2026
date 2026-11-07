"""Same-view candidate comparison: alignment, pooled PQ, bootstrap, and gates."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('compare_candidates', ROOT / 'scripts' / 'compare_candidates.py')
compare_candidates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_candidates)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding='utf-8')


def snapshot(paths):
    return {path: path.read_bytes() for path in paths}


def assert_unchanged(paths, before):
    for path, data in before.items():
        assert path.read_bytes() == data


def view(image_id, iou_sum, tp, fp=0, fn=0, pq=None, **extra):
    row = {
        'image_id': image_id,
        'pq_iou_sum': iou_sum,
        'pq_tp': tp,
        'pq_fp': fp,
        'pq_fn': fn,
    }
    if pq is not None:
        row['pq'] = pq
    row.update(extra)
    return row


def test_unequal_view_sets_are_rejected(tmp_path, capsys):
    baseline = tmp_path / 'baseline.csv'
    candidate = tmp_path / 'candidate.json'
    output = tmp_path / 'report.json'
    write_csv(baseline, [view('view-a', 1, 1), view('view-b', 1, 1)])
    write_json(candidate, [view('view-a', 1, 1), view('view-c', 1, 1)])
    before = snapshot((baseline, candidate))

    assert compare_candidates.main([
        '--baseline', str(baseline), '--candidate', str(candidate), '--output', str(output),
    ]) == 2
    error = capsys.readouterr().err
    assert 'view-b' in error and 'view-c' in error
    assert not output.exists()
    assert_unchanged((baseline, candidate), before)

    duplicated = tmp_path / 'duplicated.csv'
    write_csv(duplicated, [view('view-a', 1, 1), view('view-a', 2, 2)])
    before = snapshot((baseline, duplicated))
    assert compare_candidates.main([
        '--baseline', str(baseline), '--candidate', str(duplicated), '--output', str(output),
    ]) == 2
    assert 'repeats image_id' in capsys.readouterr().err
    assert not output.exists()
    assert_unchanged((baseline, duplicated), before)

    folded_base = tmp_path / 'folded_base.csv'
    folded_cand = tmp_path / 'folded_cand.csv'
    write_csv(folded_base, [view('view-a', 1, 1, fold=1), view('view-b', 1, 1, fold=1)])
    write_csv(folded_cand, [view('view-a', 1, 1, fold=2), view('view-b', 1, 1, fold=1)])
    before = snapshot((folded_base, folded_cand))
    assert compare_candidates.main([
        '--baseline', str(folded_base), '--candidate', str(folded_cand), '--output', str(output),
    ]) == 2
    assert 'fold disagrees' in capsys.readouterr().err
    assert_unchanged((folded_base, folded_cand), before)


def test_pooled_pq_is_not_the_mean_of_view_pq(tmp_path):
    baseline = tmp_path / 'baseline.csv'
    candidate = tmp_path / 'candidate.csv'
    output = tmp_path / 'report.json'
    morphology = {'skeleton_recall': 0.8, 'skeleton_count': 10, 'small_hits': 4, 'small_instances': 10}
    write_csv(baseline, [
        view('view-a', 9, 10, pq=0.9, **morphology),
        view('view-b', 0, 0, fn=2, pq=0.0, **morphology),
    ])
    write_csv(candidate, [
        view('view-a', 2, 10, pq=0.2, **morphology),
        view('view-b', 1, 1, pq=1.0, **morphology),
    ])
    before = snapshot((baseline, candidate))

    assert compare_candidates.main([
        '--baseline', str(baseline), '--candidate', str(candidate), '--output', str(output),
        '--draws', '20',
    ]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['baseline']['pooled_pq'] == pytest.approx(9 / 11)
    assert report['baseline']['mean_view_pq'] == pytest.approx(0.45)
    assert report['candidate']['pooled_pq'] == pytest.approx(3 / 11)
    assert report['candidate']['mean_view_pq'] == pytest.approx(0.6)
    assert report['delta']['pooled_pq'] == pytest.approx(-6 / 11)
    assert report['delta']['mean_view_pq'] == pytest.approx(0.15)
    assert report['gates']['pq_gain']['value'] == pytest.approx(report['delta']['pooled_pq'])
    assert report['gates']['pq_gain']['status'] == 'fail'
    assert report['decision'] == 'reject'
    assert_unchanged((baseline, candidate), before)


def test_paired_temporal_block_bootstrap_uses_manifest_groups(tmp_path):
    index = pd.MultiIndex.from_tuples([(1, 1), (1, 2)], names=['fold', 'temporal_group'])
    columns = ['pq_iou_sum', 'pq_tp', 'pq_fp', 'pq_fn']
    baseline_blocks = pd.DataFrame([[0, 0, 0, 10], [5, 5, 0, 0]], index=index, columns=columns)
    candidate_blocks = pd.DataFrame([[5, 5, 0, 0], [0, 0, 0, 10]], index=index, columns=columns)
    deltas = compare_candidates.paired_block_deltas(baseline_blocks, candidate_blocks, draws=400, seed=7)
    # Each paired draw copies a block into both series. Unpaired draws can land on +/- 0.5.
    assert {float(value) for value in np.round(deltas, 10)} == {-1.0, 0.0, 1.0}

    baseline = tmp_path / 'baseline.csv'
    candidate = tmp_path / 'candidate.csv'
    manifest = tmp_path / 'folds.csv'
    output = tmp_path / 'report.json'
    write_csv(baseline, [
        view('a1', 0, 0, fn=10, skeleton_recall=1.0, skeleton_count=1, small_hits=0, small_instances=1),
        view('a2', 0, 0, fn=10, skeleton_recall=1.0, skeleton_count=1, small_hits=0, small_instances=1),
        view('b1', 10, 10, skeleton_recall=0.5, skeleton_count=8, small_hits=9, small_instances=9),
    ])
    write_csv(candidate, [
        view('a1', 5, 5, skeleton_recall=1.0, skeleton_count=1, small_hits=0, small_instances=1),
        view('a2', 5, 5, skeleton_recall=1.0, skeleton_count=1, small_hits=0, small_instances=1),
        view('b1', 5, 5, skeleton_recall=0.5, skeleton_count=8, small_hits=9, small_instances=9),
    ])
    write_csv(manifest, [
        {'image_id': 'a1', 'file_name': 'sun-a.jpeg', 'fold': 1, 'temporal_group': 7},
        {'image_id': 'a2', 'file_name': 'sun-a.jpeg', 'fold': 1, 'temporal_group': 7},
        {'image_id': 'b1', 'file_name': 'sun-b.jpeg', 'fold': 2, 'temporal_group': 8},
    ])
    before = snapshot((baseline, candidate, manifest))
    report = compare_candidates.compare_paths(
        baseline, candidate, manifest, output, draws=30, seed=20261102,
        baseline_name='baseline', candidate_name='candidate',
    )
    assert report['annotation_views'] == 3
    assert report['physical_images']['value'] == 2
    assert report['bootstrap']['status'] == 'evaluable'
    assert report['bootstrap']['temporal_blocks'] == 2
    assert report['bootstrap']['folds'] == [1, 2]
    assert report['bootstrap']['delta_pooled_pq_95_percentile'] == pytest.approx([0.5, 0.5])
    assert report['bootstrap']['positive_interval'] is True
    assert 'retraining' in report['bootstrap']['limitation']
    assert report['baseline']['skeleton_coverage']['value'] == pytest.approx(0.6)
    assert report['baseline']['small_hit_rate']['value'] == pytest.approx(9 / 11)
    assert report['gates']['pq_gain']['status'] == 'pass'
    assert report['gates']['skeleton_coverage']['status'] == 'pass'
    assert report['gates']['small_hit_rate']['status'] == 'pass'
    assert report['decision'] == 'numeric_pass_pending_visual_review'
    assert 'visual review' in report['decision_reason'].lower()

    without_manifest = compare_candidates.compare_paths(
        baseline, candidate, None, draws=30, seed=1,
        baseline_name='baseline', candidate_name='candidate',
    )
    assert without_manifest['bootstrap']['status'] == 'not_evaluable'
    assert 'independent' in without_manifest['bootstrap']['reason']
    assert without_manifest['physical_images']['status'] == 'not_evaluable'
    assert without_manifest['decision'] == 'inconclusive'
    assert_unchanged((baseline, candidate, manifest), before)


def test_missing_morphology_gates_are_not_evaluable(tmp_path):
    baseline = tmp_path / 'baseline.json'
    candidate = tmp_path / 'candidate.json'
    output = tmp_path / 'report.json'
    write_json(baseline, {'views': [view('view-a', 1, 2)]})
    write_json(candidate, {'views': [view('view-a', 2, 2)]})
    before = snapshot((baseline, candidate))
    report = compare_candidates.compare_paths(baseline, candidate, output=output, draws=10, seed=1)
    assert report['gates']['pq_gain']['status'] == 'pass'
    assert report['gates']['pq_gain']['value'] == pytest.approx(0.5)
    for name in ('skeleton_coverage', 'small_hit_rate'):
        assert report['gates'][name]['status'] == 'not_evaluable'
        assert report['gates'][name]['decline'] is None
        assert report['baseline'][name]['value'] is None
        assert report['candidate'][name]['value'] is None
    assert report['decision'] == 'not_evaluable'
    assert_unchanged((baseline, candidate), before)

    unweighted_base = tmp_path / 'unweighted_base.csv'
    unweighted_cand = tmp_path / 'unweighted_cand.csv'
    write_csv(unweighted_base, [view('view-a', 1, 2, skeleton_recall=1.0, small_hit_rate=1.0)])
    write_csv(unweighted_cand, [view('view-a', 2, 2, skeleton_recall=0.0, small_hit_rate=0.0)])
    unweighted = compare_candidates.compare_paths(
        unweighted_base, unweighted_cand, draws=10, seed=1)
    assert unweighted['gates']['skeleton_coverage']['status'] == 'not_evaluable'
    assert 'weight' in unweighted['gates']['skeleton_coverage']['reason']
    assert unweighted['gates']['small_hit_rate']['status'] == 'not_evaluable'
    assert unweighted['candidate']['skeleton_coverage']['value'] is None
    assert unweighted['decision'] == 'not_evaluable'

    explicit_base = tmp_path / 'explicit_base.csv'
    explicit_cand = tmp_path / 'explicit_cand.json'
    write_csv(explicit_base, [view('view-a', 1, 2, skeleton_recall=0.8, skeleton_count=10,
                                   small_hits=4, small_instances=10)])
    write_json(explicit_cand, {
        'skeleton_coverage': 0.70,
        'small_hit_rate': 0.20,
        'views': [view('view-a', 2, 2)],
    })
    explicit_before = snapshot((explicit_base, explicit_cand))
    explicit = compare_candidates.compare_paths(explicit_base, explicit_cand, draws=10, seed=1)
    assert explicit['baseline']['skeleton_coverage']['value'] == pytest.approx(0.8)
    assert explicit['candidate']['skeleton_coverage']['value'] == pytest.approx(0.70)
    assert explicit['candidate']['small_hit_rate']['value'] == pytest.approx(0.20)
    assert explicit['gates']['skeleton_coverage']['status'] == 'fail'
    assert explicit['gates']['skeleton_coverage']['decline'] == pytest.approx(0.10)
    assert explicit['gates']['small_hit_rate']['status'] == 'fail'
    assert explicit['gates']['small_hit_rate']['decline'] == pytest.approx(0.20)
    assert explicit['decision'] == 'reject'
    assert_unchanged((explicit_base, explicit_cand), explicit_before)
