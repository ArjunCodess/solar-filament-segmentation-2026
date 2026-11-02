"""Execute only pure scoring functions extracted from the pinned public notebook."""
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from src.metrics import instance_scores, aggregate_reference_pq


def reference():
    source = Path(__file__).resolve().parents[1]/'reference/self_evaluation_source.py'
    parsed = ast.parse(source.read_text(encoding='utf-8'))
    wanted = {'get_overlap_matrices', 'fp_count_hit', 'fn_count_hit', 'get_pq_score'}
    module = ast.Module(body=[n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in wanted], type_ignores=[])
    namespace = {'torch':torch, 'pd':pd}
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace


def test_reference_parity_overlaps_empty_and_micro_aggregation():
    ref = reference()
    rng = np.random.default_rng(12)
    cases = []
    truth = np.zeros((24, 24), bool)
    truth[5:15, 5:15] = True
    cases.extend([([truth], [truth]), ([truth], [truth, truth]), ([], []), ([truth], []), ([], [truth])])
    for _ in range(20):
        g = [rng.random((24, 24)) > .4 for _ in range(rng.integers(1, 4))]
        p = [x.copy() for x in g]+[rng.random((24, 24)) > .5]
        cases.append((g, p))
    rows, ours = [], []
    for g, p in cases:
        gt = torch.tensor(np.stack(g) if g else np.zeros((0, 24, 24)), dtype=torch.float32)
        pred = torch.tensor(np.stack(p) if p else np.zeros((0, 24, 24)), dtype=torch.float32)
        iou, dice = ref['get_overlap_matrices'](gt, pred)
        row = {'iou_matrix':iou, 'n_gt':len(g), 'n_pred':len(p)}
        rows.append(row)
        ours.append(instance_scores(g, p))
        assert ours[-1]['pq'] == pytest.approx(ref['get_pq_score'](pd.DataFrame([row])), abs=1e-6)
    assert aggregate_reference_pq(ours) == pytest.approx(ref['get_pq_score'](pd.DataFrame(rows)), abs=1e-6)
