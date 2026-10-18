"""Tests for the CETT rows artifact (issue #26)."""

import numpy as np
import pytest
import torch
from torch import nn

from hprobes.rows import CETTRows, fingerprint_model, rows_key


def _toy_rows(n_rows=8, n_feat=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n_rows, n_feat))
    return CETTRows(
        rows=rows,
        labels=np.arange(n_rows) % 2,
        row_to_sample=list(range(n_rows)),
        prompts=[f"p{i}" for i in range(n_rows)],
        ground_truth=["A"] * n_rows,
        per_sample=[{"is_correct": bool(i % 2)} for i in range(n_rows)],
        samples=[{"question": f"q{i}", "answer": "A"} for i in range(n_rows)],
        welford_n=n_rows,
        welford_mean=rows.mean(axis=0),
        welford_M2=rows.var(axis=0, ddof=1) * (n_rows - 1),
        meta={"layers": [0]},
    )


def test_validate_accepts_consistent_artifact():
    _toy_rows().validate()


def test_validate_rejects_mismatched_labels():
    art = _toy_rows()
    art.labels = art.labels[:-1]
    with pytest.raises(ValueError, match="labels"):
        art.validate()


def test_validate_rejects_mismatched_welford():
    art = _toy_rows()
    art.welford_M2 = art.welford_M2[:-1]
    with pytest.raises(ValueError, match="welford"):
        art.validate()


def test_variance_matches_numpy_sample_variance():
    art = _toy_rows()
    np.testing.assert_allclose(art.variance(), art.rows.var(axis=0, ddof=1), rtol=1e-9)


def test_fingerprint_catches_weight_change():
    m = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    before = fingerprint_model(m, [0, 1])
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    assert fingerprint_model(m, [0, 1]) != before


def test_fingerprint_stable_and_layer_sensitive():
    m = nn.Linear(4, 4)
    assert fingerprint_model(m, [0]) == fingerprint_model(m, [0])
    assert fingerprint_model(m, [0]) != fingerprint_model(m, [1])


def test_fingerprint_handles_bf16_params():
    """Low-precision params (real models) must fingerprint without fp32 copies."""
    m = nn.Linear(8, 4).to(torch.bfloat16)
    a = fingerprint_model(m, [0])
    assert a == fingerprint_model(m, [0])
    with torch.no_grad():
        next(m.parameters()).add_(1.0)
    assert fingerprint_model(m, [0]) != a


def test_rows_key_stable_and_sensitive():
    samples = [{"question": "q", "answer": "A"}]
    cfg = {"layers": [0], "batch_size": 1}
    k = rows_key("fp", samples, cfg)
    assert rows_key("fp", samples, cfg) == k
    assert rows_key("fp2", samples, cfg) != k
    assert rows_key("fp", [{"question": "q2", "answer": "A"}], cfg) != k
    assert rows_key("fp", samples, {"layers": [1], "batch_size": 1}) != k


def test_save_load_round_trip(tmp_path):
    from hprobes.rows import load_rows, save_rows

    art = _toy_rows()
    paths = save_rows(art, str(tmp_path / "rows"))
    assert len(paths) == 2
    back = load_rows(str(tmp_path / "rows"))
    np.testing.assert_array_equal(back.rows, art.rows)
    np.testing.assert_array_equal(back.labels, art.labels)
    assert back.row_to_sample == art.row_to_sample
    assert back.prompts == art.prompts
    assert back.samples == art.samples
    assert back.meta == art.meta
    np.testing.assert_allclose(back.variance(), art.variance(), rtol=1e-9)
