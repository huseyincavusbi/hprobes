"""Tests for data-parallel CETT extraction (issue #25)."""

import functools

import numpy as np
import pytest
import torch

from hprobes import HProbes
from hprobes.parallel import _ShardResult, _shard_indices, merge_shards
from hprobes.rows import CETTRows

from _parallel_helpers import TinyModel, TinyTokenizer, make_tiny_model


def _samples():
    samples = [
        {
            "question": f"Q{i}?",
            "options": ["alpha", "beta", "gamma", "delta"],
            "answer": i % 2,
        }
        for i in range(20)
    ]
    samples.insert(7, {"question": "Qbad?", "options": ["alpha", "beta"], "answer": None})
    return samples


def _labeled_samples():
    samples = _samples()
    for i, sample in enumerate(samples):
        sample["label"] = i % 2
    return samples


def _label_fn(pred, gt, sample):
    return sample["label"]


@pytest.fixture(scope="module")
def state_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("parallel") / "model.pt"
    torch.manual_seed(0)
    torch.save(TinyModel().state_dict(), path)
    return str(path)


def _model_from_state(state_path):
    model = TinyModel()
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    return model


def test_shard_indices_are_deterministic():
    assert _shard_indices(5, 2, 0) == [0, 2, 4]
    assert _shard_indices(5, 2, 1) == [1, 3]
    assert _shard_indices(3, 4, 2) == [2]
    assert _shard_indices(3, 4, 3) == []


def test_parallel_extraction_matches_serial(state_path):
    probe = HProbes(_model_from_state(state_path), TinyTokenizer(), l1_C=10, readout="logits")
    serial = probe.extract_features(_samples())

    factory = functools.partial(make_tiny_model, state_path)
    parallel = probe.extract_features_parallel(
        _samples(), factory, world_size=2, devices=["cpu", "cpu"]
    )

    assert np.array_equal(serial.rows, parallel.rows)
    assert np.array_equal(serial.labels, parallel.labels)
    assert serial.row_to_sample == parallel.row_to_sample
    assert serial.prompts == parallel.prompts
    assert serial.ground_truth == parallel.ground_truth
    assert serial.per_sample == parallel.per_sample
    assert serial.samples == parallel.samples
    assert serial.welford_n == parallel.welford_n
    assert np.array_equal(serial.welford_mean, parallel.welford_mean)
    assert np.array_equal(serial.welford_M2, parallel.welford_M2)
    assert serial.meta == parallel.meta


def test_parallel_fit_matches_serial(state_path):
    samples = _labeled_samples()
    serial = HProbes(_model_from_state(state_path), TinyTokenizer(), l1_C=10, readout="logits")
    serial.fit(samples, label_fn=_label_fn)

    factory = functools.partial(make_tiny_model, state_path)
    parallel = HProbes(_model_from_state(state_path), TinyTokenizer(), l1_C=10, readout="logits")
    parallel.fit_parallel(
        samples, factory, world_size=2, devices=["cpu", "cpu"], label_fn=_label_fn
    )

    assert parallel.is_fitted_
    assert parallel.h_neurons_ == serial.h_neurons_
    assert parallel.accuracy_ == serial.accuracy_
    assert np.array_equal(parallel._clf.coef_, serial._clf.coef_)
    assert np.array_equal(parallel._clf.intercept_, serial._clf.intercept_)


@pytest.mark.skipif(torch.cuda.is_available(), reason="auto uses CUDA when available")
def test_auto_world_size_on_cpu_is_serial(state_path):
    probe = HProbes(_model_from_state(state_path), TinyTokenizer(), l1_C=10, readout="logits")
    serial = probe.extract_features(_samples())

    factory = functools.partial(make_tiny_model, state_path)
    parallel = probe.extract_features_parallel(_samples(), factory)

    assert np.array_equal(serial.rows, parallel.rows)
    assert serial.row_to_sample == parallel.row_to_sample
    assert serial.welford_n == parallel.welford_n


def test_parallel_fit_with_config_only_parent(state_path):
    samples = _labeled_samples()
    serial = HProbes(_model_from_state(state_path), TinyTokenizer(), l1_C=10, readout="logits")
    serial.fit(samples, label_fn=_label_fn)

    factory = functools.partial(make_tiny_model, state_path)
    parallel = HProbes(None, None, l1_C=10, readout="logits")
    parallel.fit_parallel(
        samples, factory, world_size=2, devices=["cpu", "cpu"], label_fn=_label_fn
    )

    assert parallel.is_fitted_
    assert parallel.h_neurons_ == serial.h_neurons_


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CUDA-free host")
def test_parallel_requires_explicit_cpu_devices(state_path):
    probe = HProbes(_model_from_state(state_path), TinyTokenizer(), l1_C=10, readout="logits")
    factory = functools.partial(make_tiny_model, state_path)

    with pytest.raises(ValueError, match="Pass devices"):
        probe.extract_features_parallel(_samples(), factory, world_size=2)


def test_parallel_serial_requires_loaded_parent(state_path):
    probe = HProbes(None, None, l1_C=10, readout="logits")
    factory = functools.partial(make_tiny_model, state_path)

    with pytest.raises(ValueError, match="loaded parent model"):
        probe.extract_features_parallel(_samples(), factory, world_size=1)


def test_parallel_handles_empty_shards(state_path):
    samples = _samples()[:3]
    probe = HProbes(_model_from_state(state_path), TinyTokenizer(), l1_C=10, readout="logits")
    serial = probe.extract_features(samples)

    factory = functools.partial(make_tiny_model, state_path)
    parallel = probe.extract_features_parallel(samples, factory, world_size=4, devices=["cpu"] * 4)

    assert np.array_equal(serial.rows, parallel.rows)
    assert serial.row_to_sample == parallel.row_to_sample
    assert serial.welford_n == parallel.welford_n


def test_parallel_rejects_batch_size_gt_one(state_path):
    probe = HProbes(_model_from_state(state_path), TinyTokenizer(), batch_size=2, readout="logits")
    factory = functools.partial(make_tiny_model, state_path)

    with pytest.raises(ValueError, match="batch_size=1"):
        probe.fit_parallel(_samples(), factory, world_size=2, devices=["cpu", "cpu"])


def test_parallel_rejects_resident_cuda_parent(state_path):
    model = torch.nn.Module()
    model.hf_device_map = {"": "cuda:0"}
    probe = HProbes(model, TinyTokenizer(), readout="logits")
    factory = functools.partial(make_tiny_model, state_path)

    with pytest.raises(RuntimeError, match="config-only"):
        probe.fit_parallel(_samples(), factory, world_size=2, devices=["cpu", "cpu"])


def _artifact(meta):
    return CETTRows(
        rows=np.zeros((1, 2), dtype=np.float32),
        labels=np.array([0]),
        row_to_sample=[0],
        prompts=["p"],
        ground_truth=["A"],
        per_sample=[{"is_correct": True}],
        samples=[{"question": "q"}],
        welford_n=1,
        welford_mean=np.zeros(2),
        welford_M2=np.zeros(2),
        meta=meta,
    )


def test_merge_rejects_meta_mismatch():
    results = [
        _ShardResult(0, [0], _artifact({"config": {"readout": "logits"}})),
        _ShardResult(1, [1], _artifact({"config": {"readout": "generate"}})),
    ]

    with pytest.raises(ValueError, match="metadata"):
        merge_shards(results)
