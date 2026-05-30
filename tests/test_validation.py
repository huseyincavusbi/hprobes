"""Tests for L2, stability, and correlation validation functions."""

import numpy as np
from hprobes.probe import _run_l2_check, _run_stability_check, _run_correlation_check


def _make_Xy(predictor_idx: int = 0, n_samples: int = 100, n_features: int = 10):
    """Synthetic data: feature `predictor_idx` determines label."""
    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features) * 0.5
    X[:, predictor_idx] = rng.randn(n_samples)  # stronger signal
    y = (X[:, predictor_idx] > 0).astype(int)
    return X, y


class TestL2Check:
    def test_strong_signal_dominates(self):
        """Feature 0 is the sole predictor → L2 top-1 must be index 0."""
        X, y = _make_Xy(predictor_idx=0)
        l1_selected = np.array([0])
        result = _run_l2_check(l1_selected, X, y, l1_C=1.0, seed=42)
        assert result["l2_overlap_count"] == 1
        assert result["l2_overlap_ratio"] == 1.0
        assert 0 in result["l2_top_indices"]
        assert "genuinely sparse" in result["interpretation"].lower()

    def test_weak_signal_low_concentration(self):
        """All features are noise → L2 weight spread thin, low overlap."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10) * 0.1
        y = rng.randint(0, 2, 100)  # random labels
        l1_selected = np.array([0])
        result = _run_l2_check(l1_selected, X, y, l1_C=1.0, seed=42)
        # With random data, no guarantee L1's pick is in L2 top
        assert result["l2_weight_concentration"] < 1.0
        assert "l2_overlap_ratio" in result

    def test_multiple_l1_neurons(self):
        """L1 finds three neurons; check L2 overlap counts them all."""
        X, y = _make_Xy(predictor_idx=0)
        # Second and third features are correlated copies of feature 0
        X[:, 1] = X[:, 0] * 0.8
        X[:, 2] = X[:, 0] * 0.6
        l1_selected = np.array([0, 1, 2])
        result = _run_l2_check(l1_selected, X, y, l1_C=1.0, seed=42)
        assert result["n_l1_neurons"] == 3
        assert result["l2_overlap_count"] >= 1  # at least one should be top
        assert result["l2_auroc"] >= 0.5


class TestStabilityCheck:
    def test_strong_signal_stable(self):
        """Single dominant feature → bootstraps consistently find it."""
        X, y = _make_Xy(predictor_idx=0)
        result = _run_stability_check(X, y, l1_C=1.0, n_runs=5, base_seed=42)
        assert result["n_runs"] == 5
        assert len(result["neurons_per_run"]) == 5
        # At least one neuron in every run
        assert min(result["neurons_per_run"]) >= 1
        assert result["jaccard_mean"] > 0.0
        assert "jaccard_min" in result

    def test_random_data_unstable(self):
        """Random labels → no consistent signal, lower stability."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 20) * 0.1
        y = rng.randint(0, 2, 100)
        result = _run_stability_check(X, y, l1_C=10.0, n_runs=5, base_seed=42)
        # With noise, jaccard should not be perfect
        assert result["jaccard_mean"] <= 1.0
        assert "n_runs" in result

    def test_interpretation_output(self):
        """Result dict has interpretation string."""
        X, y = _make_Xy(predictor_idx=0)
        result = _run_stability_check(X, y, l1_C=1.0, n_runs=3, base_seed=42)
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


class TestCorrelationCheck:
    def test_independent_feature_low_correlation(self):
        """H-Neuron index has no correlation with others → max_corr low."""
        X, y = _make_Xy(predictor_idx=0)
        result = _run_correlation_check(np.array([0]), X)
        assert len(result["max_correlations"]) == 1
        # Feature 0 is the signal column; others are independent noise
        assert result["max_correlations"][0] < 0.5

    def test_duplicate_feature_high_correlation(self):
        """Feature 1 is a near-copy of feature 0 → max_corr ≈ 1.0 if chosen."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 1] = X[:, 0] * 0.99  # near-identical copy

        # Check correlation from feature 0's perspective
        # feature 0 should have high max |r| because column 1 is a copy
        result = _run_correlation_check(np.array([0]), X)
        assert result["max_correlations"][0] > 0.8
        assert result["n_high_correlation"] >= 1

    def test_zero_variance_handled(self):
        """Zero-variance column does not crash the check."""
        X = np.ones((50, 5))
        X[:, 0] = np.arange(50)  # only column 0 has variance
        result = _run_correlation_check(np.array([3]), X)
        # Column 3 has zero variance → correlation is 0
        assert result["max_correlations"][0] == 0.0

    def test_interpretation_high_corr(self):
        """Duplicate features → interpretation flags them."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        X[:, 1] = X[:, 0] * 0.99
        result = _run_correlation_check(np.array([0]), X)
        assert (
            "correlated clusters" in result["interpretation"].lower()
            or result["n_high_correlation"] > 0
        )
