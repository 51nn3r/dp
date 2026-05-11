from types import SimpleNamespace

import numpy as np
import pytest

from shapley_dreamer.shapley.base import (
    ShapleyEstimator,
    _REGISTRY,
    get_estimator,
    register,
    registered_estimators,
)
from shapley_dreamer.shapley.exact import ExactShapley
from shapley_dreamer.shapley.factory import build_shapley
from shapley_dreamer.shapley.permutation_mc import PermutationMC


def _settings(**overrides):
    base = dict(SHAPLEY_ALGORITHM="perm_mc", SHAPLEY_M=32, RANDOM_SEED=0)
    base.update(overrides)
    return SimpleNamespace(**base)


# --- registry hygiene -------------------------------------------------------

def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        ShapleyEstimator()  # type: ignore[abstract]


def test_register_rejects_non_subclass():
    with pytest.raises(TypeError):
        @register("not_an_estimator")
        class _Bogus:  # type: ignore[misc]
            pass


def test_register_rejects_duplicates():
    class _Dummy(ShapleyEstimator):
        def estimate(self, coalition_value, n_players):
            return np.zeros(n_players)

        @classmethod
        def from_settings(cls, settings):
            return cls()

    register("dup_test_shapley")(_Dummy)
    try:
        with pytest.raises(ValueError):
            register("dup_test_shapley")(_Dummy)
    finally:
        _REGISTRY.pop("dup_test_shapley", None)


def test_get_estimator_unknown_raises():
    with pytest.raises(KeyError):
        get_estimator("definitely_not_registered")


def test_registered_estimators_contains_builtins():
    names = registered_estimators()
    assert "perm_mc" in names
    assert "exact" in names


# --- factory ---------------------------------------------------------------

def test_factory_builds_perm_mc():
    est = build_shapley(_settings(SHAPLEY_ALGORITHM="perm_mc"))
    assert isinstance(est, PermutationMC)


def test_factory_builds_exact():
    est = build_shapley(_settings(SHAPLEY_ALGORITHM="exact"))
    assert isinstance(est, ExactShapley)


def test_factory_rejects_unknown_algorithm():
    with pytest.raises(ValueError):
        build_shapley(_settings(SHAPLEY_ALGORITHM="kernel_shap"))


# --- exact: axioms ---------------------------------------------------------

def _additive_v(weights):
    weights = np.asarray(weights, dtype=float)

    def v(mask):
        return float(weights[mask].sum())

    return v


def test_exact_additive_recovers_weights():
    weights = np.array([1.0, -2.0, 3.5, 0.25])
    phi = ExactShapley().estimate(_additive_v(weights), n_players=len(weights))
    np.testing.assert_allclose(phi, weights, atol=1e-12)


def test_exact_efficiency():
    rng = np.random.default_rng(0)
    n = 5
    table = rng.normal(size=1 << n)

    def v(mask):
        idx = int(np.packbits(np.pad(mask, (0, 8 - n)), bitorder="little")[0])
        return float(table[idx])

    phi = ExactShapley().estimate(v, n_players=n)
    full = np.ones(n, dtype=bool)
    empty = np.zeros(n, dtype=bool)
    np.testing.assert_allclose(phi.sum(), v(full) - v(empty), atol=1e-12)


def test_exact_symmetry():
    def v(mask):
        return float(mask[0]) + float(mask[1]) + 2.0 * float(mask[0] and mask[1])

    phi = ExactShapley().estimate(v, n_players=2)
    np.testing.assert_allclose(phi[0], phi[1], atol=1e-12)


def test_exact_dummy_gets_zero():
    def v(mask):
        return float(mask[0]) + 2.0 * float(mask[1])

    phi = ExactShapley().estimate(v, n_players=3)
    assert abs(phi[2]) < 1e-12


# --- permutation MC --------------------------------------------------------

def test_perm_mc_rejects_zero_samples():
    with pytest.raises(ValueError):
        PermutationMC(n_samples=0)


def test_perm_mc_seed_determinism():
    weights = np.array([0.3, -1.2, 0.7, 2.1])
    v = _additive_v(weights)
    a = PermutationMC(n_samples=64, seed=123).estimate(v, n_players=4)
    b = PermutationMC(n_samples=64, seed=123).estimate(v, n_players=4)
    np.testing.assert_array_equal(a, b)


def test_perm_mc_different_seeds_differ():
    weights = np.array([0.3, -1.2, 0.7, 2.1])
    v = _additive_v(weights)
    a = PermutationMC(n_samples=8, seed=1).estimate(v, n_players=4)
    b = PermutationMC(n_samples=8, seed=2).estimate(v, n_players=4)
    assert not np.array_equal(a, b)


def test_perm_mc_additive_is_exact_per_permutation():
    # For an additive v(S) = sum w_i, every permutation gives phi_i = w_i
    # exactly, so MC matches the closed form regardless of n_samples.
    weights = np.array([1.0, -2.0, 3.5, 0.25])
    phi = PermutationMC(n_samples=4, seed=0).estimate(
        _additive_v(weights), n_players=len(weights)
    )
    np.testing.assert_allclose(phi, weights, atol=1e-12)


def test_perm_mc_converges_to_exact():
    rng = np.random.default_rng(7)
    n = 4
    table = rng.normal(size=1 << n)

    def v(mask):
        idx = int(np.packbits(np.pad(mask, (0, 8 - n)), bitorder="little")[0])
        return float(table[idx])

    phi_exact = ExactShapley().estimate(v, n_players=n)
    phi_mc = PermutationMC(n_samples=4000, seed=0).estimate(v, n_players=n)
    np.testing.assert_allclose(phi_mc, phi_exact, atol=0.05)


def test_perm_mc_efficiency_in_expectation():
    rng = np.random.default_rng(3)
    n = 4
    table = rng.normal(size=1 << n)

    def v(mask):
        idx = int(np.packbits(np.pad(mask, (0, 8 - n)), bitorder="little")[0])
        return float(table[idx])

    phi_mc = PermutationMC(n_samples=2000, seed=0).estimate(v, n_players=n)
    full = np.ones(n, dtype=bool)
    empty = np.zeros(n, dtype=bool)
    # Per-permutation efficiency is exact; averaging preserves it exactly.
    np.testing.assert_allclose(phi_mc.sum(), v(full) - v(empty), atol=1e-12)