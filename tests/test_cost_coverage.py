"""Tests for cost-coverage curve helpers (P-AUCCC)."""

import numpy as np
import pytest

from model_router_toolkit.evaluate import (
    _build_cost_map,
    _build_routing_curve,
    _padded_auc,
    _pareto_frontier,
    _route_at_tolerance,
)


# ---------------------------------------------------------------------------
# _pareto_frontier
# ---------------------------------------------------------------------------

class TestParetoFrontier:
    def test_single_point(self):
        assert _pareto_frontier([(1.0, 0.8)]) == [(1.0, 0.8)]

    def test_dominated_points_removed(self):
        points = [(1.0, 0.6), (2.0, 0.5), (3.0, 0.9)]
        result = _pareto_frontier(points)
        assert result == [(1.0, 0.6), (3.0, 0.9)]

    def test_all_on_frontier(self):
        points = [(1.0, 0.5), (2.0, 0.7), (3.0, 0.9)]
        result = _pareto_frontier(points)
        assert result == points

    def test_sorted_by_cost(self):
        points = [(3.0, 0.9), (1.0, 0.5), (2.0, 0.7)]
        result = _pareto_frontier(points)
        costs = [p[0] for p in result]
        assert costs == sorted(costs)

    def test_empty_input(self):
        assert _pareto_frontier([]) == []

    def test_equal_cost_keeps_higher_acc(self):
        points = [(1.0, 0.5), (1.0, 0.8), (2.0, 0.9)]
        result = _pareto_frontier(points)
        assert (1.0, 0.5) in result
        assert (2.0, 0.9) in result


# ---------------------------------------------------------------------------
# _route_at_tolerance
# ---------------------------------------------------------------------------

class TestRouteAtTolerance:
    def test_zero_tolerance_picks_most_confident(self):
        probs = np.array([
            [0.9, 0.3],
            [0.4, 0.8],
        ])
        costs = np.array([1.0, 2.0])
        choices = _route_at_tolerance(probs, costs, tol=0.0)
        np.testing.assert_array_equal(choices, [0, 1])

    def test_high_tolerance_picks_cheapest(self):
        probs = np.array([
            [0.9, 0.3],
            [0.4, 0.8],
        ])
        costs = np.array([1.0, 2.0])
        choices = _route_at_tolerance(probs, costs, tol=1.0)
        np.testing.assert_array_equal(choices, [0, 0])

    def test_tolerance_threshold_boundary(self):
        probs = np.array([[0.9, 0.85, 0.5]])
        costs = np.array([3.0, 2.0, 1.0])
        choices = _route_at_tolerance(probs, costs, tol=0.05)
        assert choices[0] == 1

    def test_single_model(self):
        probs = np.array([[0.7], [0.3]])
        costs = np.array([1.0])
        choices = _route_at_tolerance(probs, costs, tol=0.0)
        np.testing.assert_array_equal(choices, [0, 0])


# ---------------------------------------------------------------------------
# _build_routing_curve
# ---------------------------------------------------------------------------

class TestBuildRoutingCurve:
    def test_returns_sorted_curve(self):
        rng = np.random.default_rng(42)
        N, M = 100, 3
        model_names = ["cheap", "mid", "expensive"]
        costs = {"cheap": 0.5, "mid": 1.0, "expensive": 2.0}
        Y = rng.integers(0, 2, size=(N, M))
        probs = rng.random((N, M))

        curve, dists = _build_routing_curve(Y, probs, model_names, costs)

        assert len(curve) > 0
        assert len(curve) == len(dists)
        curve_costs = [p[0] for p in curve]
        assert curve_costs == sorted(curve_costs)

    def test_distributions_sum_to_one(self):
        rng = np.random.default_rng(42)
        N, M = 50, 2
        model_names = ["a", "b"]
        costs = {"a": 1.0, "b": 2.0}
        Y = rng.integers(0, 2, size=(N, M))
        probs = rng.random((N, M))

        _, dists = _build_routing_curve(Y, probs, model_names, costs)

        for d in dists:
            model_sum = sum(v for k, v in d.items() if k != "_tol")
            assert abs(model_sum - 1.0) < 1e-5

    def test_curve_accuracy_in_bounds(self):
        rng = np.random.default_rng(42)
        N, M = 80, 3
        model_names = ["a", "b", "c"]
        costs = {"a": 1.0, "b": 2.0, "c": 3.0}
        Y = rng.integers(0, 2, size=(N, M))
        probs = rng.random((N, M))

        curve, _ = _build_routing_curve(Y, probs, model_names, costs)
        for _, acc in curve:
            assert 0.0 <= acc <= 1.0

    def test_deduplication(self):
        probs = np.array([[0.9, 0.1], [0.9, 0.1]])
        Y = np.array([[1, 0], [1, 0]])
        model_names = ["a", "b"]
        costs = {"a": 1.0, "b": 2.0}
        curve, _ = _build_routing_curve(Y, probs, model_names, costs)
        curve_keys = [(round(c, 10), round(a, 10)) for c, a in curve]
        assert len(curve_keys) == len(set(curve_keys))


# ---------------------------------------------------------------------------
# _padded_auc
# ---------------------------------------------------------------------------

class TestPaddedAuc:
    def test_flat_curve_returns_zero(self):
        curve = [(1.0, 0.5), (2.0, 0.5)]
        result = _padded_auc(curve, c_min=1.0, c_max=2.0, floor_acc=0.5)
        assert abs(result) < 1e-10

    def test_perfect_step_curve(self):
        curve = [(1.0, 1.0), (2.0, 1.0)]
        result = _padded_auc(curve, c_min=1.0, c_max=2.0, floor_acc=0.0)
        assert abs(result - 1.0) < 1e-10

    def test_padding_extends_range(self):
        curve = [(1.5, 0.8)]
        result = _padded_auc(curve, c_min=1.0, c_max=2.0, floor_acc=0.5)
        assert result > 0

    def test_equal_range_returns_zero(self):
        curve = [(1.0, 0.8)]
        assert _padded_auc(curve, c_min=1.0, c_max=1.0, floor_acc=0.5) == 0.0

    def test_empty_curve(self):
        result = _padded_auc([], c_min=1.0, c_max=2.0, floor_acc=0.5)
        assert result == 0.0

    def test_normalized_to_unit_range(self):
        curve = [(0.0, 0.8), (10.0, 0.8)]
        result = _padded_auc(curve, c_min=0.0, c_max=10.0, floor_acc=0.5)
        assert abs(result - 0.3) < 1e-10

    def test_monotonic_lift(self):
        low = _padded_auc(
            [(1.0, 0.6), (2.0, 0.7)], c_min=1.0, c_max=2.0, floor_acc=0.5,
        )
        high = _padded_auc(
            [(1.0, 0.9), (2.0, 0.95)], c_min=1.0, c_max=2.0, floor_acc=0.5,
        )
        assert high > low


# ---------------------------------------------------------------------------
# _build_cost_map
# ---------------------------------------------------------------------------

class TestBuildCostMap:
    def test_from_checkpoint_pool_config(self):
        ckpt = {
            "pool_config": [
                {"name": "a", "cost_per_m_input_tokens": 0.5},
                {"name": "b", "cost_per_m_input_tokens": 1.5},
            ],
        }
        result = _build_cost_map(ckpt, ["a", "b"])
        assert result == {"a": 0.5, "b": 1.5}

    def test_missing_model_returns_none(self):
        ckpt = {
            "pool_config": [
                {"name": "a", "cost_per_m_input_tokens": 0.5},
            ],
        }
        assert _build_cost_map(ckpt, ["a", "b"]) is None

    def test_all_zero_costs_returns_none(self):
        ckpt = {
            "pool_config": [
                {"name": "a", "cost_per_m_input_tokens": 0.0},
                {"name": "b", "cost_per_m_input_tokens": 0.0},
            ],
        }
        assert _build_cost_map(ckpt, ["a", "b"]) is None

    def test_empty_checkpoint_returns_none(self):
        assert _build_cost_map({}, ["a", "b"]) is None

    def test_config_fallback(self):
        ckpt = {"pool_config": []}

        class FakeModel:
            def __init__(self, name, cost):
                self.name = name
                self.cost_per_m_input_tokens = cost

        class FakeConfig:
            models = [FakeModel("a", 0.5), FakeModel("b", 1.5)]

        result = _build_cost_map(ckpt, ["a", "b"], config=FakeConfig())
        assert result == {"a": 0.5, "b": 1.5}

    def test_checkpoint_takes_priority_over_config(self):
        ckpt = {
            "pool_config": [
                {"name": "a", "cost_per_m_input_tokens": 0.5},
                {"name": "b", "cost_per_m_input_tokens": 1.5},
            ],
        }

        class FakeModel:
            def __init__(self, name, cost):
                self.name = name
                self.cost_per_m_input_tokens = cost

        class FakeConfig:
            models = [FakeModel("a", 9.0), FakeModel("b", 9.0)]

        result = _build_cost_map(ckpt, ["a", "b"], config=FakeConfig())
        assert result == {"a": 0.5, "b": 1.5}


# ---------------------------------------------------------------------------
# End-to-end: metrics from known data
# ---------------------------------------------------------------------------

class TestEndToEndMetrics:
    """Verify the full pipeline produces sensible metrics on synthetic data."""

    @pytest.fixture()
    def scenario(self):
        """Two models: cheap (60% accuracy) and expensive (90% accuracy).

        Router probabilities are well-calibrated: p(correct) ~ actual.
        """
        rng = np.random.default_rng(42)
        N = 200
        model_names = ["cheap", "expensive"]
        costs = {"cheap": 0.5, "expensive": 2.0}

        Y = np.zeros((N, 2), dtype=int)
        Y[:, 0] = (rng.random(N) < 0.6).astype(int)
        Y[:, 1] = (rng.random(N) < 0.9).astype(int)

        probs = np.zeros((N, 2))
        probs[:, 0] = Y[:, 0] * 0.7 + (1 - Y[:, 0]) * 0.3 + rng.normal(0, 0.05, N)
        probs[:, 1] = Y[:, 1] * 0.8 + (1 - Y[:, 1]) * 0.2 + rng.normal(0, 0.05, N)
        probs = np.clip(probs, 0.01, 0.99)

        return Y, probs, model_names, costs

    def test_p_auccc_positive(self, scenario):
        Y, probs, model_names, costs = scenario
        c_min, c_max = 0.5, 2.0
        floor_acc = float(Y[:, 0].mean())
        curve, _ = _build_routing_curve(Y, probs, model_names, costs)
        p_auccc = _padded_auc(curve, c_min, c_max, floor_acc)
        assert p_auccc > 0

    def test_mdp_auccc_positive_for_good_router(self, scenario):
        Y, probs, model_names, costs = scenario
        c_min, c_max = 0.5, 2.0
        floor_acc = float(Y[:, 0].mean())

        model_points = [
            (costs[mn], float(Y[:, mi].mean()))
            for mi, mn in enumerate(model_names)
        ]
        curve, _ = _build_routing_curve(Y, probs, model_names, costs)
        pareto_curve = _pareto_frontier(model_points)

        p_auccc = _padded_auc(curve, c_min, c_max, floor_acc)
        pareto_auccc = _padded_auc(pareto_curve, c_min, c_max, floor_acc)
        mdp_auccc = p_auccc - pareto_auccc
        assert mdp_auccc > 0, "well-calibrated router should beat model-only Pareto"

    def test_pdp_auccc_nonnegative(self, scenario):
        Y, probs, model_names, costs = scenario
        c_min, c_max = 0.5, 2.0
        floor_acc = float(Y[:, 0].mean())

        model_points = [
            (costs[mn], float(Y[:, mi].mean()))
            for mi, mn in enumerate(model_names)
        ]
        curve, _ = _build_routing_curve(Y, probs, model_names, costs)
        combined_pareto = _pareto_frontier(model_points + curve)

        p_auccc = _padded_auc(curve, c_min, c_max, floor_acc)
        combined_auccc = _padded_auc(combined_pareto, c_min, c_max, floor_acc)
        pdp_auccc = combined_auccc - p_auccc
        assert pdp_auccc >= -1e-10

    def test_pareto_auccc_less_than_p_auccc(self, scenario):
        Y, probs, model_names, costs = scenario
        c_min, c_max = 0.5, 2.0
        floor_acc = float(Y[:, 0].mean())

        model_points = [
            (costs[mn], float(Y[:, mi].mean()))
            for mi, mn in enumerate(model_names)
        ]
        curve, _ = _build_routing_curve(Y, probs, model_names, costs)
        pareto_curve = _pareto_frontier(model_points)

        p_auccc = _padded_auc(curve, c_min, c_max, floor_acc)
        pareto_auccc = _padded_auc(pareto_curve, c_min, c_max, floor_acc)
        assert p_auccc >= pareto_auccc
