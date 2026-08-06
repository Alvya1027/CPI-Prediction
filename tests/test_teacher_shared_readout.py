from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.teacher_shared_readout import (
    TrainOnlyStateStandardizer,
    aggregate_pair_predictions,
    fit_absolute_ridge,
    fit_joint_shared_readout,
    target_balanced_pair_weights,
)
from src.teacher_shared_readout_pipeline import select_references


def _linear_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    states = rng.normal(size=(12, 4))
    targets = 100.0 + states @ np.asarray([0.4, -0.2, 0.1, 0.3])
    return states, targets


def test_standardizer_is_fitted_only_on_supplied_training_states() -> None:
    train = np.asarray([[1.0, 4.0], [3.0, 4.0]])
    scaler = TrainOnlyStateStandardizer.fit(train)

    assert np.allclose(scaler.mean_, [2.0, 4.0])
    assert np.allclose(scaler.scale_, [1.0, 1.0])
    assert np.allclose(scaler.transform([[5.0, 6.0]]), [[3.0, 2.0]])


def test_shared_output_identity_antisymmetry_and_bias_cancellation() -> None:
    states, targets = _linear_fixture()
    pair_i = np.asarray([4, 5, 6, 7, 8])
    pair_j = np.asarray([0, 1, 2, 3, 4])
    model = fit_joint_shared_readout(
        states,
        targets,
        pair_i,
        pair_j,
        alpha=1e-6,
        pair_weight=1.0,
    )

    direct_i = model.predict_direct(states[pair_i])
    direct_j = model.predict_direct(states[pair_j])
    delta = model.predict_delta(states[pair_i], states[pair_j])

    assert np.allclose(delta, direct_i - direct_j, atol=1e-12)
    assert np.allclose(
        model.predict_delta(states[pair_j], states[pair_i]), -delta, atol=1e-12
    )
    assert np.allclose(model.predict_delta(states[:3], states[:3]), 0.0, atol=1e-12)


def test_zero_pair_weight_is_identical_to_absolute_ridge() -> None:
    states, targets = _linear_fixture()
    absolute = fit_absolute_ridge(states, targets, alpha=0.25)
    joint = fit_joint_shared_readout(
        states,
        targets,
        np.asarray([3, 4, 5]),
        np.asarray([0, 1, 2]),
        alpha=0.25,
        pair_weight=0.0,
    )

    assert np.array_equal(absolute.coef_, joint.coef_)
    assert absolute.intercept_ == joint.intercept_
    assert np.array_equal(absolute.predict_direct(states), joint.predict_direct(states))


def test_normalized_alpha_matches_legacy_sklearn_scale_for_train50() -> None:
    rng = np.random.default_rng(19)
    states = rng.normal(size=(50, 5))
    targets = rng.normal(size=50)
    normalized = fit_absolute_ridge(states, targets, alpha=2.0)
    scaler = StandardScaler().fit(states)
    legacy = Ridge(alpha=100.0).fit(scaler.transform(states), targets)

    assert np.allclose(normalized.coef_, legacy.coef_, atol=1e-12)
    assert np.isclose(normalized.intercept_, legacy.intercept_, atol=1e-12)
    assert np.allclose(
        normalized.predict_direct(states),
        legacy.predict(scaler.transform(states)),
        atol=1e-12,
    )


def test_pair_weights_give_every_target_equal_total_influence() -> None:
    target_ids = np.asarray([10, 11, 11, 12, 12, 12])
    weights = target_balanced_pair_weights(target_ids)

    assert np.isclose(weights.sum(), 1.0)
    for target_id in np.unique(target_ids):
        assert np.isclose(weights[target_ids == target_id].sum(), 1.0 / 3.0)


def test_inverse_distance_aggregation_uses_only_distance() -> None:
    result = aggregate_pair_predictions(
        np.asarray([1, 1, 2, 2]),
        np.asarray([100.0, 102.0, 99.0, 103.0]),
        method="inverse_distance",
        distances=np.asarray([1.0, 3.0, 2.0, 2.0]),
    )

    assert result.target_ids.tolist() == [1, 2]
    assert np.allclose(result.predictions, [100.5, 101.0])
    assert np.allclose(result.pair_weights, [0.75, 0.25, 0.5, 0.5])


def test_reference_ranking_does_not_use_cpi_labels() -> None:
    candidates = pd.DataFrame(
        {
            "sample_i_id": [10, 10, 10],
            "sample_j_id": [1, 2, 3],
            "target_j_date": ["2017-01", "2017-02", "2017-03"],
            "window_distance": [0.4, 0.1, 0.2],
            "cpi_i": [100.0, 100.0, 100.0],
            "cpi_j": [90.0, 110.0, 105.0],
            "delta_cpi": [10.0, -10.0, -5.0],
        }
    )
    selected = select_references(candidates, 2)
    changed = candidates.copy()
    changed[["cpi_i", "cpi_j", "delta_cpi"]] = np.asarray(
        [[999.0, -1.0, 1000.0], [0.0, 500.0, -500.0], [2.0, 2.0, 0.0]]
    )
    selected_changed = select_references(changed, 2)

    assert selected["sample_j_id"].tolist() == [2, 3]
    assert selected_changed["sample_j_id"].tolist() == [2, 3]
