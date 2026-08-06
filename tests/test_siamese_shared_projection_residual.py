from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import src.siamese_shared_projection_residual as residual_module
from src.siamese_shared_projection import DATA_DIR, NetworkSpec, STATE_DIR, _state_matrix
from src.siamese_shared_projection_residual import (
    ResidualSpec,
    SharedProjectionResidual,
    _ensemble_predictions,
    _predict_pairs,
    load_frozen_ridge_base,
    load_pretest_pairs,
)
from src.siamese_validation_search import load_validation_state_lookup
from src.siamese_split_isolation import build_isolated_closed_train_pool


def _small_model() -> SharedProjectionResidual:
    return SharedProjectionResidual(
        state_width=3,
        spec=NetworkSpec(
            projection_hidden_dim=4,
            embedding_dim=2,
            relation_hidden_dim=3,
            dropout=0.0,
            epochs=1,
        ),
    ).eval()


def test_zero_initialized_residual_is_exactly_zero() -> None:
    torch.manual_seed(9)
    model = _small_model()
    first = torch.tensor([[0.2, -0.1, 0.8]])
    second = torch.tensor([[-0.3, 0.4, 0.1]])

    with torch.no_grad():
        correction, _, _ = model(first, second)

    assert torch.equal(correction, torch.zeros_like(correction))


def test_residual_remains_antisymmetric_after_parameter_change() -> None:
    torch.manual_seed(10)
    model = _small_model()
    with torch.no_grad():
        model.relation[-1].weight.fill_(0.2)
        model.relation[-1].bias.fill_(0.1)
    first = torch.tensor([[0.2, -0.1, 0.8]])
    second = torch.tensor([[-0.3, 0.4, 0.1]])

    with torch.no_grad():
        forward, _, _ = model(first, second)
        reverse, _, _ = model(second, first)

    assert torch.allclose(forward, -reverse, atol=1e-7)


def test_seed_ensemble_is_equal_weight_average() -> None:
    base = pd.DataFrame(
        {
            "sample_i_id": [1, 2],
            "target_date": ["2020-01", "2020-02"],
            "cpi_actual": [100.0, 101.0],
        }
    )
    first = base.assign(cpi_predicted=[99.0, 102.0])
    second = base.assign(cpi_predicted=[101.0, 100.0])

    result = _ensemble_predictions({42: first, 43: second})

    assert np.allclose(result["cpi_predicted"], [100.0, 101.0])
    assert np.allclose(result["prediction_seed_std"], [1.0, 1.0])


def test_epoch_zero_reproduces_frozen_ssa_validation_without_test_states() -> None:
    train_pool = build_isolated_closed_train_pool(DATA_DIR)
    _, validation_pairs = load_pretest_pairs()
    state_lookup = load_validation_state_lookup(STATE_DIR)
    train_ids = train_pool["index"]["sample_id"].to_numpy(dtype=int)
    state_scaler = StandardScaler().fit(_state_matrix(train_ids, state_lookup))
    model = SharedProjectionResidual(50, ResidualSpec().network_spec()).eval()

    _, predictions = _predict_pairs(
        model,
        state_scaler,
        load_frozen_ridge_base(),
        correction_scale=1.0,
        pairs=validation_pairs,
        state_lookup=state_lookup,
    )

    assert int(state_scaler.n_samples_seen_) == 50
    assert predictions["sample_i_id"].nunique() == 45
    assert np.isclose(np.mean(predictions["absolute_error"]), 0.35304854906951594)
    assert np.isclose(
        np.sqrt(np.mean(np.square(predictions["error"]))),
        0.44679092931184033,
    )


def test_validation_only_pipeline_never_calls_test_loaders(
    monkeypatch,
    tmp_path,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("a test-stage loader was called during validation-only")

    monkeypatch.setattr(residual_module, "audit_profile", forbidden)
    monkeypatch.setattr(residual_module, "load_postfreeze_test_pairs", forbidden)
    monkeypatch.setattr(residual_module, "load_state_lookup", forbidden)

    result = residual_module.run_pipeline(
        output_dir=tmp_path / "validation_only",
        seeds=(42,),
        similarity_weights=(0.0,),
        spec=residual_module.ResidualSpec(max_epochs=0),
        validation_only=True,
    )

    assert np.isclose(result["val_rmse"].iloc[0], 0.44679092931184144)
