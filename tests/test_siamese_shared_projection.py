from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.siamese_shared_projection import (
    NetworkSpec,
    SharedProjectionSiamese,
    _chronological_block_rmse_std,
    select_embedding_references,
)


def test_shared_projection_uses_one_parameter_set_for_both_branches() -> None:
    model = SharedProjectionSiamese(
        state_width=4,
        spec=NetworkSpec(
            projection_hidden_dim=3,
            embedding_dim=2,
            relation_hidden_dim=2,
            dropout=0.0,
            epochs=1,
        ),
    ).eval()
    states = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

    with torch.no_grad():
        first = model.encode(states)
        second = model.encode(states)

    assert torch.allclose(first, second)
    assert len(list(model.projection.parameters())) > 0


def test_relation_prediction_is_antisymmetric() -> None:
    torch.manual_seed(7)
    model = SharedProjectionSiamese(
        state_width=3,
        spec=NetworkSpec(
            projection_hidden_dim=4,
            embedding_dim=2,
            relation_hidden_dim=3,
            dropout=0.0,
            epochs=1,
        ),
    ).eval()
    first = torch.tensor([[0.2, -0.1, 0.8]])
    second = torch.tensor([[-0.3, 0.4, 0.1]])

    with torch.no_grad():
        forward, _, _ = model(first, second)
        reverse, _, _ = model(second, first)

    assert torch.allclose(forward, -reverse, atol=1e-7)


def test_embedding_reference_ranking_does_not_use_target_labels() -> None:
    candidates = pd.DataFrame(
        {
            "sample_i_id": [10, 10, 10],
            "sample_j_id": [1, 2, 3],
            "target_i_date": ["2020-01"] * 3,
            "target_j_date": ["2017-01", "2017-02", "2017-03"],
            "cpi_i": [100.0, 100.0, 100.0],
            "cpi_j": [99.0, 101.0, 105.0],
            "delta_cpi": [1.0, -1.0, -5.0],
        }
    )
    target_embeddings = {10: np.asarray([0.0, 0.0])}
    reference_embeddings = {
        1: np.asarray([2.0, 0.0]),
        2: np.asarray([0.1, 0.0]),
        3: np.asarray([0.3, 0.0]),
    }

    selected = select_embedding_references(
        candidates,
        target_embeddings,
        reference_embeddings,
        k_references=2,
    )
    changed_labels = candidates.copy()
    changed_labels["cpi_i"] = 999.0
    changed_labels["delta_cpi"] = [-999.0, 400.0, 0.0]
    selected_after_change = select_embedding_references(
        changed_labels,
        target_embeddings,
        reference_embeddings,
        k_references=2,
    )

    assert selected["sample_j_id"].tolist() == [2, 3]
    assert selected_after_change["sample_j_id"].tolist() == [2, 3]


def test_chronological_stability_uses_three_dataframe_blocks() -> None:
    predictions = pd.DataFrame(
        {
            "target_date": [f"2020-{month:02d}" for month in range(1, 7)],
            "error": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )

    result = _chronological_block_rmse_std(predictions)

    assert np.isclose(result, np.std([1.0, 2.0, 3.0]))
