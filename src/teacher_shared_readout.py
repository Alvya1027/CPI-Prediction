"""Float64 linear readout for a fixed, weight-shared optical reservoir.

Only the output layer is fitted.  If ``z`` is a train-standardized reservoir
state, both Siamese branches use ``f(z) = b + w.T @ z``.  Hence a pair obeys
``f(z_i)-f(z_j) = w.T @ (z_i-z_j)`` and the intercept cancels exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "AggregationResult",
    "SharedReadoutModel",
    "TrainOnlyStateStandardizer",
    "aggregate_pair_predictions",
    "fit_absolute_ridge",
    "fit_joint_shared_readout",
    "target_balanced_pair_weights",
]


def _states(values: object, name: str, width: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty 2D state matrix")
    if width is not None and matrix.shape[1] != width:
        raise ValueError(f"{name} has {matrix.shape[1]} features; expected {width}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return matrix


def _vector(
    values: object,
    name: str,
    length: int | None = None,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim == 2 and 1 in vector.shape:
        vector = vector.reshape(-1)
    if vector.ndim != 1 or (len(vector) == 0 and not allow_empty):
        raise ValueError(f"{name} must be a {'non-empty ' if not allow_empty else ''}1D vector")
    if length is not None and len(vector) != length:
        raise ValueError(f"{name} has length {len(vector)}; expected {length}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return vector


def _indices(values: object | None, name: str, upper: int) -> np.ndarray:
    if values is None:
        return np.empty(0, dtype=np.int64)
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if len(raw) == 0:
        return np.empty(0, dtype=np.int64)
    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer indices") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError(f"{name} must contain finite integer indices")
    result = numeric.astype(np.int64)
    if np.any(result < 0) or np.any(result >= upper):
        raise ValueError(f"{name} contains an index outside [0, {upper - 1}]")
    return result


def _nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class TrainOnlyStateStandardizer:
    """Scaling statistics fitted only on the supplied training states."""

    mean_: np.ndarray
    scale_: np.ndarray
    var_: np.ndarray
    n_samples_seen_: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean_, dtype=np.float64).reshape(-1).copy()
        scale = np.asarray(self.scale_, dtype=np.float64).reshape(-1).copy()
        variance = np.asarray(self.var_, dtype=np.float64).reshape(-1).copy()
        if not len(mean) or len(scale) != len(mean) or len(variance) != len(mean):
            raise ValueError("standardizer statistics must have equal non-zero widths")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("standardizer statistics must be finite")
        if not np.isfinite(variance).all() or np.any(variance < 0) or np.any(scale <= 0):
            raise ValueError("standardizer variances/scales are invalid")
        if int(self.n_samples_seen_) <= 0:
            raise ValueError("n_samples_seen_ must be positive")
        for array in (mean, scale, variance):
            array.setflags(write=False)
        object.__setattr__(self, "mean_", mean)
        object.__setattr__(self, "scale_", scale)
        object.__setattr__(self, "var_", variance)
        object.__setattr__(self, "n_samples_seen_", int(self.n_samples_seen_))

    @property
    def n_features_in_(self) -> int:
        return int(len(self.mean_))

    @classmethod
    def fit(cls, train_states: object) -> "TrainOnlyStateStandardizer":
        states = _states(train_states, "train_states")
        mean = np.mean(states, axis=0, dtype=np.float64)
        variance = np.var(states, axis=0, dtype=np.float64)
        scale = np.sqrt(variance)
        scale[scale == 0] = 1.0
        return cls(mean, scale, variance, states.shape[0])

    def transform(self, states: object) -> np.ndarray:
        matrix = _states(states, "states", self.n_features_in_)
        return np.asarray((matrix - self.mean_) / self.scale_, dtype=np.float64)

    def to_npz_dict(self, prefix: str = "state_scaler_") -> dict[str, np.ndarray]:
        return {
            f"{prefix}mean": self.mean_.copy(),
            f"{prefix}scale": self.scale_.copy(),
            f"{prefix}var": self.var_.copy(),
            f"{prefix}n_samples_seen": np.asarray(self.n_samples_seen_, dtype=np.int64),
        }


@dataclass(frozen=True)
class SharedReadoutModel:
    """One fitted output vector shared by target and reference branches."""

    standardizer: TrainOnlyStateStandardizer
    coef_: np.ndarray
    intercept_: float
    alpha: float
    pair_weight: float
    absolute_weight: float
    fit_kind: str
    n_absolute_samples_: int
    n_pairs_: int
    n_pair_targets_: int
    solver_rank_: int

    def __post_init__(self) -> None:
        coef = np.asarray(self.coef_, dtype=np.float64).reshape(-1).copy()
        if len(coef) != self.standardizer.n_features_in_:
            raise ValueError("coef_ width does not match the standardizer")
        if not np.isfinite(coef).all() or not np.isfinite(self.intercept_):
            raise ValueError("readout parameters must be finite")
        _nonnegative(self.alpha, "alpha")
        _nonnegative(self.pair_weight, "pair_weight")
        _nonnegative(self.absolute_weight, "absolute_weight")
        coef.setflags(write=False)
        object.__setattr__(self, "coef_", coef)
        object.__setattr__(self, "intercept_", float(self.intercept_))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "pair_weight", float(self.pair_weight))
        object.__setattr__(self, "absolute_weight", float(self.absolute_weight))

    @property
    def lambda_pair(self) -> float:
        """Report-friendly alias for ``pair_weight``."""
        return self.pair_weight

    def predict_direct(self, states: object) -> np.ndarray:
        z = self.standardizer.transform(states)
        return np.asarray(self.intercept_ + z @ self.coef_, dtype=np.float64)

    def predict_delta(self, target_states: object, reference_states: object) -> np.ndarray:
        """Predict target-minus-reference; the shared intercept is absent."""
        target = self.standardizer.transform(target_states)
        reference = self.standardizer.transform(reference_states)
        if target.shape != reference.shape:
            raise ValueError("target_states and reference_states must have identical shapes")
        return np.asarray((target - reference) @ self.coef_, dtype=np.float64)

    def predict_from_references(
        self,
        target_states: object,
        reference_states: object,
        reference_targets: object,
    ) -> np.ndarray:
        """Return pair predictions ``y_j + w.T @ (z_i-z_j)``."""
        delta = self.predict_delta(target_states, reference_states)
        y_reference = _vector(reference_targets, "reference_targets", len(delta))
        return np.asarray(y_reference + delta, dtype=np.float64)

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        """Return all fitted values for ``numpy.savez(..., **result)``."""
        result = self.standardizer.to_npz_dict()
        result.update(
            {
                "coef": self.coef_.copy(),
                "intercept": np.asarray(self.intercept_, dtype=np.float64),
                "alpha": np.asarray(self.alpha, dtype=np.float64),
                "pair_weight": np.asarray(self.pair_weight, dtype=np.float64),
                "absolute_weight": np.asarray(self.absolute_weight, dtype=np.float64),
                "fit_kind": np.asarray(self.fit_kind),
                "n_absolute_samples": np.asarray(self.n_absolute_samples_, dtype=np.int64),
                "n_pairs": np.asarray(self.n_pairs_, dtype=np.int64),
                "n_pair_targets": np.asarray(self.n_pair_targets_, dtype=np.int64),
                "solver_rank": np.asarray(self.solver_rank_, dtype=np.int64),
            }
        )
        return result


def target_balanced_pair_weights(pair_target_indices: object) -> np.ndarray:
    """Give every distinct target equal total pair-loss weight.

    A target with ``m_i`` references receives row weight ``1/(T*m_i)``, where
    ``T`` is the number of distinct target months.  The result sums to one.
    """
    raw = np.asarray(pair_target_indices)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("pair_target_indices must be a non-empty 1D vector")
    positions: dict[object, int] = {}
    group = np.empty(len(raw), dtype=np.int64)
    for row, value in enumerate(raw.tolist()):
        try:
            position = positions.get(value)
        except TypeError as exc:
            raise ValueError("pair target identifiers must be hashable") from exc
        if position is None:
            position = len(positions)
            positions[value] = position
        group[row] = position
    counts = np.bincount(group, minlength=len(positions)).astype(np.float64)
    return np.asarray(1.0 / (len(positions) * counts[group]), dtype=np.float64)


def _solve(
    z: np.ndarray,
    y: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    alpha: float,
    pair_weight: float,
    absolute_weight: float,
) -> tuple[np.ndarray, int, int]:
    """Solve weighted least squares; column zero is the unpenalized bias."""
    n_samples, n_features = z.shape
    designs: list[np.ndarray] = []
    responses: list[np.ndarray] = []
    if absolute_weight > 0:
        design = np.column_stack([np.ones(n_samples), z])
        scale = np.sqrt(absolute_weight / n_samples)
        designs.append(scale * design)
        responses.append(scale * y)

    n_pair_targets = 0
    if pair_weight > 0:
        if len(pair_i) == 0:
            raise ValueError("positive pair_weight requires at least one pair")
        pair_design = np.column_stack([np.zeros(len(pair_i)), z[pair_i] - z[pair_j]])
        pair_response = y[pair_i] - y[pair_j]
        scale = np.sqrt(pair_weight * target_balanced_pair_weights(pair_i))
        designs.append(scale[:, None] * pair_design)
        responses.append(scale * pair_response)
        n_pair_targets = int(np.unique(pair_i).size)

    if alpha > 0:
        penalty = np.zeros((n_features, n_features + 1), dtype=np.float64)
        penalty[:, 1:] = np.sqrt(alpha) * np.eye(n_features)
        designs.append(penalty)
        responses.append(np.zeros(n_features, dtype=np.float64))

    solution, _residuals, rank, _singular = np.linalg.lstsq(
        np.vstack(designs), np.concatenate(responses), rcond=None
    )
    if not np.isfinite(solution).all():
        raise FloatingPointError("shared readout fit produced non-finite parameters")
    return solution, int(rank), n_pair_targets


def fit_joint_shared_readout(
    train_states: object,
    train_targets: object,
    pair_target_indices: object | None,
    pair_reference_indices: object | None,
    *,
    alpha: float = 1.0,
    pair_weight: float = 1.0,
    absolute_weight: float = 1.0,
) -> SharedReadoutModel:
    """Fit a shared output layer by closed-form weighted least squares.

    Pair indices are positional rows of the training arrays.  Their labels are
    derived as ``y_i-y_j`` from the same training targets, so pairs do not add
    accessible months.  The objective is::

        absolute_weight * mean_i (b + w.T z_i - y_i)^2
        + pair_weight * mean_target_i mean_ref_j
              (w.T(z_i-z_j) - (y_i-y_j))^2
        + alpha * ||w||^2

    ``absolute_weight=0`` provides a pair-only ablation (its unidentified bias
    is returned as zero).  ``pair_weight=0, absolute_weight=1`` is numerically
    identical to :func:`fit_absolute_ridge`.
    """
    states = _states(train_states, "train_states")
    targets = _vector(train_targets, "train_targets", states.shape[0])
    alpha_value = _nonnegative(alpha, "alpha")
    pair_value = _nonnegative(pair_weight, "pair_weight")
    absolute_value = _nonnegative(absolute_weight, "absolute_weight")
    if pair_value == 0 and absolute_value == 0:
        raise ValueError("at least one loss weight must be positive")

    pair_i = _indices(pair_target_indices, "pair_target_indices", states.shape[0])
    pair_j = _indices(pair_reference_indices, "pair_reference_indices", states.shape[0])
    if len(pair_i) != len(pair_j):
        raise ValueError("pair target/reference index vectors must have equal lengths")

    standardizer = TrainOnlyStateStandardizer.fit(states)
    solution, rank, n_pair_targets = _solve(
        standardizer.transform(states),
        targets,
        pair_i,
        pair_j,
        alpha=alpha_value,
        pair_weight=pair_value,
        absolute_weight=absolute_value,
    )
    return SharedReadoutModel(
        standardizer=standardizer,
        coef_=solution[1:],
        intercept_=float(solution[0]),
        alpha=alpha_value,
        pair_weight=pair_value,
        absolute_weight=absolute_value,
        fit_kind="joint_shared_readout",
        n_absolute_samples_=states.shape[0] if absolute_value > 0 else 0,
        n_pairs_=len(pair_i) if pair_value > 0 else 0,
        n_pair_targets_=n_pair_targets,
        solver_rank_=rank,
    )


def fit_absolute_ridge(
    train_states: object,
    train_targets: object,
    *,
    alpha: float = 1.0,
) -> SharedReadoutModel:
    """Fit ``mean squared error + alpha*||w||^2``; bias is unpenalized."""
    fitted = fit_joint_shared_readout(
        train_states,
        train_targets,
        None,
        None,
        alpha=alpha,
        pair_weight=0.0,
        absolute_weight=1.0,
    )
    return SharedReadoutModel(
        standardizer=fitted.standardizer,
        coef_=fitted.coef_,
        intercept_=fitted.intercept_,
        alpha=fitted.alpha,
        pair_weight=0.0,
        absolute_weight=1.0,
        fit_kind="absolute_only_ridge",
        n_absolute_samples_=fitted.n_absolute_samples_,
        n_pairs_=0,
        n_pair_targets_=0,
        solver_rank_=fitted.solver_rank_,
    )


@dataclass(frozen=True)
class AggregationResult:
    """Grouped predictions plus weights aligned with input pair rows."""

    target_ids: np.ndarray
    predictions: np.ndarray
    num_references: np.ndarray
    pair_weights: np.ndarray


def aggregate_pair_predictions(
    target_ids: object,
    pair_predictions: object,
    *,
    method: Literal["mean", "inverse_distance"] = "mean",
    distances: object | None = None,
    distance_epsilon: float = 1e-12,
) -> AggregationResult:
    """Aggregate reference-derived predictions in first-target order.

    With inverse distance, zero-distance references share all weight equally;
    otherwise weights are proportional to ``1/d``.  ``pair_weights`` align to
    the input rows and sum to one separately for every target.
    """
    ids = np.asarray(target_ids)
    if ids.ndim != 1 or len(ids) == 0:
        raise ValueError("target_ids must be a non-empty 1D vector")
    values = _vector(pair_predictions, "pair_predictions", len(ids))
    normalized_method = str(method).replace("-", "_")
    if normalized_method not in {"mean", "inverse_distance"}:
        raise ValueError("method must be 'mean' or 'inverse_distance'")
    epsilon = float(distance_epsilon)
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("distance_epsilon must be finite and positive")

    distance_values: np.ndarray | None = None
    if normalized_method == "inverse_distance":
        if distances is None:
            raise ValueError("inverse_distance aggregation requires distances")
        distance_values = _vector(distances, "distances", len(ids))
        if np.any(distance_values < 0):
            raise ValueError("distances must be non-negative")

    ordered_ids: list[object] = []
    groups: list[list[int]] = []
    positions: dict[object, int] = {}
    for row, value in enumerate(ids.tolist()):
        try:
            position = positions.get(value)
        except TypeError as exc:
            raise ValueError("target_ids must contain hashable values") from exc
        if position is None:
            position = len(ordered_ids)
            positions[value] = position
            ordered_ids.append(value)
            groups.append([])
        groups[position].append(row)

    weights = np.zeros(len(ids), dtype=np.float64)
    predictions = np.empty(len(groups), dtype=np.float64)
    counts = np.empty(len(groups), dtype=np.int64)
    for group_number, row_list in enumerate(groups):
        rows = np.asarray(row_list, dtype=np.int64)
        counts[group_number] = len(rows)
        if normalized_method == "mean":
            group_weights = np.full(len(rows), 1.0 / len(rows), dtype=np.float64)
        else:
            assert distance_values is not None
            group_distances = distance_values[rows]
            zero = group_distances <= epsilon
            group_weights = zero.astype(np.float64) if np.any(zero) else 1.0 / group_distances
            group_weights /= np.sum(group_weights)
        weights[rows] = group_weights
        predictions[group_number] = np.dot(group_weights, values[rows])

    return AggregationResult(
        target_ids=np.asarray(ordered_ids, dtype=ids.dtype),
        predictions=predictions,
        num_references=counts,
        pair_weights=weights,
    )
