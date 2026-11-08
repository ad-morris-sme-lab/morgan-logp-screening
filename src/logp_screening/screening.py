"""Candidate prediction and ranking."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor

from .fingerprints import molecular_features


def rank_candidates(
    model: ExtraTreesRegressor, candidates: Sequence[str]
) -> list[tuple[str, float]]:
    """Predict logP for candidates and rank from highest to lowest.

    Parameters
    ----------
    model : ExtraTreesRegressor
        Fitted model trained on Morgan fingerprints.
    candidates : sequence of str
        Valid candidate molecule SMILES.

    Returns
    -------
    list of tuple[str, float]
        ``(SMILES, predicted_logP)`` records sorted in descending prediction.

    Raises
    ------
    ValueError
        If the model is unfitted or candidates are empty or invalid.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")
    if not hasattr(model, "n_features_in_"):
        raise ValueError("model must be fitted before screening")
    features = molecular_features(candidates)
    if features.shape[1] != model.n_features_in_:
        raise ValueError("candidate fingerprint size does not match fitted model")
    predicted: NDArray[np.float64] = model.predict(features)
    return sorted(
        ((smiles, float(value)) for smiles, value in zip(candidates, predicted, strict=True)),
        key=lambda item: item[1], reverse=True,
    )
