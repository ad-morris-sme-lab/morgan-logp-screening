"""Reproducible model fitting, data splitting, and evaluation."""

from dataclasses import dataclass
from collections.abc import Sequence
from collections import Counter

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from .fingerprints import molecular_features


@dataclass(frozen=True)
class Evaluation:
    """Holdout and cross-validation metrics for a fitted model."""

    r2: float
    mae: float
    cv_r2_mean: float
    cv_r2_std: float


@dataclass(frozen=True)
class ScaffoldEvaluation:
    """Metrics for a holdout in which no Bemis-Murcko scaffold is shared."""

    r2: float
    mae: float
    test_molecules: int
    test_scaffolds: int


def split_data(
    fingerprints: NDArray[np.float64],
    labels: NDArray[np.float64],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Create a deterministic train/test split.

    Parameters
    ----------
    fingerprints : numpy.ndarray
        Two-dimensional feature matrix.
    labels : numpy.ndarray
        One-dimensional target vector matching the feature rows.
    test_size : float, default=0.2
        Fraction of samples reserved for testing.
    seed : int, default=42
        Random seed for the split.

    Returns
    -------
    tuple of numpy.ndarray
        ``X_train, X_test, y_train, y_test``.

    Raises
    ------
    ValueError
        If input dimensions, sizes, or split fraction are invalid.
    """
    features = np.asarray(fingerprints, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64)
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("features must be 2D and labels 1D with matching rows")
    if len(target) < 10 or not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
        raise ValueError("inputs must have at least 10 finite samples")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between zero and one")
    return train_test_split(features, target, test_size=test_size, random_state=seed)


def make_model(seed: int = 42) -> ExtraTreesRegressor:
    """Create a fixed-seed Extra Trees regression model.

    Parameters
    ----------
    seed : int, default=42
        Seed controlling tree construction.

    Returns
    -------
    sklearn.ensemble.ExtraTreesRegressor
        Unfitted regression model.
    """
    return ExtraTreesRegressor(
        n_estimators=100, min_samples_leaf=1, max_features=0.8,
        random_state=seed, n_jobs=1,
    )


def train_model(
    smiles: Sequence[str], labels: NDArray[np.float64], seed: int = 42
) -> tuple[ExtraTreesRegressor, Evaluation]:
    """Train on an 80% split and evaluate holdout plus five-fold CV metrics.

    Parameters
    ----------
    smiles : sequence of str
        Molecular SMILES strings.
    labels : numpy.ndarray
        Target logP values aligned to the SMILES.
    seed : int, default=42
        Seed used for splitting, CV shuffling, and the estimator.

    Returns
    -------
    tuple of ExtraTreesRegressor, Evaluation
        Fitted model and performance summary.
    """
    features = molecular_features(smiles)
    X_train, X_test, y_train, y_test = split_data(features, labels, seed=seed)
    model = make_model(seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(make_model(seed), X_train, y_train, cv=folds, scoring="r2")
    evaluation = Evaluation(
        r2=float(r2_score(y_test, predictions)),
        mae=float(mean_absolute_error(y_test, predictions)),
        cv_r2_mean=float(np.mean(cv_scores)),
        cv_r2_std=float(np.std(cv_scores, ddof=1)),
    )
    return model, evaluation


def evaluate_scaffold_holdout(
    smiles: Sequence[str], labels: NDArray[np.float64], seed: int = 42
) -> ScaffoldEvaluation:
    """Evaluate generalization to Bemis-Murcko scaffolds absent from training.

    Acyclic molecules share the empty Murcko scaffold and are kept in one group.
    A seeded greedy group selection targets 20% of molecules without splitting
    a scaffold group; exact test size is limited by group sizes.

    Parameters
    ----------
    smiles : sequence of str
        Valid SMILES strings aligned with labels.
    labels : numpy.ndarray
        Finite logP targets.
    seed : int, default=42
        Seed used to select held-out scaffolds and construct the model.

    Returns
    -------
    ScaffoldEvaluation
        R², MAE, test molecule count, and number of held-out scaffolds.

    Raises
    ------
    ValueError
        If inputs are invalid or fewer than three distinct scaffolds exist.
    """
    target = np.asarray(labels, dtype=np.float64)
    if len(smiles) != len(target) or target.ndim != 1 or not np.all(np.isfinite(target)):
        raise ValueError("SMILES and finite one-dimensional labels must have matching lengths")
    if len(smiles) < 10:
        raise ValueError("at least 10 molecules are required")
    groups: list[str] = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {value!r}")
        groups.append(MurckoScaffold.MurckoScaffoldSmiles(mol=molecule))
    if len(set(groups)) < 3:
        raise ValueError("at least three distinct Murcko scaffolds are required")
    features = molecular_features(smiles)
    group_sizes = Counter(groups)
    ordered_groups = list(group_sizes)
    np.random.default_rng(seed).shuffle(ordered_groups)
    ordered_groups.sort(key=group_sizes.get, reverse=True)
    target_test_size = 0.2 * len(smiles)
    selected_groups: set[str] = set()
    selected_count = 0
    for group in ordered_groups:
        size = group_sizes[group]
        if abs(target_test_size - (selected_count + size)) < abs(target_test_size - selected_count):
            selected_groups.add(group)
            selected_count += size
    if not selected_groups or len(selected_groups) == len(group_sizes):
        raise ValueError("could not form nonempty train and scaffold-test partitions")
    test_indices = np.asarray(
        [index for index, group in enumerate(groups) if group in selected_groups], dtype=int
    )
    train_indices = np.asarray(
        [index for index, group in enumerate(groups) if group not in selected_groups], dtype=int
    )
    model = make_model(seed)
    model.fit(features[train_indices], target[train_indices])
    predictions = model.predict(features[test_indices])
    return ScaffoldEvaluation(
        r2=float(r2_score(target[test_indices], predictions)),
        mae=float(mean_absolute_error(target[test_indices], predictions)),
        test_molecules=len(test_indices),
        test_scaffolds=len(set(groups[index] for index in test_indices)),
    )
