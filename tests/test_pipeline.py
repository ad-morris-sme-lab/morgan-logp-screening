"""Tests for fingerprinting, data handling, modeling, screening, and plotting."""

import csv
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Crippen

from logp_screening.data import candidate_smiles, load_dataset
from logp_screening.fingerprints import molecular_features, morgan_fingerprints
from logp_screening.model import (
    evaluate_scaffold_holdout,
    make_model,
    split_data,
    train_model,
)
from logp_screening.screening import rank_candidates
from logp_screening.utils import save_parity_plot


def test_fingerprints_shape_and_binary_values() -> None:
    """Default fingerprints are binary rows of 2048 bits."""
    result = morgan_fingerprints(["CCO", "c1ccccc1"])
    assert result.shape == (2, 2048)
    assert set(np.unique(result)).issubset({0.0, 1.0})
    assert molecular_features(["CCO"]).shape == (1, 2062)


def test_fingerprint_input_errors() -> None:
    """Empty, malformed, and invalid SMILES inputs are rejected."""
    with pytest.raises(ValueError, match="at least one"):
        morgan_fingerprints([])
    with pytest.raises(ValueError, match="invalid SMILES"):
        morgan_fingerprints(["not_a_smiles"])
    with pytest.raises(ValueError, match="n_bits"):
        morgan_fingerprints(["CCO"], n_bits=0)


def test_csv_loader_checks_schema_and_rows(tmp_path) -> None:
    """Valid CSV loads, while missing columns and invalid labels fail."""
    good = tmp_path / "good.csv"
    with good.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "logp"])
        for i in range(10):
            writer.writerow(["C" * (i + 1), i / 10])
    smiles, labels = load_dataset(good)
    assert len(smiles) == 10 and labels.shape == (10,)
    bad = tmp_path / "bad.csv"
    bad.write_text("smiles,target\nCCO,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        load_dataset(bad)
    invalid = tmp_path / "invalid.csv"
    invalid.write_text(
        "smiles,logp\n" + "".join(f"{'C' * (i + 1)},{i}\n" for i in range(9))
        + "not_a_smiles,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid SMILES"):
        load_dataset(invalid)
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        "smiles,logp\n" + "".join(f"{'C' * (i + 1)},{i}\n" for i in range(9))
        + "C(C),1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate canonical SMILES"):
        load_dataset(duplicate)


def test_split_shapes_and_invalid_inputs() -> None:
    """The split is 80/20 and rejects mismatched shapes."""
    features = np.arange(200, dtype=float).reshape(100, 2)
    labels = np.arange(100, dtype=float)
    X_train, X_test, y_train, y_test = split_data(features, labels)
    assert X_train.shape == (80, 2) and X_test.shape == (20, 2)
    assert len(y_train) == 80 and len(y_test) == 20
    with pytest.raises(ValueError, match="matching rows"):
        split_data(features, labels[:-1])


def test_model_and_five_fold_cv_exceed_metric_threshold() -> None:
    """Model has strong deterministic holdout fit to the generated logP target."""
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "logp_toy.csv"
    smiles, labels = load_dataset(dataset_path)
    excluded = set(candidate_smiles())
    indices = [i for i, value in enumerate(smiles) if value not in excluded]
    model, scores = train_model([smiles[i] for i in indices], labels[indices], seed=42)
    assert scores.r2 > 0.5
    assert scores.mae >= 0.0
    assert np.isfinite([scores.cv_r2_mean, scores.cv_r2_std]).all()
    assert model.n_features_in_ == 2062


def test_scaffold_holdout_keeps_scaffolds_separate_and_reports_metrics() -> None:
    """Scaffold-aware evaluation runs across generated ring-system families."""
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "logp_toy.csv"
    smiles, labels = load_dataset(dataset_path)
    excluded = set(candidate_smiles())
    indices = [i for i, value in enumerate(smiles) if value not in excluded]
    result = evaluate_scaffold_holdout(
        [smiles[i] for i in indices], labels[indices], seed=42
    )
    assert result.test_molecules > 0
    assert result.test_scaffolds >= 1
    assert np.isfinite([result.r2, result.mae]).all()


def test_scaffold_holdout_rejects_too_few_scaffolds() -> None:
    """A scaffold score is refused when a valid grouped split is impossible."""
    smiles = ["C" * count for count in range(1, 11)]
    with pytest.raises(ValueError, match="three distinct Murcko scaffolds"):
        evaluate_scaffold_holdout(smiles, np.arange(10, dtype=float))


def test_screening_ranks_highest_prediction_first_and_rejects_unfitted() -> None:
    """Candidates are returned in descending predicted logP order."""
    model = make_model()
    with pytest.raises(ValueError, match="fitted"):
        rank_candidates(model, ["CCO"])
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "logp_toy.csv"
    smiles, labels = load_dataset(dataset_path)
    candidates = candidate_smiles()
    excluded = set(candidates)
    indices = [i for i, value in enumerate(smiles) if value not in excluded]
    model.fit(molecular_features([smiles[i] for i in indices]), labels[indices])
    ranked = rank_candidates(model, candidates)
    assert len(ranked) == len(candidates)
    assert all(ranked[i][1] >= ranked[i + 1][1] for i in range(len(ranked) - 1))
    assert all(Chem.MolFromSmiles(item[0]) is not None for item in ranked)
    label_by_smiles = dict(zip(smiles, labels, strict=True))
    assert all(abs(score - label_by_smiles[smiles_value]) < 0.6
               for smiles_value, score in ranked[:3])
    with pytest.raises(ValueError, match="at least one"):
        rank_candidates(model, [])


def test_plot_is_written_and_rejects_bad_shapes(tmp_path) -> None:
    """Plotting handles constant values, writes a file, and validates shapes."""
    target = save_parity_plot([0.0, 1.0], [0.1, 0.9], tmp_path / "fig" / "plot.png")
    assert target.is_file() and target.stat().st_size > 100
    constant = save_parity_plot([1.0], [1.0], tmp_path / "constant.png")
    assert constant.is_file() and constant.stat().st_size > 100
    with pytest.raises(ValueError, match="equal length"):
        save_parity_plot([1.0], [1.0, 2.0], tmp_path / "bad.png")


def test_generated_dataset_contains_unique_molecules_and_rdkit_labels(tmp_path) -> None:
    """Generator writes at least 100 unique structures with known labels."""
    from data.generate_dataset import write_dataset

    path = tmp_path / "toy.csv"
    count = write_dataset(path)
    smiles, labels = load_dataset(path)
    assert count >= 100 and len(set(smiles)) == count
    metadata = path.with_suffix(".metadata.json")
    assert metadata.is_file()
    assert '"labels_are_experimental": false' in metadata.read_text(encoding="utf-8")
    molecule = Chem.MolFromSmiles(smiles[0])
    assert labels[0] == pytest.approx(Crippen.MolLogP(molecule), abs=1e-7)
