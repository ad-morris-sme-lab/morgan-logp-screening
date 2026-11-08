"""Train a toy logP model, report screening ranks, and save a parity plot."""

import argparse
from pathlib import Path

from logp_screening.data import candidate_smiles, load_dataset
from logp_screening.fingerprints import molecular_features
from logp_screening.model import split_data, train_model
from logp_screening.screening import rank_candidates
from logp_screening.utils import save_parity_plot


def main() -> None:
    """Run the complete training, evaluation, ranking, and plotting workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, default=Path("logp_parity.png"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    smiles, labels = load_dataset(root / "data" / "logp_toy.csv")
    candidates = candidate_smiles()
    candidate_set = set(candidates)
    training_indices = [index for index, value in enumerate(smiles) if value not in candidate_set]
    training_smiles = [smiles[index] for index in training_indices]
    training_labels = labels[training_indices]
    model, metrics = train_model(training_smiles, training_labels, seed=42)
    features = molecular_features(training_smiles)
    _, X_test, _, y_test = split_data(features, training_labels, seed=42)
    y_pred = model.predict(X_test)
    plot_path = save_parity_plot(y_test, y_pred, args.plot)
    print(f"Dataset molecules: {len(smiles)}")
    print(f"Test R²: {metrics.r2:.3f} | MAE: {metrics.mae:.3f} logP")
    print(f"5-fold train CV R²: {metrics.cv_r2_mean:.3f} ± {metrics.cv_r2_std:.3f}")
    print("\nRanked candidates (higher predicted logP first)")
    print(f"{'Rank':>4}  {'Predicted logP':>14}  SMILES")
    for rank, (candidate, value) in enumerate(rank_candidates(model, candidates), start=1):
        print(f"{rank:>4}  {value:>14.3f}  {candidate}")
    print(f"\nSaved parity plot: {plot_path}")


if __name__ == "__main__":
    main()
