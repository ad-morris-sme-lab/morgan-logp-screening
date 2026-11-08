"""Generate a deterministic set of small molecules labeled with RDKit Crippen logP."""

import argparse
import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Crippen


def generated_smiles() -> list[str]:
    """Enumerate a diverse, deterministic set of simple organic structures.

    Returns
    -------
    list of str
        Unique canonical SMILES for the generated compound families.
    """
    proposals: list[str] = []
    for count in range(1, 19):
        chain = "C" * count
        proposals.extend((chain, chain + "O", chain + "N", chain + "F", chain + "Cl", chain + "Br"))
        if count >= 2:
            proposals.extend((chain[:-1] + "=C", chain + "#N", chain + "(=O)O", chain + "(=O)C"))
        proposals.extend((chain + "c1ccccc1", chain + "Oc1ccccc1"))
    for first in range(1, 11):
        for second in range(1, 11):
            proposals.extend(("C" * first + "O" + "C" * second,
                              "C" * first + "(=O)O" + "C" * second))
    canonical: set[str] = set()
    for smiles in proposals:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            canonical.add(Chem.MolToSmiles(molecule, canonical=True))
    return sorted(canonical)


def write_dataset(path: str | Path) -> int:
    """Write generated structures and RDKit Crippen logP labels to CSV.

    Parameters
    ----------
    path : str or pathlib.Path
        Output CSV path.

    Returns
    -------
    int
        Number of unique molecules written.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    smiles_values = generated_smiles()
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("smiles", "logp"))
        for smiles in smiles_values:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise RuntimeError(f"generated invalid SMILES: {smiles}")
            writer.writerow((smiles, f"{Crippen.MolLogP(molecule):.8f}"))
    return len(smiles_values)


def main() -> None:
    """Generate the CSV at a caller-selected destination."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("logp_toy.csv"))
    args = parser.parse_args()
    print(f"Wrote {write_dataset(args.output)} molecules to {args.output}")


if __name__ == "__main__":
    main()
