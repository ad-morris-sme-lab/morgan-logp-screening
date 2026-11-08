"""CSV dataset loading and candidate structures."""

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem


def load_dataset(path: str | Path) -> tuple[list[str], NDArray[np.float64]]:
    """Load and validate the project's SMILES/logP CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        CSV with ``smiles`` and ``logp`` columns.

    Returns
    -------
    tuple of list[str], numpy.ndarray
        SMILES strings and logP labels in matching row order.

    Raises
    ------
    ValueError
        If columns, rows, or numeric labels are invalid.
    """
    molecules: list[str] = []
    labels: list[float] = []
    seen_smiles: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"smiles", "logp"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain smiles and logp columns")
        for line, row in enumerate(reader, start=2):
            smiles = (row.get("smiles") or "").strip()
            try:
                label = float(row["logp"] or "nan")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid logp value on CSV line {line}") from exc
            if not smiles or not np.isfinite(label):
                raise ValueError(f"invalid molecule or logp value on CSV line {line}")
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"invalid SMILES on CSV line {line}: {smiles!r}")
            canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
            if canonical_smiles in seen_smiles:
                raise ValueError(f"duplicate canonical SMILES on CSV line {line}: {canonical_smiles!r}")
            seen_smiles.add(canonical_smiles)
            molecules.append(canonical_smiles)
            labels.append(label)
    if len(molecules) < 10:
        raise ValueError("dataset must contain at least 10 rows")
    return molecules, np.asarray(labels, dtype=np.float64)


def candidate_smiles() -> list[str]:
    """Return twelve held-aside candidates from the generated chemical families.

    Returns
    -------
    list of str
        Candidate SMILES used by the runnable demonstration.
    """
    return [
        "CCC", "CCCCCC", "CCCCCCCCCCCC",
        "CCO", "CCCCCCO", "CCCCCCCCCCCCO",
        "CCCl", "CCCCCCCl",
        "CCc1ccccc1", "CCOc1ccccc1", "CCCCCCc1ccccc1",
        "CCCC(=O)O",
    ]
