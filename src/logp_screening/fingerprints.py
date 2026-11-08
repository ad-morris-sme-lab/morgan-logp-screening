"""Molecular fingerprint generation."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


def morgan_fingerprints(
    smiles: Sequence[str], radius: int = 2, n_bits: int = 2048
) -> NDArray[np.float64]:
    """Convert SMILES strings to binary Morgan fingerprints.

    Parameters
    ----------
    smiles : sequence of str
        Valid, non-empty molecular SMILES.
    radius : int, default=2
        Morgan neighborhood radius. Radius two corresponds to ECFP4-like
        circular environments.
    n_bits : int, default=2048
        Length of each folded binary fingerprint.

    Returns
    -------
    numpy.ndarray
        Fingerprint matrix of shape ``(len(smiles), n_bits)``.

    Raises
    ------
    ValueError
        If parameters are invalid or any SMILES cannot be parsed.
    """
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a non-negative integer")
    if isinstance(n_bits, bool) or not isinstance(n_bits, int) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer")
    if len(smiles) == 0:
        raise ValueError("at least one SMILES string is required")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows: list[NDArray[np.float64]] = []
    for value in smiles:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("SMILES entries must be non-empty strings")
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {value!r}")
        rows.append(np.asarray(generator.GetFingerprint(molecule), dtype=np.float64))
    return np.vstack(rows)


def molecular_features(smiles: Sequence[str]) -> NDArray[np.float64]:
    """Combine Morgan bits with simple composition and size/polarity features.

    Parameters
    ----------
    smiles : sequence of str
        Valid, non-empty molecular SMILES.

    Returns
    -------
    numpy.ndarray
        A matrix with 2048 Morgan bits followed by 14 scalar features:
        atom count, C/O/N/F/Cl/Br/S counts, aromatic atom count, ring count,
        molecular weight, topological polar surface area, H-bond acceptors,
        and H-bond donors.
    """
    fingerprints = morgan_fingerprints(smiles)
    descriptor_rows: list[list[float]] = []
    elements = ("C", "O", "N", "F", "Cl", "Br", "S")
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"invalid SMILES: {value!r}")
        symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        descriptor_rows.append([
            float(molecule.GetNumAtoms()),
            *(float(symbols.count(element)) for element in elements),
            float(sum(atom.GetIsAromatic() for atom in molecule.GetAtoms())),
            float(molecule.GetRingInfo().NumRings()),
            float(Descriptors.MolWt(molecule)),
            float(Descriptors.TPSA(molecule)),
            float(Descriptors.NumHAcceptors(molecule)),
            float(Descriptors.NumHDonors(molecule)),
        ])
    return np.hstack((fingerprints, np.asarray(descriptor_rows, dtype=np.float64)))
