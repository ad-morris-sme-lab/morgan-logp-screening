# Theory and scope

## Task and provenance

This project learns the **n-octanol/water partition coefficient (logP)** as a molecular property regression example. logP is dimensionless. Its CSV contains a deterministic, generated set of simple organic SMILES, each labeled by RDKit's implementation of the Wildman-Crippen atom-contribution model (`rdkit.Chem.Crippen.MolLogP`). The committed CSV was generated with RDKit 2026.9.1; `data/generate_dataset.py` recreates it using the installed RDKit version. The labels are computed model outputs, not measured experimental values, and the dataset is not sourced from ChEMBL, PubChem, or an assay database.

The generated set spans short-to-medium alkane chains, alcohols, amines, halides, nitriles, acids, ketones, ethers, esters, and alkyl aromatics. The held-out performance therefore measures how well fingerprints and trees approximate this specific deterministic descriptor over the generated chemical space. It does not establish predictive validity for experimental logP or drug candidates outside that domain.

## Morgan / ECFP4-like representation

For every molecule, RDKit generates a circular Morgan fingerprint with radius 2 and 2048 bits. Atom-centered neighborhoods are iteratively hashed, then folded into a fixed-length binary vector. Radius 2 captures local environments out to four bonds, motivating the common ECFP4-like label. The model appends fourteen transparent features: atom count; C, O, N, F, Cl, Br, and S counts; aromatic atom and ring counts; molecular weight; topological polar surface area; and hydrogen-bond donor and acceptor counts. These structural descriptors help tree models distinguish molecular size and composition, especially for held-out homologues. Hash collisions can merge distinct environments, while binary fingerprints and the simple descriptors omit 3D conformation and explicit long-range context.

## Model and validation

An Extra Trees regressor with 100 randomized decision trees maps the combined fingerprint and descriptor vector to Crippen logP. The twelve candidates are excluded from model fitting and serve as a small out-of-training-set screening list. Of the remaining compounds, we reserve 20% for a deterministic test set, report held-out coefficient of determination (R²) and mean absolute error (MAE), and calculate shuffled five-fold R² on training data only. All random number generators and estimators use fixed seeds. Candidate ranking sorts predicted logP in descending order, which is only a property-ranking demonstration, not a target-specific virtual screen or a claim that more lipophilic molecules are better drugs. Three top-ranked candidate predictions were spot-checked against their generated RDKit labels; this is still not experimental validation.

## Limitations and references

This is a toy cheminformatics/ML pipeline, not an experimental QSAR study. Crippen logP is itself a fragment contribution estimate; reproducing it from fingerprints does not independently validate an experimental property. The candidate list is illustrative and ranking uncertainty is not estimated. Real discovery use would need curated measured labels, scaffold-aware external validation, applicability-domain analysis, uncertainty estimates, and endpoint-specific decision criteria.

- Rogers, D.; Hahn, M. Extended-Connectivity Fingerprints. *J. Chem. Inf. Model.* **2010**, 50, 742-754. https://doi.org/10.1021/ci100050t
- Wildman, S. A.; Crippen, G. M. Prediction of Physicochemical Parameters by Atomic Contributions. *J. Chem. Inf. Comput. Sci.* **1999**, 39, 868-873. https://doi.org/10.1021/ci990307l
