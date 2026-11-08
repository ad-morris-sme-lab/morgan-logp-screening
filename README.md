# Morgan logP Screening

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/) [![GitHub Actions workflow](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](.github/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this project does

This project turns SMILES into 2048-bit Morgan radius-2 fingerprints plus 14 transparent size, composition, and polarity features, fits a reproducible Extra Trees regressor to logP labels, evaluates it, then ranks an unseen candidate set by predicted logP. It is Project 3 in a computational-chemistry series, moving from Lennard-Jones cluster energies and optimization into molecular representations and ligand-based ML.

**Chosen task:** logP regression is a compact way to demonstrate the full molecule-to-ranking pipeline with a continuous endpoint and interpretable regression metrics. The training labels are generated locally with RDKit's Wildman-Crippen logP implementation, allowing the repository to be self-contained while clearly separating a cheminformatics demonstration from an experimentally validated QSAR model.

## Abstract

This project demonstrates a reproducible ligand-based molecular-property pipeline that maps SMILES to molecular features, fits a regression model, evaluates predictions, and ranks an unseen candidate set. RDKit generates 2048-bit radius-2 Morgan fingerprints and fourteen size, composition, and polarity descriptors. An Extra Trees regressor predicts n-octanol/water partition coefficient (logP), with an 80/20 holdout, five-fold cross-validation, R², and MAE. The bundled dataset contains 367 deterministic organic structures labeled using RDKit's Wildman-Crippen implementation. The twelve screening candidates are excluded from fitting. These computed labels are synthetic descriptor targets rather than experimental measurements, so performance demonstrates workflow behavior within a generated chemical space and does not establish assay-level predictive validity.

## How to run locally

Use a fresh Conda environment with Python 3.11. The model runs on CPU only; no GPU is required.

```bash
conda create -n morgan-logp-screening python=3.11 pip -y
conda activate morgan-logp-screening
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python examples/train_and_screen.py --plot artifacts/logp_parity.png
```

Recreate the deterministic toy dataset, if needed:

```bash
python data/generate_dataset.py
```

The example prints holdout and five-fold training metrics, a ranked candidate table, and saves the observed-versus-predicted plot. It runs on CPU and does not require a GPU or network access.

## Portfolio blurb

Built a reproducible ligand-based screening pipeline that turns SMILES into Morgan fingerprints and molecular descriptors, predicts logP, and ranks held-out molecules. The project includes deterministic data generation, cross-validation, a test suite, and a CPU-only workflow, with clear documentation that its labels are computed descriptors rather than experimental drug-discovery results.

## Example output

Example output from the bundled 367-molecule dataset using the fixed seeds and 3.15 dependency pins. The twelve ranked candidates are excluded from model fitting.

```text
Dataset molecules: 367
Test R²: 0.986 | MAE: 0.158 logP
5-fold train CV R²: 0.987 ± 0.005

Ranked candidates (higher predicted logP first)
Rank  Predicted logP  SMILES
   1           4.707  CCCCCCCCCCCC
   2           4.286  CCCCCCc1ccccc1
   3           4.012  CCCCCCCCCCCCO
```

The complete ranked table is produced by the example. The top-three structures are valid held-out molecules, and their predicted values are within 0.5 logP unit of their generated Crippen labels in this run. Higher predicted logP means greater predicted lipophilicity in this toy endpoint, not higher drug quality or target affinity. Predictions can vary slightly across supported Python-version dependency pins.

## Dataset and limitations

`data/logp_toy.csv` contains more than 100 unique generated molecules and deterministic RDKit Crippen logP labels. These are computational descriptor values, not measurements from ChEMBL, PubChem, or experimental assays. A model trained against this dataset demonstrates representation learning and ranking mechanics only; experimental drug-discovery claims require curated assay data, scaffold-aware validation, and applicability-domain checks. See [theory and limitations](docs/theory.md).

## License

MIT, see [LICENSE](LICENSE).

## Cite this work

```bibtex
@software{morris2026morgan_logp_screening,
  author  = {Morris, Youjah},
  title   = {Morgan logP Screening: A Reproducible Molecular Regression Pipeline},
  year    = {2026},
  url     = {https://github.com/ad-morris-sme-lab/morgan-logp-screening},
  note    = {Toy labels computed using RDKit Wildman-Crippen logP}
}
```

Fingerprint method: Rogers and Hahn, *J. Chem. Inf. Model.* 2010, [doi:10.1021/ci100050t](https://doi.org/10.1021/ci100050t). Label model: Wildman and Crippen, *J. Chem. Inf. Comput. Sci.* 1999, [doi:10.1021/ci990307l](https://doi.org/10.1021/ci990307l).

## Push to GitHub

First create a new **empty public repository** on GitHub named `morgan-logp-screening`, without initializing it with a README, license, or `.gitignore`. Then run these commands from the project root:

```bash
git init -b main
git add .
git commit -m "Initial commit: Morgan logP screening pipeline"
git remote add origin https://github.com/ad-morris-sme-lab/morgan-logp-screening.git
git push -u origin main
```
