# MR-TADF E0,0 Machine-Learning Dataset and OOF Stacking Model

## Project

This package contains the machine-learning dataset, fixed split assignments,
model code, configuration, and frozen evaluation results associated with:

*Machine Learning-Guided Discovery of MR-TADF Photocatalysts: Double Gain of
Thermodynamics and Kinetics*.

It is a local publication package. It contains no Git history, credentials,
virtual-library structures, candidate structures, manuscript files, or trained
model binaries.

## Dataset

`data/E00_UNIFIED_1396.csv` contains 1,396 unique canonical structures and the
E0,0 target in eV (`primary_target`):

- 133 MR-TADF structures: 130 MR-TADF-only plus 3 cross-domain overlaps;
- 1,263 non-overlapping general chromophores;
- 1,396 non-missing targets and 1,396 unique canonical SMILES.

Source and reference fields present in the frozen table are retained unchanged.

## Final model

The final established model is:

`Random Forest + Extra Trees + XGBoost -> Ridge`

The Ridge meta-learner was trained using genuine out-of-fold predictions rather
than fitted predictions obtained from the same training samples. Each molecule
receives one held-out prediction per member in each of five shuffled five-fold
repeats; the five OOF predictions are averaged before fitting Ridge with
`alpha = 1.0`.

The frozen 20-split comparison gives mean MAE 0.116661 eV, RMSE 0.174031 eV,
and R2 0.865196. The two-member RF+ET architecture has the lowest mean MAE,
while the established three-member architecture satisfies the prespecified
near-best criterion and retains the lower RMSE and higher R2 reported in the
manuscript workflow. DNN is not part of the final model or its configuration.

## Validation

- 20 predefined structure-level outer splits;
- 1,116 training and 280 test molecules per split;
- zero canonical-structure overlap between training and test;
- five repeats of five-fold genuine OOF CV inside each outer-training set;
- no outer-test sample used for imputation, fitting, or meta-learning.

The `seed` column in `splits/outer_split_assignments.csv` is authoritative. The
historical `split_id` column preserves the original panel numbering; the public
seed order is recorded in `configs/random_seeds.json`.

## Molecular representations

- Random Forest: Morgan radius 2 (2,048 bits, chirality enabled) + MACCS
  (167 bits), total 2,215 features.
- Extra Trees and XGBoost: the same 2,215 fingerprint features + 217 RDKit 2D
  descriptors + 21 topology/ring descriptors, total 2,453 features.
- Median imputation is fitted independently inside each model training fit.
  Tree models are not scaled, and no variance, correlation, or supervised
  feature filter is applied to the final representations.

## Folder structure

- `data/`: frozen 1,396-structure dataset.
- `splits/`: all 20 outer train/test assignments.
- `configs/`: descriptor definitions, model parameters, and random seeds.
- `code/`: descriptor generation, genuine-OOF training, outer-test evaluation,
  and package verification.
- `results/`: frozen architecture summary and per-split metrics.

## Reproduction

The frozen environment used Python 3.11.13. Install the listed dependencies in
an isolated environment, then run from the package root:

```text
python code/verify_package.py
python code/generate_descriptors.py
python code/evaluate_outer_splits.py
python code/train_full_data_oof_stacking.py
```

The last three commands write only under `reproduction/`. They are included for
reproduction and were not run while assembling this package. Full retraining is
computationally expensive.

## Provenance and integrity

`SOURCE_CODE_CORRESPONDENCE.md` maps publication scripts to the actual frozen
workflow sources. `SOURCE_FILE_MAP.csv`, `QA_REPORT.md`, and `SHA256SUMS.txt`
record file origins, validation, exclusions, and package hashes.

## Licensing

Code is under the MIT License. Data, results, and documentation are under CC BY
4.0, with Deep4Chem attribution and modification notices retained in
`THIRD_PARTY_NOTICES.md`.
