# Publication package QA report

Overall package QA: **PASS**

## Dataset QA

- Frozen source SHA256 preserved: **PASS**
- Rows: **1,396 — PASS**
- Unique canonical structure IDs: **1,396 — PASS**
- Unique canonical SMILES: **1,396 — PASS**
- Non-missing `primary_target` values: **1,396 — PASS**
- General-only chromophores: **1,263 — PASS**
- MR-TADF-only structures: **130 — PASS**
- Cross-domain overlaps retained once: **3 — PASS**
- Total MR-TADF structures (130 + 3): **133 — PASS**
- E0,0, SMILES, references, and other source values modified: **NO — PASS**

## Split QA

- Outer seeds present: **20/20 — PASS**
- Aggregate assignment rows: **27,920 — PASS**
- Each seed has 1,396 assignments: **PASS**
- Each seed has 1,116 train and 280 test assignments: **PASS**
- Dataset IDs and canonical SMILES match for every seed: **PASS**
- Train/test canonical structure overlap for every seed: **0 — PASS**
- Each individual source split equals its aggregate block: **PASS**
- Test IDs equal frozen outer-test checkpoint IDs for every seed: **PASS**
- Train IDs equal frozen genuine-OOF checkpoint IDs for every seed: **PASS**

## Descriptor QA

The publication descriptor script was run with the frozen environment (Python
3.11.13, RDKit 2025.03.3) against all 1,396 structures and compared with the
frozen `FEATURE_COMPONENTS_1396.npz` cache.

- Morgan radius-2, 2,048 bits: exact array match; maximum difference **0 — PASS**
- MACCS, 167 bits: exact array match; maximum difference **0 — PASS**
- RDKit 2D, 217 descriptors: exact array and NaN-pattern match; maximum difference **0 — PASS**
- Topology/ring, 21 descriptors: exact array and NaN-pattern match; maximum difference **0 — PASS**
- Feature names and order: exact match — **PASS**
- RF dimension 2,215: **PASS**
- ET/XGBoost dimension 2,453: **PASS**

## Model QA

- Final architecture is Random Forest + Extra Trees + XGBoost -> Ridge: **PASS**
- DNN is absent from the final publication configuration and training code: **PASS**
- Ridge alpha = 1.0: **PASS**
- Genuine OOF procedure = five repeats x five folds: **PASS**
- OOF repeat seeds = 42, 73, 101, 314, 777: **PASS**
- Ridge input is averaged held-out predictions, not fitted predictions: **PASS**
- Median imputation is inside each model training pipeline: **PASS**
- No outer-test samples enter imputation, base fitting, or meta-learning: **PASS**
- No model retraining or molecular prediction was run during package assembly: **PASS**

## Results QA

- Frozen result files copied byte-for-byte: **PASS**
- Final architecture rows in per-split table: **20 — PASS**
- Summary recomputed from the copied per-split rows: **exact match — PASS**
- Mean MAE: **0.11666084988629699 eV — PASS**
- Mean RMSE: **0.17403083165137 eV — PASS**
- Mean R2: **0.865195926254162 — PASS**
- Prespecified near-best status: **True — PASS**

The RF+ET architecture has the lowest mean MAE. The established RF+ET+XGBoost
architecture ranks fifth by mean MAE but passes the prespecified near-best rule
and has lower mean RMSE and higher mean R2 than RF+ET. This distinction is
disclosed in the README and was not altered.

## Requirements QA

- Exact frozen versions recovered for Python, NumPy, pandas, scikit-learn,
  XGBoost, RDKit, and joblib: **PASS**
- SciPy included for the published evaluation metrics: **PASS**
- Third-party packages vendored: **NO — PASS**

## Security and scope QA

- Windows absolute paths in public `.py`, `.json`, `.md`, `.txt`, `.csv`: **0 — PASS**
- Passwords, tokens, API keys, secrets, or credentials detected: **0 — PASS**
- `.env` files: **0 — PASS**
- Trained `.joblib` models: **0 — PASS**
- Virtual-library structures or predictions: **0 — PASS**
- Candidate structures, SDF/MOL/ChemDraw files: **0 — PASS**
- Manuscript, Supporting Information Word, PPT, screenshots, caches: **0 — PASS**
- Large precomputed feature matrices included: **NO — PASS**

## License QA

- Local project records identify the Deep4Chem/Figshare source as CC BY 4.0: **PASS**
- Deep4Chem article DOI and dataset DOI retained: **PASS**
- Attribution and modification notice retained: **PASS**
- Existing author-confirmed MIT code and CC BY 4.0 data/documentation choices retained: **PASS**

## Publication decision

No unresolved scientific, security, path, dependency, or redistribution blocker
was found in this local package.

**READY FOR MANUAL GITHUB UPLOAD: YES**

No GitHub connection, `git` command, `gh` command, remote-service operation, or
upload was performed.
