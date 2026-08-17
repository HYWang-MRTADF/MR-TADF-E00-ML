# Source-code correspondence

The publication scripts use package-relative paths and exclude unrelated
screening and figure-generation functions. Descriptor definitions, model
parameters, split assignments, OOF procedure, Ridge fitting, and metrics are
unchanged.

| Publication script | Frozen workflow source | Relationship |
|---|---|---|
| `code/generate_descriptors.py` | `outputs/review_revision_v4/phase5i_mrtadf133_seed41_independent_20split/src/panel_local_features.py` | Exact `topology_proxy`, `build_components`, and representation definitions extracted; virtual-library prediction code omitted. |
| `code/train_full_data_oof_stacking.py` | `outputs/review_revision_v4/phase5c_stacking_6540_prediction/src/02_build_full_data_oof_stacking.py` | Same three members, parameters, 5x5 OOF procedure, Ridge alpha, seed offsets, and full-data refit; inputs changed to package-relative files. |
| `code/evaluate_outer_splits.py` | `outputs/review_revision_v4/phase5i_mrtadf133_seed41_independent_20split/src/run_panelA_seed41_independent.py` and `outputs/review_revision_v4/final_ml_model_descriptor_validation/src/run_model_architecture_comparison.py` | Final three-member outer-test path isolated; virtual inference and DNN comparison omitted. |
| `code/verify_package.py` | Package QA implementation | Read-only publication-package checks; no scientific computation. |

The historical RF+XGBoost+DNN stack is not configured or trained by the
publication scripts.
