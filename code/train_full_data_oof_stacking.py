"""Fit the final full-data RF+ET+XGBoost genuine-OOF Ridge stack."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from generate_descriptors import build_components, representations


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_configuration() -> tuple[dict, dict]:
    parameters = json.loads(
        (PACKAGE_ROOT / "configs" / "model_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    seeds = json.loads(
        (PACKAGE_ROOT / "configs" / "random_seeds.json").read_text(
            encoding="utf-8"
        )
    )
    return parameters, seeds


def make_model(name: str, specification: dict, seed: int):
    params = specification["parameters"]
    if name == "M06_RF":
        estimator = RandomForestRegressor(**params, random_state=seed, n_jobs=4)
    elif name == "M07_ExtraTrees":
        estimator = ExtraTreesRegressor(**params, random_state=seed, n_jobs=4)
    elif name == "M10_XGBoost":
        estimator = XGBRegressor(
            **params, objective="reg:squarederror", random_state=seed, n_jobs=4
        )
    else:
        raise ValueError(f"Unknown member: {name}")
    return make_pipeline(SimpleImputer(strategy="median"), estimator)


def main(data_path: Path, output_dir: Path) -> None:
    started = time.time()
    parameters, seed_config = load_configuration()
    members = parameters["member_order"]
    if members != ["M06_RF", "M07_ExtraTrees", "M10_XGBoost"]:
        raise RuntimeError("Unexpected final member order")
    data = pd.read_csv(data_path)
    if (
        len(data) != 1396
        or data.canonical_structure_group_id.nunique() != 1396
        or data.canonical_smiles.nunique() != 1396
    ):
        raise RuntimeError("Frozen training data is not 1,396 unique structures")
    y = data.primary_target.to_numpy(float)
    ids = data.canonical_structure_group_id.astype(str).to_numpy()
    components, feature_names = build_components(data.canonical_smiles)
    reps = representations(components)
    repeat_seeds = seed_config["oof_repeat_seeds"]
    offsets = seed_config["member_seed_offsets"]
    repeat_predictions, repeat_fold_ids, audit_rows = [], [], []

    for repeat_index, repeat_seed in enumerate(repeat_seeds):
        matrix = np.full((len(data), len(members)), np.nan, dtype=float)
        fold_ids = np.full(len(data), -1, dtype=int)
        splitter = KFold(n_splits=5, shuffle=True, random_state=repeat_seed)
        for fold, (inner_train, inner_valid) in enumerate(
            splitter.split(np.arange(len(data)))
        ):
            if np.intersect1d(inner_train, inner_valid).size:
                raise RuntimeError("OOF leakage: train/validation overlap")
            fold_ids[inner_valid] = fold
            for member_index, name in enumerate(members):
                specification = parameters["members"][name]
                model_seed = repeat_seed + offsets[name] + fold
                model = make_model(name, specification, model_seed)
                representation = reps[specification["representation"]]
                model.fit(representation[inner_train], y[inner_train])
                matrix[inner_valid, member_index] = model.predict(
                    representation[inner_valid]
                )
        if not np.isfinite(matrix).all() or np.any(fold_ids < 0):
            raise RuntimeError(f"Incomplete OOF predictions for repeat {repeat_index}")
        repeat_predictions.append(matrix)
        repeat_fold_ids.append(fold_ids)
        for row_index, structure_id in enumerate(ids):
            audit_rows.append(
                {
                    "structure_id": structure_id,
                    "repeat": repeat_index,
                    "repeat_seed": repeat_seed,
                    "fold_id": int(fold_ids[row_index]),
                    "validation_prediction_count_per_member": 1,
                    "training_overlap_count": 0,
                }
            )

    averaged_oof = np.mean(np.stack(repeat_predictions, axis=0), axis=0)
    if averaged_oof.shape != (1396, 3) or not np.isfinite(averaged_oof).all():
        raise RuntimeError("Final averaged OOF matrix is incomplete")
    ridge_alpha = float(parameters["meta_learner"]["alpha"])
    ridge = Ridge(alpha=ridge_alpha).fit(averaged_oof, y)

    full_fit_seed = int(seed_config["full_data_base_model_seed"])
    full_models = {}
    for name in members:
        specification = parameters["members"][name]
        model = make_model(name, specification, full_fit_seed)
        model.fit(reps[specification["representation"]], y)
        full_models[name] = model

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_labels = [
        ";".join(
            f"r{repeat}f{repeat_fold_ids[repeat][row]}"
            for repeat in range(len(repeat_seeds))
        )
        for row in range(len(data))
    ]
    meta_table = pd.DataFrame(
        {
            "structure_id": ids,
            "canonical_smiles": data.canonical_smiles.astype(str),
            "experimental_E00_eV": y,
            "fold_id": fold_labels,
            "RF_OOF_prediction_eV": averaged_oof[:, 0],
            "ExtraTrees_OOF_prediction_eV": averaged_oof[:, 1],
            "XGBoost_OOF_prediction_eV": averaged_oof[:, 2],
        }
    )
    meta_path = output_dir / "FINAL_FULLDATA_OOF_META_TRAINING.csv"
    meta_table.to_csv(meta_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(audit_rows).to_csv(
        output_dir / "FULLDATA_OOF_FOLD_AUDIT.csv",
        index=False,
        encoding="utf-8-sig",
    )

    bundle = {
        "model_type": "genuine_repeated_5x5_OOF_Ridge_stacking",
        "members": members,
        "member_specs": parameters["members"],
        "base_models_full_data": full_models,
        "ridge_meta_model": ridge,
        "repeat_seeds": repeat_seeds,
        "seed_offsets": offsets,
        "full_fit_seed": full_fit_seed,
        "training_structure_ids": ids.tolist(),
        "training_data_sha256": sha256(data_path),
        "feature_dimensions": {
            name: int(matrix.shape[1]) for name, matrix in reps.items()
        },
        "feature_names": feature_names,
        "virtual_training_count": 0,
    }
    model_path = output_dir / "FINAL_FULLDATA_GENUINE_OOF_STACKING.joblib"
    joblib.dump(bundle, model_path, compress=3)
    configuration = {
        "n_training_structures": len(data),
        "base_models": members,
        "member_specs": parameters["members"],
        "oof_construction": {
            "n_repeats": 5,
            "n_folds": 5,
            "repeat_seeds": repeat_seeds,
            "seed_offsets": offsets,
            "aggregation": "mean of five genuine OOF predictions per molecule and member",
        },
        "ridge": {
            "alpha": ridge_alpha,
            "coefficients_in_member_order": ridge.coef_.astype(float).tolist(),
            "intercept": float(ridge.intercept_),
        },
        "full_data_refit_seed": full_fit_seed,
        "training_data_sha256": sha256(data_path),
        "oof_meta_training_sha256": sha256(meta_path),
        "model_sha256": sha256(model_path),
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "rdkit": rdkit.__version__,
            "joblib": joblib.__version__,
        },
        "no_virtual_molecule_entered_training": True,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "FINAL_FULLDATA_GENUINE_OOF_STACKING_CONFIG.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(configuration["ridge"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=PACKAGE_ROOT / "data" / "E00_UNIFIED_1396.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "reproduction" / "full_data",
    )
    arguments = parser.parse_args()
    main(arguments.data.resolve(), arguments.output_dir.resolve())
