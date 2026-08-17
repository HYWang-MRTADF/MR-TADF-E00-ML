"""Evaluate the final RF+ET+XGBoost genuine-OOF stack on fixed outer splits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from generate_descriptors import build_components, representations
from train_full_data_oof_stacking import make_model


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(observed, predicted)),
        "RMSE": float(mean_squared_error(observed, predicted) ** 0.5),
        "R2": float(r2_score(observed, predicted)),
        "Pearson_r": float(pearsonr(observed, predicted)[0]),
        "Spearman_r": float(spearmanr(observed, predicted)[0]),
    }


def main(data_path: Path, split_path: Path, output_dir: Path, selected_seeds) -> None:
    parameters = json.loads(
        (PACKAGE_ROOT / "configs" / "model_parameters.json").read_text(
            encoding="utf-8"
        )
    )
    seed_config = json.loads(
        (PACKAGE_ROOT / "configs" / "random_seeds.json").read_text(
            encoding="utf-8"
        )
    )
    expected_seeds = seed_config["outer_split_seeds"]
    seeds = expected_seeds if selected_seeds is None else selected_seeds
    if any(seed not in expected_seeds for seed in seeds):
        raise RuntimeError("Requested seed is not in the frozen outer-seed list")
    data = pd.read_csv(data_path)
    assignments = pd.read_csv(split_path)
    if len(data) != 1396 or data.canonical_smiles.nunique() != 1396:
        raise RuntimeError("Dataset identity check failed")
    components, _ = build_components(data.canonical_smiles)
    reps = representations(components)
    y = data.primary_target.to_numpy(float)
    ids = data.canonical_structure_group_id.astype(str).to_numpy()
    id_to_index = {value: index for index, value in enumerate(ids)}
    members = parameters["member_order"]
    repeat_seeds = seed_config["oof_repeat_seeds"]
    offsets = seed_config["member_seed_offsets"]
    ridge_alpha = float(parameters["meta_learner"]["alpha"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows = []

    for seed in seeds:
        block = assignments[assignments.seed.eq(seed)]
        train_ids = block.loc[
            block.assignment.eq("train"), "canonical_structure_group_id"
        ].astype(str)
        test_ids = block.loc[
            block.assignment.eq("test"), "canonical_structure_group_id"
        ].astype(str)
        if len(block) != 1396 or len(train_ids) != 1116 or len(test_ids) != 280:
            raise RuntimeError(f"Invalid split counts for seed {seed}")
        if set(train_ids) & set(test_ids):
            raise RuntimeError(f"Outer train/test overlap for seed {seed}")
        train_idx = np.sort([id_to_index[value] for value in train_ids])
        test_idx = np.sort([id_to_index[value] for value in test_ids])
        repeated = np.zeros((len(train_idx), len(members)), dtype=float)
        coverage = np.zeros((len(train_idx), len(repeat_seeds)), dtype=np.int8)

        for repeat, repeat_seed in enumerate(repeat_seeds):
            matrix = np.full((len(train_idx), len(members)), np.nan)
            splitter = KFold(n_splits=5, shuffle=True, random_state=repeat_seed)
            for fold, (inner_train, inner_valid) in enumerate(
                splitter.split(train_idx)
            ):
                coverage[inner_valid, repeat] += 1
                for member_index, name in enumerate(members):
                    specification = parameters["members"][name]
                    model_seed = repeat_seed + offsets[name] + fold
                    model = make_model(name, specification, model_seed)
                    representation = reps[specification["representation"]]
                    model.fit(
                        representation[train_idx[inner_train]],
                        y[train_idx[inner_train]],
                    )
                    matrix[inner_valid, member_index] = model.predict(
                        representation[train_idx[inner_valid]]
                    )
            if not np.isfinite(matrix).all():
                raise RuntimeError(f"Incomplete OOF matrix for seed {seed}")
            repeated += matrix
        if not (coverage == 1).all():
            raise RuntimeError(f"OOF coverage check failed for seed {seed}")
        averaged_oof = repeated / len(repeat_seeds)
        ridge = Ridge(alpha=ridge_alpha).fit(averaged_oof, y[train_idx])

        base_predictions = {}
        for name in members:
            specification = parameters["members"][name]
            model = make_model(name, specification, seed)
            representation = reps[specification["representation"]]
            model.fit(representation[train_idx], y[train_idx])
            base_predictions[name] = model.predict(representation[test_idx])
        base_matrix = np.column_stack(
            [base_predictions[name] for name in members]
        )
        stacking = ridge.predict(base_matrix)
        row = {
            "outer_seed": seed,
            "architecture": "RF + ET + XGBoost",
            "train_n": len(train_idx),
            "test_n": len(test_idx),
            "ridge_alpha": ridge_alpha,
            "coefficient_RF": float(ridge.coef_[0]),
            "coefficient_ET": float(ridge.coef_[1]),
            "coefficient_XGBoost": float(ridge.coef_[2]),
            "ridge_intercept": float(ridge.intercept_),
            **metrics(y[test_idx], stacking),
        }
        metric_rows.append(row)
        pd.DataFrame(
            {
                "outer_seed": seed,
                "canonical_structure_group_id": ids[test_idx],
                "canonical_smiles": data.canonical_smiles.iloc[test_idx].to_numpy(),
                "observed_E00": y[test_idx],
                "RF_prediction_eV": base_predictions["M06_RF"],
                "ET_prediction_eV": base_predictions["M07_ExtraTrees"],
                "XGBoost_prediction_eV": base_predictions["M10_XGBoost"],
                "stacking_prediction_eV": stacking,
            }
        ).to_csv(
            output_dir / f"outer_test_predictions_seed_{seed}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"seed {seed}: MAE={row['MAE']:.6f} eV")

    per_split = pd.DataFrame(metric_rows)
    per_split.to_csv(
        output_dir / "model_architecture_per_split_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "architecture": "RF + ET + XGBoost",
        "n_splits": len(per_split),
        "MAE_mean": float(per_split.MAE.mean()),
        "MAE_sample_SD": float(per_split.MAE.std(ddof=1)),
        "RMSE_mean": float(per_split.RMSE.mean()),
        "RMSE_sample_SD": float(per_split.RMSE.std(ddof=1)),
        "R2_mean": float(per_split.R2.mean()),
        "R2_sample_SD": float(per_split.R2.std(ddof=1)),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / "model_architecture_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=PACKAGE_ROOT / "data" / "E00_UNIFIED_1396.csv",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=PACKAGE_ROOT / "splits" / "outer_split_assignments.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PACKAGE_ROOT / "reproduction" / "outer_splits",
    )
    parser.add_argument(
        "--seeds",
        help="Optional comma-separated subset of frozen outer seeds",
    )
    arguments = parser.parse_args()
    parsed_seeds = (
        None
        if not arguments.seeds
        else [int(value) for value in arguments.seeds.split(",")]
    )
    main(
        arguments.data.resolve(),
        arguments.splits.resolve(),
        arguments.output_dir.resolve(),
        parsed_seeds,
    )
