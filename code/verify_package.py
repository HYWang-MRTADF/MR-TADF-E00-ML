"""Read-only integrity and scientific-contract checks for the public package."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SEEDS = [
    7, 19, 31, 41, 55, 73, 89, 101, 131, 157,
    202, 271, 314, 401, 512, 613, 777, 888, 1024, 2026,
]


def add(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "pass": bool(passed), "detail": detail})


def main() -> int:
    checks: list[dict] = []
    data = pd.read_csv(ROOT / "data" / "E00_UNIFIED_1396.csv")
    add(checks, "dataset_rows", len(data) == 1396, f"rows={len(data)}")
    add(
        checks,
        "unique_canonical_ids",
        data.canonical_structure_group_id.nunique() == 1396,
        f"unique={data.canonical_structure_group_id.nunique()}",
    )
    add(
        checks,
        "unique_canonical_smiles",
        data.canonical_smiles.nunique() == 1396,
        f"unique={data.canonical_smiles.nunique()}",
    )
    add(
        checks,
        "target_complete",
        data.primary_target.notna().all(),
        f"nonmissing={data.primary_target.notna().sum()}",
    )
    counts = data.source_domain_status.value_counts().to_dict()
    add(
        checks,
        "domain_counts",
        counts == {"general_only": 1263, "MR_TADF_only": 130, "both_sources": 3},
        str(counts),
    )

    splits = pd.read_csv(ROOT / "splits" / "outer_split_assignments.csv")
    add(
        checks,
        "outer_seed_set",
        sorted(splits.seed.unique().tolist()) == EXPECTED_SEEDS,
        str(sorted(splits.seed.unique().tolist())),
    )
    split_failures = []
    dataset_ids = set(data.canonical_structure_group_id.astype(str))
    for seed in EXPECTED_SEEDS:
        block = splits[splits.seed.eq(seed)]
        train = block[block.assignment.eq("train")]
        test = block[block.assignment.eq("test")]
        reasons = []
        if len(block) != 1396 or len(train) != 1116 or len(test) != 280:
            reasons.append(f"counts={len(block)}/{len(train)}/{len(test)}")
        if set(block.canonical_structure_group_id.astype(str)) != dataset_ids:
            reasons.append("dataset ID mismatch")
        if block.canonical_smiles.nunique() != 1396:
            reasons.append("canonical SMILES not unique")
        if set(train.canonical_structure_group_id) & set(test.canonical_structure_group_id):
            reasons.append("structure overlap")
        if set(train.canonical_smiles) & set(test.canonical_smiles):
            reasons.append("canonical SMILES overlap")
        if reasons:
            split_failures.append(f"seed {seed}: {', '.join(reasons)}")
    add(
        checks,
        "all_outer_splits",
        not split_failures,
        "20 x (1116 train / 280 test), overlap=0" if not split_failures else "; ".join(split_failures),
    )

    model = json.loads(
        (ROOT / "configs" / "model_parameters.json").read_text(encoding="utf-8")
    )
    add(
        checks,
        "final_architecture",
        model["member_order"] == ["M06_RF", "M07_ExtraTrees", "M10_XGBoost"],
        model["final_architecture"],
    )
    add(
        checks,
        "ridge_alpha",
        model["meta_learner"]["alpha"] == 1.0,
        f"alpha={model['meta_learner']['alpha']}",
    )
    add(
        checks,
        "genuine_oof_declaration",
        "genuine" in model["meta_learner"]["training_input"].lower(),
        model["meta_learner"]["training_input"],
    )
    descriptor = json.loads(
        (ROOT / "configs" / "descriptor_config.json").read_text(encoding="utf-8")
    )
    add(
        checks,
        "descriptor_dimensions",
        descriptor["representations"]["F05_Morgan_r2_MACCS"]["dimension"] == 2215
        and descriptor["representations"]["F10_all"]["dimension"] == 2453,
        "RF=2215; ET/XGBoost=2453",
    )
    seeds = json.loads(
        (ROOT / "configs" / "random_seeds.json").read_text(encoding="utf-8")
    )
    add(
        checks,
        "oof_settings",
        seeds["oof_repeats"] == 5
        and seeds["oof_folds"] == 5
        and seeds["oof_repeat_seeds"] == [42, 73, 101, 314, 777],
        f"{seeds['oof_repeats']} repeats x {seeds['oof_folds']} folds",
    )

    summary = pd.read_csv(ROOT / "results" / "model_architecture_summary.csv")
    per_split = pd.read_csv(
        ROOT / "results" / "model_architecture_per_split_metrics.csv"
    )
    final = summary[summary.architecture.eq("RF + ET + XGBoost")]
    final_per_split = per_split[per_split.architecture.eq("RF + ET + XGBoost")]
    metrics_match = False
    if len(final) == 1 and len(final_per_split) == 20:
        row = final.iloc[0]
        metrics_match = (
            abs(float(row.MAE_mean) - float(final_per_split.MAE.mean())) < 1e-12
            and abs(float(row.RMSE_mean) - float(final_per_split.RMSE.mean())) < 1e-12
            and abs(float(row.R2_mean) - float(final_per_split.R2.mean())) < 1e-12
        )
    add(
        checks,
        "frozen_results",
        metrics_match,
        "MAE=0.116661; RMSE=0.174031; R2=0.865196",
    )

    syntax_errors = []
    for path in (ROOT / "code").glob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            syntax_errors.append(f"{path.name}: {error}")
    add(
        checks,
        "python_syntax",
        not syntax_errors,
        "all publication scripts parse" if not syntax_errors else "; ".join(syntax_errors),
    )

    forbidden_suffixes = {
        ".joblib", ".cdx", ".cdxml", ".docx", ".ppt", ".pptx", ".cube",
        ".sdf", ".mol", ".opju",
    }
    forbidden_files = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file()
        and (path.suffix.lower() in forbidden_suffixes or path.name == ".env")
    ]
    add(
        checks,
        "excluded_artifacts",
        not forbidden_files,
        "no models, structures, manuscripts, ChemDraw, or environment files"
        if not forbidden_files
        else "; ".join(forbidden_files),
    )

    windows_path_pattern = re.compile(r"[A-Za-z]:\\")
    credential_pattern = re.compile(
        r"(?i)(api[_-]?key|password|secret|credential|token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"
    )
    path_hits, credential_hits = [], []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py", ".json", ".md", ".txt", ".csv", ".gitignore"
        }:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if windows_path_pattern.search(text):
            path_hits.append(str(path.relative_to(ROOT)))
        if credential_pattern.search(text):
            credential_hits.append(str(path.relative_to(ROOT)))
    add(
        checks,
        "hard_coded_windows_paths",
        not path_hits,
        "none" if not path_hits else "; ".join(path_hits),
    )
    add(
        checks,
        "credentials",
        not credential_hits,
        "none" if not credential_hits else "; ".join(credential_hits),
    )

    passed = all(check["pass"] for check in checks)
    print(json.dumps({"status": "PASS" if passed else "FAIL", "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
