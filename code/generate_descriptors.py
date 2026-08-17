"""Generate the frozen 2,453-feature representation from canonical SMILES."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def topology_proxy(mol: Chem.Mol) -> tuple[list[str], list[float]]:
    atoms = list(mol.GetAtoms())
    n_atoms = max(1, len(atoms))
    boron = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 5]
    nitrogen = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 7]
    distance_matrix = Chem.GetDistanceMatrix(mol)
    distances = [distance_matrix[i, j] for i in boron for j in nitrogen]
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    graph = {index: set() for index in range(len(rings))}
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= 2:
                graph[i].add(j)
                graph[j].add(i)
    components: list[list[int]] = []
    seen: set[int] = set()
    for index in graph:
        if index in seen:
            continue
        stack, component = [index], []
        seen.add(index)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    fused = sum(bool(neighbors) for neighbors in graph.values())
    largest = max((len(component) for component in components), default=0)
    names = [
        "B_count", "N_count", "heteroatom_count", "aromatic_atom_fraction",
        "aromatic_ring_count", "total_ring_count", "fused_ring_count",
        "largest_fused_ring_system_size", "bridgehead_atom_count",
        "rotatable_bond_count", "fraction_sp3_carbon", "molecular_weight",
        "TPSA", "formal_charge", "BN_pair_count", "min_BN_topological_distance",
        "mean_BN_topological_distance", "BN_distance_le2_count",
        "BN_distance_le3_count", "BN_distance_le4_count", "ring_fusion_density",
    ]
    values = [
        len(boron), len(nitrogen),
        sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms),
        sum(atom.GetIsAromatic() for atom in atoms) / n_atoms,
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumRings(mol),
        fused,
        largest,
        rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Descriptors.MolWt(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Chem.GetFormalCharge(mol),
        len(distances),
        min(distances) if distances else np.nan,
        np.mean(distances) if distances else np.nan,
        sum(distance <= 2 for distance in distances),
        sum(distance <= 3 for distance in distances),
        sum(distance <= 4 for distance in distances),
        fused / max(1, len(rings)),
    ]
    return names, values


def build_components(smiles_values) -> tuple[dict[str, np.ndarray], list[str]]:
    mols = [Chem.MolFromSmiles(str(smiles)) for smiles in smiles_values]
    if any(mol is None for mol in mols):
        invalid = [index for index, mol in enumerate(mols) if mol is None]
        raise RuntimeError(f"Invalid SMILES at rows: {invalid[:20]}")
    descriptors = Descriptors._descList
    if len(descriptors) != 217:
        raise RuntimeError(
            f"Expected 217 RDKit 2D descriptors, found {len(descriptors)}; "
            "use the frozen RDKit version from requirements.txt"
        )
    morgan = np.zeros((len(mols), 2048), np.float32)
    maccs = np.zeros((len(mols), 167), np.float32)
    rdkit_values, proxy_values = [], []
    generator = AllChem.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    proxy_names: list[str] | None = None
    for index, mol in enumerate(mols):
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), morgan[index])
        DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs[index])
        row = []
        for _, function in descriptors:
            try:
                row.append(float(function(mol)))
            except Exception:
                row.append(np.nan)
        rdkit_values.append(row)
        proxy_names, values = topology_proxy(mol)
        proxy_values.append(values)
    rdkit_array = np.asarray(rdkit_values, np.float32)
    proxy_array = np.asarray(proxy_values, np.float32)
    rdkit_array[~np.isfinite(rdkit_array)] = np.nan
    proxy_array[~np.isfinite(proxy_array)] = np.nan
    rdkit_array[np.abs(rdkit_array) > 1e10] = np.nan
    if proxy_names is None:
        raise RuntimeError("No structures were supplied")
    names = (
        [f"morgan_{index}" for index in range(2048)]
        + [f"maccs_{index}" for index in range(167)]
        + ["rdkit_" + name for name, _ in descriptors]
        + ["proxy_" + name for name in proxy_names]
    )
    components = {
        "Morgan_r2": morgan,
        "MACCS": maccs,
        "RDKit_2D": rdkit_array,
        "topology": proxy_array,
    }
    expected = {
        "Morgan_r2": (len(mols), 2048),
        "MACCS": (len(mols), 167),
        "RDKit_2D": (len(mols), 217),
        "topology": (len(mols), 21),
    }
    if any(components[name].shape != shape for name, shape in expected.items()):
        raise RuntimeError("Descriptor dimension check failed")
    return components, names


def representations(components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "F05_Morgan_r2_MACCS": np.hstack(
            [components["Morgan_r2"], components["MACCS"]]
        ),
        "F10_all": np.hstack(
            [
                components["Morgan_r2"],
                components["MACCS"],
                components["RDKit_2D"],
                components["topology"],
            ]
        ),
    }


def main(data_path: Path, output_path: Path) -> None:
    data = pd.read_csv(data_path)
    if len(data) != 1396 or data.canonical_smiles.nunique() != 1396:
        raise RuntimeError("Dataset must contain 1,396 unique canonical SMILES")
    components, names = build_components(data.canonical_smiles)
    reps = representations(components)
    if reps["F05_Morgan_r2_MACCS"].shape != (1396, 2215):
        raise RuntimeError("RF representation dimension mismatch")
    if reps["F10_all"].shape != (1396, 2453):
        raise RuntimeError("ET/XGBoost representation dimension mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **components,
        feature_names=np.asarray(names, dtype=object),
        canonical_structure_group_id=data.canonical_structure_group_id.to_numpy(object),
    )
    print(f"Wrote {output_path}")
    print("F05_Morgan_r2_MACCS: 2215 features")
    print("F10_all: 2453 features")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=PACKAGE_ROOT / "data" / "E00_UNIFIED_1396.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PACKAGE_ROOT
        / "reproduction"
        / "features"
        / "FEATURE_COMPONENTS_1396.npz",
    )
    arguments = parser.parse_args()
    main(arguments.data.resolve(), arguments.output.resolve())
