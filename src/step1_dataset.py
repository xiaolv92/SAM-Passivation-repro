"""
Step 1 - dataset validation -> MACCS + physical descriptors -> sparse / correlation pruning.

Inputs : data/dataset.xlsx (or --dataset name, e.g. dataset_demo)
Outputs: outputs/fingerprints_train.csv   meta columns + model features
         outputs/features.json            the retained feature list (used by step2/step3)
         outputs/step1_log.json           every row/bit that was dropped, and why
         outputs/figures/figS1_correlation.png

Feature blocks
    * MACCS keys 1-166                     (Su et al. fingerprint route)
    * 10 RDKit physical descriptors        (SAM-specific: size, aromaticity, flexibility)
    * 3 anchor-group flags                 (PO3H2 / COOH / SO3H -- the SAMs must bind ITO/NiOx)

Sparse threshold: the paper used <30 occurrences for n=78. For a small library that
threshold would delete everything, so it scales down with n (min(30, max(2, n//3)))
and the value actually used is written to step1_log.json.

Usage:
    python src/step1_dataset.py [--dataset dataset]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATA, FIG, OUT, ANCHOR_SMARTS, PHYSCHEM, read_dataset,  # noqa: E402
                    save_features)

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
CORR_LIMIT = 0.90
DROP_BY_HAND = ["M166"]          # "structures SMARTS cannot describe" -> no physical meaning


def maccs(mol):
    return [int(v) for v in MACCSkeys.GenMACCSKeys(mol)]


def anchor_flags(mol):
    flags = {}
    for tag, smarts in ANCHOR_SMARTS.items():
        flags[tag] = int(mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)))
    return flags


def physchem(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "NumAromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "TPSA": Descriptors.TPSA(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    args = ap.parse_args()

    df = read_dataset(args.dataset)
    raw_n = len(df)
    dropped = []

    # ---- (a) SMILES validity ------------------------------------------------------
    records = []
    for i, row in df.iterrows():
        smi = row["SMILES"]
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            dropped.append({"row": int(i), "molecule": str(row["Molecule"]),
                            "reason": "SMILES missing or unparsable"})
            continue
        rec = {"Molecule": row["Molecule"], "Abbrev": row["Abbrev"], "SMILES": smi,
               "Anchor": row["Anchor"], "Device_Structure": row["Device_Structure"],
               "Processing": row["Processing"], "Source": row["Source"], "DOI": row["DOI"],
               "Control_PCE": row["Control_PCE"], "Modified_PCE": row["Modified_PCE"],
               "Improved_Ratio": row["Improved_Ratio"],
               "Voc": row["Voc"], "Jsc": row["Jsc"], "FF": row["FF"]}
        rec.update(physchem(mol))
        rec.update(anchor_flags(mol))
        rec.update({f"M{k}": v for k, v in enumerate(maccs(mol))})
        records.append(rec)

    data = pd.DataFrame(records)
    if data.empty:
        raise SystemExit("no usable rows: fill data/dataset.xlsx (see data/README.md)")

    # ---- (b) duplicate structures -------------------------------------------------
    dup = data["SMILES"].duplicated(keep="first")
    for i in data.index[dup]:
        dropped.append({"row": int(i), "molecule": data.at[i, "Molecule"],
                        "reason": "duplicate SMILES (kept first occurrence)"})
    data = data.loc[~dup].reset_index(drop=True)

    # ---- (c) target column -------------------------------------------------------
    mod = pd.to_numeric(data["Modified_PCE"], errors="coerce")
    ctl = pd.to_numeric(data["Control_PCE"], errors="coerce")
    ratio = pd.to_numeric(data["Improved_Ratio"], errors="coerce")
    if ratio.notna().sum() < 3 and ctl.notna().sum() >= 3:
        ratio = (mod - ctl) / ctl
    data["Improved_Ratio"] = ratio
    data["Modified_PCE"] = mod
    data["Control_PCE"] = ctl
    data = data.loc[data["Modified_PCE"].notna() | data["Improved_Ratio"].notna()]
    data = data.reset_index(drop=True)
    n = len(data)

    # ---- (d) sparse bits ---------------------------------------------------------
    sparse_limit = min(30, max(2, n // 3))
    mcols = [c for c in data.columns if re.fullmatch(r"M\d+", c)]
    counts = data[mcols].sum(axis=0)
    sparse = counts[counts < sparse_limit].index.tolist()
    data = data.drop(columns=[c for c in mcols if c == "M0"] + DROP_BY_HAND
                     + sparse, errors="ignore")

    # ---- (e) correlation pruning (MACCS only; physical descriptors are kept) ------
    mcols = [c for c in data.columns if re.fullmatch(r"M\d+", c)]
    corr = data[mcols].corr().abs()
    pairs, drop = [], []
    for i, a in enumerate(mcols):
        for b in mcols[i + 1:]:
            v = corr.at[a, b]
            if pd.notna(v) and v > CORR_LIMIT:
                pairs.append([a, b, round(float(v), 3)])
    for a, b, _ in sorted(pairs, key=lambda t: -t[2]):
        victim = b
        if victim not in drop:
            drop.append(victim)
    data = data.drop(columns=drop, errors="ignore")

    features = [c for c in data.columns if re.fullmatch(r"M\d+", c)] + PHYSCHEM + list(ANCHOR_SMARTS)
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise SystemExit(f"feature construction failed, missing: {missing}")

    out_cols = ["Molecule", "Abbrev", "SMILES", "Anchor", "Device_Structure", "Processing",
                "Control_PCE", "Modified_PCE", "Improved_Ratio", "Voc", "Jsc", "FF",
                "Source", "DOI"] + features
    data[out_cols].to_csv(os.path.join(OUT, "fingerprints_train.csv"), index=False)
    save_features(features, sparse_limit)

    log = {"dataset": args.dataset, "rows_in": raw_n, "rows_used": n,
           "sparse_limit": sparse_limit, "maccs_removed_sparse": len(sparse),
           "maccs_removed_corr": drop, "corr_pairs": pairs,
           "n_maccs": sum(bool(re.fullmatch(r"M\d+", c)) for c in features),
           "n_physchem": len(PHYSCHEM), "n_anchor": len(ANCHOR_SMARTS),
           "n_features": len(features), "dropped": dropped}
    with open(os.path.join(OUT, "step1_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # ---- figure ------------------------------------------------------------------
    num = data[features].astype(float)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)
    im = ax.imshow(num.corr().values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7)
    ax.set_title(f"Feature correlation - {args.dataset} (n={n}, {len(features)} features)")
    fig.colorbar(im, shrink=0.75)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS1_correlation.png"))
    plt.close(fig)

    print(f"[step1] rows {raw_n} -> {n} usable | MACCS kept {log['n_maccs']} "
          f"(sparse<{sparse_limit} removed {len(sparse)}, corr removed {len(drop)}) "
          f"| total features {len(features)}")
    print("done ->", os.path.join(OUT, "fingerprints_train.csv"))


if __name__ == "__main__":
    main()
