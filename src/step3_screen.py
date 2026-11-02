"""
Step 3 - virtual screening of the SAM candidate library.

Funnel: anchor group -> size / flexibility / polarity -> applicability domain -> ML tier

Inputs : outputs/fingerprints_train.csv, outputs/features.json, outputs/model_rf.joblib,
         outputs/target.json, data/candidates.csv
Outputs: outputs/screening_results.csv, outputs/feature_resolution.txt,
         outputs/figures/fig3_screening.png

SAM-specific gates (these replace the "vapour" gates of the Su paper)
    * the molecule must carry a phosphonic (or carboxylic) acid anchor, otherwise it
      cannot chemisorb on ITO/NiOx;
    * MW 150-800, rotatable bonds <= 12, TPSA <= 120 -- oversized or floppy molecules
      give non-uniform monolayers;
    * MACCS Tanimoto >= 0.40 to the training set, otherwise the model extrapolates.

Usage:
    python src/step3_screen.py [--top 20]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ANCHOR_SMARTS, DATA, FIG, OUT, PHYSCHEM, load_features,  # noqa: E402
                    rmse)

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdMolDescriptors
from sklearn.exceptions import ConvergenceWarning

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.linewidth": 1.2})

MW_MIN, MW_MAX = 150.0, 800.0
ROT_MAX = 12
TPSA_MAX = 120.0
AD_MIN_SIM = 0.40


def describe(mol):
    rec = {
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
    for tag, smarts in ANCHOR_SMARTS.items():
        rec[tag] = int(mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)))
    rec.update({f"M{k}": int(v) for k, v in enumerate(MACCSkeys.GenMACCSKeys(mol))})
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20, help="candidates written to step4")
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(OUT, "fingerprints_train.csv"))
    features = load_features()
    with open(os.path.join(OUT, "target.json"), encoding="utf-8") as f:
        meta = json.load(f)
    target = meta["target"]

    model_path = os.path.join(OUT, "model_rf.joblib")
    if not os.path.exists(model_path):
        raise SystemExit("run src/step2_features_train.py first (no trained RF found)")
    rf = joblib.load(model_path)

    cand = pd.read_csv(os.path.join(DATA, "candidates.csv"))
    rows, mols = [], []
    for _, r in cand.iterrows():
        mol = Chem.MolFromSmiles(r["SMILES"])
        rows.append(describe(mol) if mol is not None else None)
        mols.append(mol)
    ok = [i for i, v in enumerate(rows) if v is not None]
    desc = pd.DataFrame([rows[i] for i in ok])
    cand = cand.iloc[ok].reset_index(drop=True)
    for col in ["MolWt", "NumRotatableBonds", "TPSA", "PO3H2", "COOH", "SO3H"]:
        cand[col] = desc[col].values
    X = desc[features].astype(float)
    mols = [mols[i] for i in ok]

    # ---- ML prediction with per-tree spread ---------------------------------------
    tree = np.array([t.predict(X.values) for t in rf.estimators_])
    cand["pred"] = tree.mean(axis=0)
    cand["pred_std"] = tree.std(axis=0)
    base = float(pd.to_numeric(train[target], errors="coerce").median())
    cand["vs_median"] = cand["pred"] - base

    # ---- applicability domain -----------------------------------------------------
    train_fps = [MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(s)) for s in train["SMILES"]]
    cand["max_sim"] = [max(DataStructs.BulkTanimotoSimilarity(MACCSkeys.GenMACCSKeys(m), train_fps))
                       for m in mols]

    # ---- hard physics / chemistry gates -------------------------------------------
    def gate(r):
        if not (r["PO3H2"] or r["COOH"] or r["SO3H"]):
            return "reject: no anchoring group"
        if not (MW_MIN <= r["MolWt"] <= MW_MAX):
            return "reject: MW outside 150-800"
        if r["NumRotatableBonds"] > ROT_MAX:
            return "reject: too flexible (>12 rotatable bonds)"
        if r["TPSA"] > TPSA_MAX:
            return "reject: TPSA > 120 (poor film uniformity)"
        if r["max_sim"] < AD_MIN_SIM:
            return "reject: outside training domain"
        return "eligible"

    cand["decision"] = cand.apply(gate, axis=1)
    elig = cand[cand["decision"] == "eligible"]
    if len(elig):
        q33, q67 = elig["pred"].quantile([1 / 3, 2 / 3])
        tier = lambda v: "A-shortlist" if v >= q67 else ("B-backup" if v >= q33 else "C-deprioritise")
        cand.loc[elig.index, "decision"] = elig["pred"].map(tier)

    cand["family"] = cand.get("family", "designed")
    cand = cand.sort_values(["decision", "pred"], ascending=[True, False])
    cols = ["name", "SMILES", "family", "MolWt", "NumRotatableBonds", "TPSA", "max_sim",
            "pred", "pred_std", "vs_median", "decision", "note"]
    cand[cols].to_csv(os.path.join(OUT, "screening_results.csv"), index=False)
    print(cand[["name", "MolWt", "max_sim", "pred", "pred_std", "decision"]]
          .head(25).to_string(index=False))

    # ---- fingerprint collapse -----------------------------------------------------
    groups = {}
    for name, i in zip(cand["name"], cand.index):
        key = "|".join(f"{v:.4f}" for v in X.loc[i].values)
        groups.setdefault(key, []).append(name)
    unresolved = [g for g in groups.values() if len(g) > 1]
    with open(os.path.join(OUT, "feature_resolution.txt"), "w", encoding="utf-8") as f:
        for g in unresolved:
            f.write(", ".join(g) + "\n")
    print(f"\nsame feature-vector groups: {len(unresolved)}")

    # ---- candidates for the QC layer ---------------------------------------------
    top = cand[cand["decision"].isin(["A-shortlist", "B-backup"])].head(args.top)
    top[["name", "SMILES", "pred", "pred_std", "decision"]].to_csv(
        os.path.join(OUT, "qc_candidates.csv"), index=False)

    # ---- figure ------------------------------------------------------------------
    d = cand[cand["decision"].isin(["A-shortlist", "B-backup", "C-deprioritise"])] \
        .sort_values("pred")
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(d) + 1.8), dpi=200)
    color = {"A-shortlist": "seagreen", "B-backup": "cornflowerblue",
             "C-deprioritise": "lightgray"}
    ax.barh(d["name"], d["pred"], xerr=d["pred_std"],
            color=[color.get(x, "salmon") for x in d["decision"]], edgecolor="k",
            error_kw=dict(lw=0.8, capsize=2, ecolor="dimgray"))
    ax.axvline(base, ls="--", c="gray", lw=1)
    ax.set_xlabel(f"Predicted {target} (RF, mean $\\pm$ tree s.d.)")
    ax.set_title("SAM virtual screening (green = A-shortlist -> QC)")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_screening.png"))
    plt.close(fig)
    print(f"eligible={len(elig)}  A={sum(cand['decision'] == 'A-shortlist')}  "
          f"B={sum(cand['decision'] == 'B-backup')}  -> qc_candidates.csv")


if __name__ == "__main__":
    main()
