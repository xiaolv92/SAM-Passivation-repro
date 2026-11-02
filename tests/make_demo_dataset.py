"""
Create a synthetic dataset so the whole pipeline can be smoke-tested without
publishing anyone's measurements.

The numbers are generated from a simple structure-property rule (more aromatic rings
and shorter linkers -> slightly higher PCE) plus Gaussian noise. They are SYNTHETIC:
use them only to check that step1-step3 run, never in a paper.

Usage:
    python tests/make_demo_dataset.py     # -> data/dataset_demo.xlsx (60 rows)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from common import DATA, COLS  # noqa: E402

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def smiles_for(sub, linker):
    pattern = "n1c2ccccc2c2ccccc21" if sub is None else f"n1c2ccc({sub})cc2c2cc({sub})ccc21"
    return "OP(=O)(O)" + linker + pattern


def main():
    rng = np.random.default_rng(7)
    subs = [("Et", "CC"), ("Pr", "CCC"), ("OEt", "OCC"), ("CF3", "C(F)(F)F"),
            ("NO2", "[N+](=O)[O-]"), ("NH2", "N"), ("OH", "O"), ("SH", "S")]
    linkers = [("C3", "CCC"), ("C5", "CCCCC"), ("C7", "CCCCCCC")]
    rows = []
    for sn, sub in subs:
        for ln, linker in linkers:
            smi = smiles_for(sub, linker)
            mol = Chem.MolFromSmiles(smi)
            rings = rdMolDescriptors.CalcNumAromaticRings(mol)
            flex = rdMolDescriptors.CalcNumRotatableBonds(mol)
            base = 25.0 + 0.25 * rings - 0.12 * flex + rng.normal(0, 0.35)
            rows.append({"Molecule": f"{sn}-{ln}-PACz", "Abbrev": f"{sn}{ln}",
                         "SMILES": smi, "Anchor": "PO3H2",
                         "Device_Structure": "ITO/SAM/perovskite/C60/BCP/Ag",
                         "Processing": "spin-coating",
                         "Control_PCE": np.nan, "Modified_PCE": round(base, 2),
                         "Improved_Ratio": np.nan, "Voc": np.nan, "Jsc": np.nan, "FF": np.nan,
                         "Source": "SYNTHETIC demo (tests/make_demo_dataset.py)",
                         "DOI": ""})
    # a second scaffold family, so the applicability domain has something to reject
    for name, smi in {"TPA-PA": "OP(=O)(O)c1ccc(N(c2ccccc2)c2ccccc2)cc1",
                      "BPh-PA": "OP(=O)(O)c1ccc(-c2ccccc2)cc1",
                      "Naph-PA": "OP(=O)(O)c1ccc2ccccc2c1"}.items():
        rows.append({"Molecule": f"{name}", "Abbrev": name,
                     "SMILES": smi, "Anchor": "PO3H2",
                     "Device_Structure": "ITO/SAM/perovskite/C60/BCP/Ag",
                     "Processing": "spin-coating",
                     "Control_PCE": np.nan,
                     "Modified_PCE": round(24.2 + rng.normal(0, 0.5), 2),
                     "Improved_Ratio": np.nan, "Voc": np.nan, "Jsc": np.nan, "FF": np.nan,
                     "Source": "SYNTHETIC demo (tests/make_demo_dataset.py)", "DOI": ""})

    df = pd.DataFrame(rows, columns=COLS)
    out = os.path.join(DATA, "dataset_demo.xlsx")
    df.to_excel(out, index=False)
    print(f"{len(df)} synthetic rows -> {out}")
    print("SYNTHETIC ONLY - never cite these numbers.")


if __name__ == "__main__":
    main()
