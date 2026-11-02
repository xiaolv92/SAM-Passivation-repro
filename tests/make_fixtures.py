"""
Synthetic Gaussian / VASP outputs so step5 can be smoke-tested without a licence.

Writes a fake but structurally valid log/OUTCAR layout into qc_test/:
    qc_test/gaussian/Pb2I4.log, <name>_monomer.log, <name>_Pb2I4.log
    qc_test/vasp/niO_clean|pbi2_clean|niO_<name>|pbi2_<name>/OUTCAR

The numbers are invented on purpose - they only prove that the parser and the energy
bookkeeping work. Never quote them.

Usage:
    python tests/make_fixtures.py      # -> qc_test/
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "qc_test")
GAU = os.path.join(OUT, "gaussian")
VAS = os.path.join(OUT, "vasp")
for d in (GAU, VAS):
    os.makedirs(d, exist_ok=True)

NAMES = ["H-C2-PACz", "OMe-C4-PACz", "I-C6-PACz"]


def gau_log(path, energy, homo=-0.21, lumo=-0.06, dipole=8.0):
    text = f""" SCF Done:  E(RB3LYP) =  {energy:.10f}     A.U. after   10 cycles
 Alpha  occ. eigenvalues --  {homo - 0.4:.5f} {homo:.5f}
 Alpha virt. eigenvalues --  {lumo:.5f} {lumo + 0.3:.5f}
 Dipole moment (field-independent basis, Debye):
 X={dipole * 0.6:.4f} Y={dipole * 0.5:.4f} Z={dipole * 0.62:.4f} Tot={dipole:.4f}
 Normal termination of Gaussian 16 at Sun Nov  2 12:00:00 2026.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def outcar(path, energy):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"  free  energy   TOTEN  =  {energy:.6f} eV\n")


def main():
    gau_log(os.path.join(GAU, "Pb2I4.log"), -1200.0)
    for case in ("niO_clean", "pbi2_clean"):
        os.makedirs(os.path.join(VAS, case), exist_ok=True)
    outcar(os.path.join(VAS, "niO_clean", "OUTCAR"), -900.0)
    outcar(os.path.join(VAS, "pbi2_clean", "OUTCAR"), -1500.0)
    for i, name in enumerate(NAMES):
        e = -480.0 - i
        gau_log(os.path.join(GAU, f"{name}_monomer.log"), e, dipole=7.0 + i)
        # anchored complex: ~0.35 eV more stable than the separated fragments
        gau_log(os.path.join(GAU, f"{name}_Pb2I4.log"), e - 1200.0 - 0.0128)
        os.makedirs(os.path.join(VAS, f"niO_{name}"), exist_ok=True)
        os.makedirs(os.path.join(VAS, f"pbi2_{name}"), exist_ok=True)
        outcar(os.path.join(VAS, f"niO_{name}", "OUTCAR"), -900.0 + e * 27.2114 - 0.5 * (1 + 0.1 * i))
        outcar(os.path.join(VAS, f"pbi2_{name}", "OUTCAR"), -1500.0 + e * 27.2114 - 0.4)
    print(f"synthetic QC fixtures -> {OUT}")


if __name__ == "__main__":
    main()
