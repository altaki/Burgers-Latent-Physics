#!/usr/bin/env python3
"""Audit versionné (numpy pur, lecture seule) de la strate test_extra ν=0.02.

Vérifie le npz généré par main_generate_data_nu_extra.py AVANT toute évaluation :
clés, formes, ν unique, dataset_version/protocol, gates Step 3
(evaluate_nu_sweep_gates, seuils config.py:375-380), familles de CI en round-robin,
energy0 = 1/2 * target_l2_norm^2, plage target [0.5, 2.0], mass0 <= 3.5e-15,
recompute energy depuis states, times (dt_saved = 0.02), grille x/dx,
seeds uniques DANS le npz ET disjoints des 900 seeds des splits nu_sweep existants
(npz lus en allow_pickle=True, jamais écrits), SHA-256.

Exit 0 = GO, exit 2 = NO-GO. Le rapport est refusé s'il existe déjà.
--smoke audite le npz de smoke (/tmp/nu_extra_smoke, expect-n 2, rapport dans /tmp).
Aucune écriture dans outputs/ en mode --smoke ; en mode réel, seul --report est écrit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import pde_dataset_builder as b  # numpy pur
from main_generate_data_nu_extra import nu_slug  # même slug que le générateur

DATA_DIR = REPO / "outputs" / "data"
EXISTING_NPZ = [DATA_DIR / f"{s}_data_nu_sweep.npz" for s in ("train", "val", "test")]
EXPECTED_KEYS = {
    "states", "times", "energy", "dissipation", "u0", "x", "dx", "nu_values",
    "ic_type_ids", "ic_type_names", "initial_max_abs_u", "initial_advective_cfl",
    "initial_diffusive_number", "energy_monotone_fraction",
    "energy_mean_positive_violation", "target_l2_norm", "energy0", "mass0",
    "dissipation_physical", "trajectory_seeds", "initial_cell_peclet",
    "dataset_version", "protocol",
}
# Seuils Step 3 (config.py:375-380, README:310-318)
GATES = dict(gate_monotone_frac=0.99, gate_monotone_atol_rel=1e-8,
             gate_pass_fraction=0.95, gate_max_abs_u=5.0,
             gate_dedt_rel_err=1e-2, gate_max_cell_peclet=2.0)
EXPECTED_IC_NAMES = ["fourier", "gaussian_bumps", "random_smooth"]
MASS0_MAX = 3.5e-15          # README:317
TARGET_RANGE = (0.5, 2.0)    # config.py:326
NX, N_TIMES, DT_SAVED = 1024, 101, 0.02
L = 2.0 * np.pi


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_seeds() -> np.ndarray:
    out = []
    for p in EXISTING_NPZ:
        d = np.load(p, allow_pickle=True)  # lecture seule
        out.append(np.asarray(d["trajectory_seeds"], dtype=np.int64))
    return np.concatenate(out)


def audit(npz: Path, expect_n: int, expect_nu: float) -> dict:
    d = np.load(npz, allow_pickle=True)  # lecture seule
    rep = {"npz": str(npz), "sha256": sha256_file(npz), "checks": {}, "diagnostics": {}}
    c = rep["checks"]

    c["keys_exact"] = set(d.files) == EXPECTED_KEYS
    states = np.asarray(d["states"], float)
    c["states_shape"] = states.shape == (expect_n, N_TIMES, NX)
    c["all_finite"] = bool(np.all(np.isfinite(states)) and np.all(np.isfinite(d["energy"]))
                           and np.all(np.isfinite(d["dissipation_physical"])))
    nu = np.asarray(d["nu_values"], float)
    c["nu_unique_valeur"] = bool(np.allclose(nu, expect_nu)) and len(np.unique(nu)) == 1
    c["dataset_version"] = str(d["dataset_version"]) == f"nu_sweep_v1_test_extra_{nu_slug(expect_nu)}"
    proto = str(d["protocol"])
    c["protocol_mentionne_nu_et_seeds"] = (f"nu={expect_nu:g}" in proto and "seeds disjoints" in proto)

    gates = b.evaluate_nu_sweep_gates(d, **GATES)
    rep["gates"] = gates
    c["gates_passed"] = gates["gates_passed"]
    c["cell_peclet_diagnostic_ok"] = gates["diagnostics"]["cell_peclet_ok"]

    ids = np.asarray(d["ic_type_ids"], int)
    names = [str(s) for s in d["ic_type_names"]]
    c["ic_names"] = names == EXPECTED_IC_NAMES
    counts = [int(np.sum(ids == k)) for k in range(len(EXPECTED_IC_NAMES))]
    c["ic_round_robin"] = max(counts) - min(counts) <= 1
    rep["diagnostics"]["ic_counts"] = dict(zip(names, counts))

    tln = np.asarray(d["target_l2_norm"], float)
    e0 = np.asarray(d["energy0"], float)
    c["target_l2_dans_plage"] = bool(np.all((tln >= TARGET_RANGE[0]) & (tln <= TARGET_RANGE[1])))
    c["energy0_egal_moitie_target_carre"] = bool(np.allclose(e0, 0.5 * tln ** 2, rtol=0, atol=1e-12))
    mass0 = np.asarray(d["mass0"], float)
    c["mass0_seuil"] = bool(np.all(np.abs(mass0) <= MASS0_MAX))
    rep["diagnostics"]["mass0_max_abs"] = float(np.max(np.abs(mass0)))

    # energy recomputée depuis states : E = 1/2 * dx * sum(u^2)
    dx = float(d["dx"])
    energy = np.asarray(d["energy"], float)
    energy_re = 0.5 * dx * np.sum(states ** 2, axis=-1)
    rep["diagnostics"]["energy_recompute_max_abs_diff"] = float(np.max(np.abs(energy - energy_re)))
    c["energy_recompute"] = rep["diagnostics"]["energy_recompute_max_abs_diff"] <= 1e-12
    c["u0_egal_states_t0"] = bool(np.array_equal(np.asarray(d["u0"], float), states[:, 0, :]))

    times = np.asarray(d["times"], float)
    c["times_triees_dt_saved"] = bool(np.all(np.diff(times) > 0) and np.allclose(np.diff(times), DT_SAVED, rtol=0, atol=1e-12))
    x = np.asarray(d["x"], float)
    c["grille_x_dx"] = bool(np.isclose(dx, L / NX, rtol=0, atol=1e-15)
                            and x.shape == (NX,) and np.isclose(x[1] - x[0], dx, rtol=0, atol=1e-15)
                            and np.isclose(x[0], 0.0))

    seeds = np.asarray(d["trajectory_seeds"], dtype=np.int64)
    c["seeds_uniques_dans_npz"] = bool(len(np.unique(seeds)) == expect_n)
    ex = existing_seeds()
    overlap = np.intersect1d(seeds, ex)
    c["seeds_disjoints_des_900_existants"] = bool(overlap.size == 0)
    rep["diagnostics"]["n_seeds_existants"] = int(ex.size)
    rep["diagnostics"]["overlap_seeds"] = int(overlap.size)

    rep["go"] = bool(all(v for v in c.values()))
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=Path, default=DATA_DIR / "test_extra_data_nu002.npz")
    ap.add_argument("--report", type=Path, default=REPO / "outputs" / "logs" / "audit_nu_extra_nu002.json")
    ap.add_argument("--expect-n", type=int, default=150)
    ap.add_argument("--expect-nu", type=float, default=0.02)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-npz", type=Path, default=Path("/tmp/nu_extra_smoke/test_extra_smoke_nu0p02.npz"))
    ap.add_argument("--smoke-report", type=Path, default=Path("/tmp/nu_extra_smoke/audit_smoke_report.json"))
    ap.add_argument("--smoke-expect-n", type=int, default=2, help="n attendu en mode --smoke (npz de smoke = 2 traj)")
    args = ap.parse_args()

    if args.smoke:
        if not args.smoke_npz.exists():
            print(f"[SMOKE][ABORT] {args.smoke_npz} absent (lancer d'abord le --smoke de main_generate_data_nu_extra.py).")
            return 2
        if args.smoke_report.exists():
            print(f"[SMOKE][ABORT] {args.smoke_report} existe déjà.")
            return 2
        rep = audit(args.smoke_npz, expect_n=args.smoke_expect_n, expect_nu=args.expect_nu)
        rep["smoke"] = True
        args.smoke_report.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        print(f"[SMOKE] rapport -> {args.smoke_report} ; GO={rep['go']}")
        print(json.dumps(rep["checks"], indent=2))
        return 0 if rep["go"] else 2

    if args.report.exists():
        print(f"[ABORT] {args.report} existe déjà (aucune écrasement).")
        return 2
    if not args.npz.exists():
        print(f"[ABORT] {args.npz} absent (générer d'abord).")
        return 2
    rep = audit(args.npz, expect_n=args.expect_n, expect_nu=args.expect_nu)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    print(f"[AUDIT] rapport -> {args.report} ; GO={rep['go']} ; sha256={rep['sha256']}")
    print(json.dumps(rep["checks"], indent=2))
    print("[AUDIT] gates:", json.dumps(rep["gates"]["checks"]), "peclet:", rep["gates"]["diagnostics"])
    return 0 if rep["go"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
