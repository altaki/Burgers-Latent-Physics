"""Test automatique de déterminisme (prereg TD 515cf5b §9, exigence 0.2).

Lance DEUX fois le même mini-run in-process (même graine 42, 2 époques, données
tronquées à 20 trajectoires copiées dans le dossier temp, encodeur td0, protocole
heldout_nu, z_target_mode=energy_centered) et compare :
  - toutes les valeurs numériques du metrics JSON (hors traceability : hash, chemins,
    timestamps) ;
  - les tableaux des predictions_test.npz (z, h, rec si présent, u).
Échoue (exit 1) si un écart absolu > 1e-6 ou un NaN est détecté.
Tant que ce test ne passe pas, aucune expérience appariée par graine n'est
interprétable au-delà du smoke (règle 0.2).

Usage (env torch, python3, racine du repo) : python3 main_determinism_check.py
Aucune écriture dans outputs/ (dossier temp uniquement).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import td_protocols as tp  # source de vérité unique des protocoles (correction bug filtre)

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
TOL = 1e-6


def _truncated_copy(tmp: Path) -> tuple:
    paths = {}
    for name, src in (("train", DATA / "train_data_nu_sweep.npz"),
                      ("val", DATA / "val_data_nu_sweep.npz"),
                      ("test", DATA / "test_data_nu_sweep.npz")):
        d = dict(np.load(src, allow_pickle=True))
        for k in ("states", "energy", "dissipation", "dissipation_physical", "u0"):
            if k in d:
                d[k] = d[k][:20]
        for k in ("nu_values", "ic_type_ids", "energy0", "mass0", "trajectory_seeds",
                  "initial_max_abs_u", "initial_advective_cfl", "initial_diffusive_number",
                  "energy_monotone_fraction", "energy_mean_positive_violation",
                  "target_l2_norm", "initial_cell_peclet"):
            if k in d:
                d[k] = d[k][:20]
        p = tmp / f"{name}_data_nu_sweep.npz"
        np.savez(p, **d)
        paths[name] = p
    return paths["train"], paths["val"], paths["test"]


def _strip_volatile(obj):
    """Retire les champs non numériques / dépendants du run (traceability, chemins)."""
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in ("traceability", "checkpoint_path", "best_model_path")}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _numeric_leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numeric_leaves(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _numeric_leaves(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def main() -> int:
    import ml_train_experiment_a_seq as mt

    tmp = Path(tempfile.mkdtemp(prefix="td_determinism_check_"))
    train_path, val_path, test_path = _truncated_copy(tmp)

    outs = []
    for i in (1, 2):
        out = tmp / f"run{i}" / "w25_heldout_nu_zenergy_centered_s42"
        mt.train_and_evaluate_a_seq(
            protocol="heldout_nu",
            train_path=train_path, val_path=val_path, test_path=test_path,
            output_dir=out,
            window_size=25,
            heldout_nu_values=tp.PROTOCOLS["heldout_nu"]["train_val_nu"],
            z_target_mode="energy_centered",
            z_weight=1.0,
            monotonicity_weight=0.0,
            nu_strat=0.01,
            num_epochs=2,
            seed=42,
            encoder_mode="td0",
            smoke=True,          # working tree non exigé propre pour ce test
            require_clean_tree=False,
            verbose=False,
        )
        outs.append(out)

    m1 = json.load(open(outs[0] / "metrics_window_a_w25_heldout_nu_zenergy_centered.json"))
    m2 = json.load(open(outs[1] / "metrics_window_a_w25_heldout_nu_zenergy_centered.json"))
    l1 = dict(_numeric_leaves(_strip_volatile(m1)))
    l2 = dict(_numeric_leaves(_strip_volatile(m2)))
    if set(l1) != set(l2):
        print("ÉCHEC : clés numériques différentes entre les deux runs")
        print("  seulement run1 :", sorted(set(l1) - set(l2))[:10])
        print("  seulement run2 :", sorted(set(l2) - set(l1))[:10])
        return 1
    worst = 0.0
    worst_key = ""
    for k in l1:
        a, b = l1[k], l2[k]
        if np.isnan(a) or np.isnan(b):
            print(f"ÉCHEC : NaN dans {k}")
            return 1
        d = abs(a - b)
        if d > worst:
            worst, worst_key = d, k
    print(f"metrics JSON : {len(l1)} valeurs numériques comparées, écart max = {worst:.3e} ({worst_key})")

    p1 = np.load(outs[0] / "predictions_test.npz", allow_pickle=True)
    p2 = np.load(outs[1] / "predictions_test.npz", allow_pickle=True)
    if set(p1.files) != set(p2.files):
        print("ÉCHEC : clés npz différentes")
        return 1
    worst_npz = 0.0
    worst_arr = ""
    for k in p1.files:
        a, b = p1[k], p2[k]
        if a.dtype.kind in "fiu":
            if a.shape != b.shape:
                print(f"ÉCHEC : shape différente pour {k}")
                return 1
            if np.isnan(a.astype(float)).any():
                print(f"ÉCHEC : NaN dans npz[{k}]")
                return 1
            d = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
            if d > worst_npz:
                worst_npz, worst_arr = d, k
    print(f"predictions_test.npz : {len(p1.files)} tableaux, écart max = {worst_npz:.3e} ({worst_arr})")

    ok = worst <= TOL and worst_npz <= TOL
    print(f"TOL={TOL:.0e} -> {'DÉTERMINISTE (exit 0)' if ok else 'NON DÉTERMINISTE (exit 1)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
