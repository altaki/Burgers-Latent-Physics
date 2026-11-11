"""Intervalles de confiance pour les métriques z vs nu*D à partir des
`predictions_test.npz` (écrits par train_and_evaluate_a_seq ou main_export_predictions).

Deux IC, rapportés SÉPARÉMENT :
  1. IC bootstrap par trajectoire : on rééchantillonne les trajectoires de la
     strate nu=0.01 (test) avec remise (B resamples), on recalcule les métriques
     sur les fenêtres regroupées -> percentiles 2.5 / 97.5.
  2. IC inter-graines : si plusieurs npz partagent (window_size, protocol,
     z_target_mode) et ne diffèrent que par la graine, on rapporte moyenne, écart-type
     et IC à 95 % (t de Student, ddl = n-1) des métriques ponctuelles par graine.

Usage (numpy/pandas uniquement, pas besoin de torch) :
    python main_bootstrap_a_seq.py --predictions outputs/models/latent_dynamics_autoencoder_seq/window_w25_heldout_nu_s42/predictions_test.npz ...
    python main_bootstrap_a_seq.py --glob "outputs/models/latent_dynamics_autoencoder_seq/*/predictions_test.npz" --out outputs/logs/bootstrap_a_seq.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent

# t de Student à 97.5 % pour petits ddl (inter-graines)
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def subset_metrics(pred: np.ndarray, targ: np.ndarray) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=float).reshape(-1)
    targ = np.asarray(targ, dtype=float).reshape(-1)
    if len(targ) < 2:
        return {"n": int(len(targ))}
    rx = pd.Series(pred).rank().to_numpy()
    ry = pd.Series(targ).rank().to_numpy()
    ss_res = float(np.sum((pred - targ) ** 2))
    ss_tot = float(np.sum((targ - targ.mean()) ** 2))
    return {
        "n": int(len(targ)),
        "spearman": float(np.corrcoef(rx, ry)[0, 1]),
        "pearson": float(np.corrcoef(pred, targ)[0, 1]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "rel_rmse": float(np.sqrt(np.mean((pred - targ) ** 2)) / np.mean(np.abs(targ))),
        "calibration_slope": float(np.polyfit(targ, pred, 1)[0]),
    }


METRIC_KEYS = ("spearman", "pearson", "r2", "rel_rmse", "calibration_slope")


def bootstrap_run(
    z: np.ndarray,
    nu: np.ndarray,
    nuD: np.ndarray,
    traj: np.ndarray,
    nu_value: float,
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    z = np.asarray(z, dtype=float).reshape(-1)
    nu = np.asarray(nu, dtype=float).reshape(-1)
    nuD = np.asarray(nuD, dtype=float).reshape(-1)
    traj = np.asarray(traj).reshape(-1)
    m = np.isclose(nu, nu_value)
    z1, t1, tr1 = z[m], nuD[m], traj[m]
    point = subset_metrics(z1, t1)
    uniq = np.unique(tr1)
    rng = np.random.default_rng(seed)
    boot: Dict[str, List[float]] = {k: [] for k in METRIC_KEYS}
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        mask = np.isin(tr1, pick)
        # chaque trajectoire tirée k fois contribue k fois (poids implicite)
        counts = {u: int(np.sum(pick == u)) for u in pick}
        idx = np.concatenate([np.flatnonzero(tr1 == u) for u in pick for _ in range(counts[u])])
        mm = subset_metrics(z1[idx], t1[idx])
        for k in METRIC_KEYS:
            boot[k].append(mm[k])
    ci = {}
    for k in METRIC_KEYS:
        arr = np.asarray(boot[k], dtype=float)
        ci[k] = {
            "mean": float(np.mean(arr)),
            "p2.5": float(np.percentile(arr, 2.5)),
            "p97.5": float(np.percentile(arr, 97.5)),
        }
    return {
        "nu": float(nu_value),
        "n_windows": int(m.sum()),
        "n_trajectories": int(len(uniq)),
        "n_boot": int(n_boot),
        "point": point,
        "bootstrap_ci95": ci,
    }


def inter_seed_ci(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    t = _T975.get(n - 1, 1.96) if n > 1 else 1.96
    half = t * std / np.sqrt(n) if n > 1 else float("nan")
    return {"n_seeds": n, "mean": mean, "std": std,
            "ci95": [mean - half, mean + half] if n > 1 else [mean, mean]}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="*", default=None)
    parser.add_argument("--glob", action="append", default=None, help="Pattern glob de predictions_test.npz (répétable)")
    parser.add_argument("--nu", type=float, default=0.01, help="Strate nu évaluée (held-out)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=str(REPO / "outputs" / "logs" / "bootstrap_a_seq.json"))
    args = parser.parse_args(argv)

    paths: List[str] = list(args.predictions or [])
    if args.glob:
        for g in args.glob:
            paths += sorted(glob.glob(g))
    if not paths:
        raise SystemExit("Aucun predictions_test.npz fourni (--predictions ou --glob).")

    runs: Dict[str, Any] = {}
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in paths:
        p = Path(p)
        d = np.load(p, allow_pickle=True)
        res = bootstrap_run(
            d["z"], d["nu"], d["nu_dissipation_t"], d["trajectory_index"],
            nu_value=args.nu, n_boot=args.n_boot, seed=args.seed,
        )
        entry = {
            "file": str(p),
            "run_name": str(d["run_name"]) if "run_name" in d else p.parent.name,
            "protocol": str(d["protocol"]) if "protocol" in d else "?",
            "window_size": int(d["window_size"]) if "window_size" in d else -1,
            "z_target_mode": str(d["z_target_mode"]) if "z_target_mode" in d else "?",
            "seed": int(d["seed"]) if "seed" in d else -1,
            "source": str(d["source"]) if "source" in d else "?",
            **res,
        }
        runs[p.parent.name] = entry
        key = (entry["source"], entry["window_size"], entry["protocol"], entry["z_target_mode"])
        groups.setdefault(key, []).append(entry)

    group_stats: Dict[str, Any] = {}
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        g = {"members": [e["run_name"] for e in entries], "per_seed": {}}
        for k in METRIC_KEYS:
            vals = [e["point"].get(k, float("nan")) for e in entries]
            g["per_seed"][k] = inter_seed_ci(vals)
        group_stats["|".join(map(str, key))] = g

    out = {"nu": args.nu, "n_boot": args.n_boot, "runs": runs, "inter_seed": group_stats}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Bootstrap écrit : {out_path}")
    for name, e in runs.items():
        pt = e["point"]
        print(f"  {name}: spearman={pt.get('spearman', float('nan')):.4f} "
              f"r2={pt.get('r2', float('nan')):.4f} rel_rmse={pt.get('rel_rmse', float('nan')):.4f} "
              f"(n={e['n_windows']} fenetres, {e['n_trajectories']} traj)")


if __name__ == "__main__":
    main()
