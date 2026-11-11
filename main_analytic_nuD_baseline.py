"""Baseline analytique de nu*D (Experiment A-Physics, tâches 1 + 4) — SANS réseau.

Depuis les états/énergie stockés dans les npz (lecture seule), calcule :
  (a) cible centrée      nuD_est_centered(t) = (E_{t-1} - E_{t+1}) / (2*dt_saved)
  (b) backward local     nuD_est_backward(t) = (E_{t-1} - E_t) / dt_saved
  (c) causale intra-fenêtre  nuD_est_causal_w(t) = (E_{t-w+1} - E_t) / ((w-1)*dt_saved), w = 25, 50

Évalue chaque estimateur vs le nuD stocké avec EXACTEMENT les mêmes sous-ensembles et
métriques que window-A (build_eval_subsets / evaluate_predictions / baseline_predictions
de ml_train_experiment_c) : global, par nu (surtout nu=0.01), par t, par quartile E(0),
par trajectoire ; Spearman, Pearson, R², pente de calibration, rel_rmse, baselines
global_mean + nu_oracle (fit sur train complet ET sur train heldout_nu ν∈{0.05,0.2}).

Validation de la cible (tâche 4) : erreur relative |est - nuD|/nuD — médiane, p95, max,
par strate de nu, par instant t≈0/1/2, et pendant le raidissement (pas où dD/dt > 0).

Sorties : outputs/logs/experiment_a_seq_analytic_baseline.json + .csv (long)

Usage : python main_analytic_nuD_baseline.py [--window-sizes 25 50]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "outputs" / "data"
LOGS_DIR = REPO_ROOT / "outputs" / "logs"


def load_split(name: str) -> Dict[str, np.ndarray]:
    d = np.load(DATA_DIR / f"{name}_data_nu_sweep.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def estimators(energy: np.ndarray, times: np.ndarray, window_sizes: Sequence[int]):
    dt = float(times[1] - times[0])
    n_times = energy.shape[1]
    out: Dict[str, np.ndarray] = {}
    est = np.full_like(energy, np.nan)
    est[:, 1:-1] = (energy[:, :-2] - energy[:, 2:]) / (2 * dt)
    est[:, 0] = (energy[:, 0] - energy[:, 1]) / dt
    est[:, -1] = (energy[:, -2] - energy[:, -1]) / dt
    out["centered"] = est
    est = np.full_like(energy, np.nan)
    est[:, 1:] = (energy[:, :-1] - energy[:, 1:]) / dt
    est[:, 0] = est[:, 1]
    out["backward"] = est
    for w in window_sizes:
        est = np.full_like(energy, np.nan)
        for t in range(w - 1, n_times):
            est[:, t] = (energy[:, t - w + 1] - energy[:, t]) / ((w - 1) * dt)
        out[f"causal_w{w}"] = est
    return out, dt


def pair_index_map(n_traj: int, n_times: int, w: int, stride: int = 1):
    return [(i, t) for i in range(n_traj) for t in range(w - 1, n_times - stride)]


def _err_summary(ev: np.ndarray) -> Dict[str, float]:
    if ev.size == 0:
        return {"median": None, "p95": None, "max": None, "n": 0}
    return {"median": float(np.median(ev)), "p95": float(np.percentile(ev, 95)),
            "max": float(np.max(ev)), "n": int(ev.size)}


def rel_err_stats(est, nuD, nu, times, diss):
    """Erreur relative |est-nuD|/nuD : global, par nu, par t, pendant le raidissement."""
    stats: Dict[str, Any] = {}
    m = ~np.isnan(est)
    e = np.abs(est[m] - nuD[m]) / np.maximum(nuD[m], 1e-30)
    stats["global"] = _err_summary(e)
    per_nu = {}
    for v in np.unique(nu):
        mv = m & np.isclose(nu[:, None], v)
        ev = np.abs(est[mv] - nuD[mv]) / np.maximum(nuD[mv], 1e-30)
        per_nu[f"nu={v:g}"] = _err_summary(ev)
    stats["per_nu"] = per_nu
    per_t = {}
    dt = float(times[1] - times[0])
    for t in (0.0, 1.0, 2.0):
        j = int(round(t / dt))
        mv = m[:, j]
        ev = np.abs(est[mv, j] - nuD[mv, j]) / np.maximum(nuD[mv, j], 1e-30)
        per_t[f"t={t:g}"] = _err_summary(ev)
    stats["per_t"] = per_t
    # pendant le raidissement : pas où D augmente (dD/dt > 0), cible au pas t+1
    dD = np.diff(diss, axis=1)
    steep = (dD > 0) & m[:, 1:]
    ev = np.abs(est[:, 1:][steep] - nuD[:, 1:][steep]) / np.maximum(nuD[:, 1:][steep], 1e-30)
    s = _err_summary(ev)
    s["frac_steps_dD_gt_0"] = float((dD > 0).mean())
    stats["steepening_dD_gt_0"] = s
    return stats


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[25, 50])
    args = parser.parse_args(argv)

    import ml_train_experiment_c as mt  # sous-ensembles + métriques identiques à window-A

    train = load_split("train")
    test = load_split("test")
    eval_times = (1.0, 2.0)

    e0_edges = np.quantile(np.asarray(train["energy0"], dtype=float),
                           [0.0, 0.25, 0.5, 0.75, 1.0])

    est_test, dt = estimators(test["energy"].astype(float), test["times"].astype(float), args.window_sizes)
    est_train, _ = estimators(train["energy"].astype(float), train["times"].astype(float), args.window_sizes)

    nuD_test = test["dissipation_physical"].astype(float)
    nu_test = test["nu_values"].astype(float)
    nuD_train = train["dissipation_physical"].astype(float)
    nu_train = train["nu_values"].astype(float)

    out: Dict[str, Any] = {
        "experiment": "analytic_nuD_baseline",
        "dataset_version": "nu_sweep_v1",
        "dt_saved": dt,
        "window_sizes": list(args.window_sizes),
        "estimators": {},
        "baselines": {},
        "target_validation": {},
    }
    all_rows = []

    for w in args.window_sizes:
        pairs = pair_index_map(test["states"].shape[0], test["states"].shape[1], w)
        idx_i = np.array([p[0] for p in pairs])
        idx_t = np.array([p[1] for p in pairs])
        meta = {
            "nu": nu_test[idx_i],
            "time": test["times"].astype(float)[idx_t],
            "energy0": test["energy0"].astype(float)[idx_i],
            "trajectory_index": idx_i,
        }
        targets = nuD_test[idx_i, idx_t].reshape(-1, 1)

        # baselines : train complet (standard) et train heldout_nu (nu in {0.05, 0.2})
        train_pairs = pair_index_map(train["states"].shape[0], train["states"].shape[1], w)
        tr_i = np.array([p[0] for p in train_pairs])
        tr_t = np.array([p[1] for p in train_pairs])
        train_targets = nuD_train[tr_i, tr_t].reshape(-1, 1)
        train_nu = nu_train[tr_i]
        held_mask = np.isclose(train_nu[:, None], [0.05, 0.2]).any(axis=1)
        for proto, tt, tn in (("standard", train_targets, train_nu),
                               ("heldout_nu", train_targets[held_mask], train_nu[held_mask])):
            for kind in ("global_mean", "nu_oracle"):
                bp = mt.baseline_predictions(tt, tn, meta["nu"], kind=kind)
                bn, br = mt.evaluate_predictions(
                    bp, targets, meta, ("nu_dissipation",),
                    e0_quartile_edges=e0_edges, eval_times=eval_times,
                    protocol=proto, model_label=f"baseline_{kind}",
                )
                out["baselines"][f"{proto}/{kind}"] = bn
                all_rows.extend(br)

        for est_name, est in est_test.items():
            if est_name.startswith("causal_w") and int(est_name.split("w")[1]) != w:
                continue
            preds = est[idx_i, idx_t].reshape(-1, 1)
            nested, rows = mt.evaluate_predictions(
                preds, targets, meta, ("nu_dissipation",),
                e0_quartile_edges=e0_edges, eval_times=eval_times,
                protocol="standard", model_label=f"analytic_{est_name}_w{w}",
            )
            out["estimators"][f"{est_name}_w{w}"] = nested
            all_rows.extend(rows)
            # validation de la cible (tâche 4)
            out["target_validation"][f"{est_name}_w{w}"] = rel_err_stats(
                est, nuD_test, nu_test, test["times"].astype(float), test["dissipation"].astype(float)
            )
            g = nested["global"]["nu_dissipation"]
            n1 = nested["nu=0.01"]["nu_dissipation"]
            print(f"w={w} {est_name:12s} global: spearman={g['spearman']:.4f} r2={g['r2']:.4f} | "
                  f"nu=0.01: spearman={n1['spearman']:.4f} r2={n1['r2']:.4f} slope={n1['calibration_slope']:.3f} "
                  f"rel_rmse={n1['rel_rmse_mean']:.3f}")

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = LOGS_DIR / "experiment_a_seq_analytic_baseline.json"
    json_path.write_text(json.dumps(out, indent=1))
    with open(LOGS_DIR / "experiment_a_seq_analytic_baseline.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["protocol", "model", "subset", "target", "metric", "value"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nEcrit: {json_path}")
    print(f"Ecrit: {LOGS_DIR / 'experiment_a_seq_analytic_baseline.csv'}")


if __name__ == "__main__":
    main()
