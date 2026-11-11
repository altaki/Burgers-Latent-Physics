"""Analyse des latents des checkpoints window-A (Step 5) : h vs nu*D, probe lineaire.

Pour chaque run `outputs/models/latent_dynamics_autoencoder_seq/window_w{w}_{protocole}` :
  - recharge best_model.pt + standardizer sauvegardes (aucun re-entraînement),
  - infère z et h sur TRAIN (avec nu_filter pour heldout_nu) et TEST,
  - métriques per-dim (h0..h7, z) vs nu_dissipation_t : Spearman, Pearson, R² affine
    (calibrée), pente, rel_rmse — sur global / nu=0.01 / nu=0.05 / nu=0.2,
  - identification des dims : Spearman de chaque h_j vs dissipation_t (D) et energy_t (E),
  - probe lineaire (moindres carrés, ajusté sur TRAIN) nuD ~ [h, 1] et nuD ~ [z, 1],
    évalué sur TEST (généralisation held-out incluse).

Sorties : outputs/logs/h_dims_window_a_analysis.{json,csv} et
          outputs/logs/h_linear_probe_window_a.json

Usage :
    python main_analyze_window_a_latents.py [--window-sizes 25 50]
                                            [--protocols standard heldout_nu]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

import ml_datasets as mld
import ml_datasets_experiment_c as ec_ds
import ml_models_experiment_c as ec_models
import ml_train_experiment_a_seq as aseq
import physics_candidates as phys

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "outputs" / "models" / "latent_dynamics_autoencoder_seq"
TRAIN_NPZ = REPO_ROOT / "outputs" / "data" / "train_data_nu_sweep.npz"
TEST_NPZ = REPO_ROOT / "outputs" / "data" / "test_data_nu_sweep.npz"
LOGS_DIR = REPO_ROOT / "outputs" / "logs"


def _load_run(run_dir: Path, window_size: int, protocol: str):
    metrics_name = f"metrics_window_a_{run_dir.name.removeprefix('window_')}.json"
    metrics = json.load(open(run_dir / metrics_name))
    cfg = dict(metrics["model_config"])
    cfg.pop("model", None)
    cfg.pop("loss", None)
    model = ec_models.build_window_dynamics_autoencoder(**cfg)
    aseq.load_best_model_a_seq(model, run_dir / "best_model.pt", device=torch.device("cpu"))
    std = json.load(open(run_dir / "standardizers.json"))["input_standardizer"]
    standardizer = mld.GlobalStandardizer(mean=std["mean"], std=std["std"], eps=std["eps"])
    nu_filter = (0.05, 0.2) if protocol == "heldout_nu" else None
    train_raw = mld.load_dataset_npz(TRAIN_NPZ)
    test_raw = mld.load_dataset_npz(TEST_NPZ)
    train_ds = ec_ds.NuSweepWindowPairDataset(
        train_raw, window_size=window_size, stride=1, normalize=True,
        standardizer=standardizer, nu_filter=nu_filter,
    )
    test_ds = ec_ds.NuSweepWindowPairDataset(
        test_raw, window_size=window_size, stride=1, normalize=True,
        standardizer=standardizer, nu_filter=None,
    )
    return model, train_ds, test_ds


def _affine_metrics(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, float)[mask]
    y = np.asarray(y, float)[mask]
    cal = phys.fit_affine_calibration(x, y)  # y ~ a*x + b
    rmse = cal["metrics"]["rmse"]
    return {
        "spearman": float(phys.spearman_correlation(x, y)),
        "pearson": float(phys.pearson_correlation(x, y)),
        "r2_affine": float(cal["metrics"]["r2"]),
        "slope": float(cal["a"]),
        "rel_rmse_affine": float(rmse / max(np.mean(np.abs(y)), 1e-30)),
        "n": int(mask.sum()),
    }


def _plain_metrics(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, float)[mask]
    y = np.asarray(y, float)[mask]
    r2 = 1.0 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "spearman": float(phys.spearman_correlation(pred, y)),
        "pearson": float(phys.pearson_correlation(pred, y)),
        "r2": float(r2),
        "rel_rmse": float(np.sqrt(np.mean((pred - y) ** 2)) / max(np.mean(np.abs(y)), 1e-30)),
        "n": int(mask.sum()),
    }


def analyze_run(run_dir: Path, window_size: int, protocol: str, verbose: bool = True):
    model, train_ds, test_ds = _load_run(run_dir, window_size, protocol)
    device = torch.device("cpu")
    p_tr = aseq.predict_a_seq(model, train_ds, device=device, batch_size=1024)
    p_te = aseq.predict_a_seq(model, test_ds, device=device, batch_size=1024)

    h_te, z_te = p_te["h"], p_te["z"]
    h_tr, z_tr = p_tr["h"], p_tr["z"]
    nuD_te = np.asarray(p_te["meta"]["nu_dissipation_t"], float)
    nuD_tr = np.asarray(p_tr["meta"]["nu_dissipation_t"], float)
    nu_te = np.asarray(p_te["meta"]["nu"], float)

    subsets = {"global": np.ones(len(nuD_te), bool)}
    for v in (0.01, 0.05, 0.2):
        subsets[f"nu={v:g}"] = np.isclose(nu_te, v)

    # --- per-dim vs nuD ---
    dims_res: Dict[str, Any] = {}
    rows = []
    vecs = [(f"h{j}", h_te[:, j]) for j in range(h_te.shape[1])] + [("z", z_te)]
    for name, vec in vecs:
        dims_res[name] = {}
        for sname, mask in subsets.items():
            m = _affine_metrics(vec, nuD_te, mask)
            dims_res[name][sname] = m
            for k in ("spearman", "pearson", "r2_affine", "slope", "rel_rmse_affine"):
                rows.append({"run": run_dir.name, "dim": name, "subset": sname,
                             "metric": k, "value": m[k]})

    # --- identification des dims vs D et E ---
    d_te = np.asarray(p_te["meta"]["dissipation_t"], float)
    e_te = np.asarray(p_te["meta"]["energy_t"], float)
    dims_vs_de = {}
    for j in range(h_te.shape[1]):
        dims_vs_de[f"h{j}"] = {
            "spearman_vs_D": float(phys.spearman_correlation(h_te[:, j], d_te)),
            "spearman_vs_E": float(phys.spearman_correlation(h_te[:, j], e_te)),
            "spearman_vs_nuD": float(phys.spearman_correlation(h_te[:, j], nuD_te)),
        }

    # --- probe lineaire ajuste sur train ---
    a_h, *_ = np.linalg.lstsq(np.column_stack([h_tr, np.ones(len(h_tr))]), nuD_tr, rcond=None)
    pred_h = np.column_stack([h_te, np.ones(len(h_te))]) @ a_h
    a_z, *_ = np.linalg.lstsq(np.column_stack([z_tr, np.ones(len(z_tr))]), nuD_tr, rcond=None)
    pred_z = np.column_stack([z_te, np.ones(len(z_te))]) @ a_z
    probe = {"h_linear": {}, "z_linear": {}}
    for label, pred in (("h_linear", pred_h), ("z_linear", pred_z)):
        for sname, mask in subsets.items():
            probe[label][sname] = _plain_metrics(pred, nuD_te, mask)

    if verbose:
        print(f"\n===== {run_dir.name} (train n={len(nuD_tr)}, test n={len(nuD_te)}) =====")
        print(f"{'dim':5s} | {'global sp/pear/r2aff':>24s} | {'nu=0.01 sp/pear/r2aff':>24s}")
        for name, _ in vecs:
            g = dims_res[name]["global"]
            n1 = dims_res[name]["nu=0.01"]
            print(f"{name:5s} | {g['spearman']:6.3f} {g['pearson']:6.3f} {g['r2_affine']:6.3f} | "
                  f"{n1['spearman']:6.3f} {n1['pearson']:6.3f} {n1['r2_affine']:6.3f}")
        print(f"{'probe':8s} {'subset':9s} {'spearman':>8s} {'pearson':>8s} {'r2':>7s} {'rel_rmse':>8s}")
        for label in ("h_linear", "z_linear"):
            for sname in ("global", "nu=0.01"):
                m = probe[label][sname]
                print(f"{label:8s} {sname:9s} {m['spearman']:8.3f} {m['pearson']:8.3f} "
                      f"{m['r2']:7.3f} {m['rel_rmse']:8.3f}")

    return {
        "window_size": window_size,
        "protocol": protocol,
        "n_train": int(len(nuD_tr)),
        "n_test": int(len(nuD_te)),
        "dims": dims_res,
        "dims_vs_D_E": dims_vs_de,
        "probe": probe,
    }, rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[25, 50])
    parser.add_argument("--protocols", nargs="+", default=["standard", "heldout_nu"],
                        choices=["standard", "heldout_nu"])
    args = parser.parse_args(argv)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"experiment": "window_a_h_dims_analysis", "runs": {}}
    probe_summary = {"experiment": "window_a_h_linear_probe", "runs": {}}
    all_rows = []
    # découverte des runs : window_w{w}_{protocole}[_z{mode}] avec best_model.pt
    run_dirs = sorted(
        d for d in MODELS_DIR.glob("window_w*")
        if d.is_dir() and (d / "best_model.pt").exists()
    )
    for run_dir in run_dirs:
        m = re.match(r"window_w(\d+)_(standard|heldout_nu)(?:_z(\w+))?(?:_s(\d+))?$", run_dir.name)
        if not m:
            print(f"[skip] {run_dir.name} (nom non reconnu)")
            continue
        w = int(m.group(1))
        protocol = m.group(2)
        if w not in args.window_sizes or protocol not in args.protocols:
            continue
        res, rows = analyze_run(run_dir, w, protocol)
        all_rows.extend(rows)
        summary["runs"][run_dir.name] = res
        probe_summary["runs"][run_dir.name] = {
            "window_size": res["window_size"], "protocol": res["protocol"],
            "n_train": res["n_train"], "n_test": res["n_test"],
            "probe": res["probe"], "dims_vs_D_E": res["dims_vs_D_E"],
        }

    (LOGS_DIR / "h_dims_window_a_analysis.json").write_text(json.dumps(summary, indent=1))
    (LOGS_DIR / "h_linear_probe_window_a.json").write_text(json.dumps(probe_summary, indent=1))
    with open(LOGS_DIR / "h_dims_window_a_analysis.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "dim", "subset", "metric", "value"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nEcrit: {LOGS_DIR / 'h_dims_window_a_analysis.json'}")
    print(f"Ecrit: {LOGS_DIR / 'h_dims_window_a_analysis.csv'}")
    print(f"Ecrit: {LOGS_DIR / 'h_linear_probe_window_a.json'}")


if __name__ == "__main__":
    main()
