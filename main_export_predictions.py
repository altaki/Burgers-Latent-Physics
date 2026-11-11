"""Réinfère les checkpoints existants et exporte `predictions_test.npz` au format
standard (le même que celui écrit par `train_and_evaluate_a_seq`), pour
bootstrapper les références historiques (window-A w25/w50, C2 c3_window w25)
de la même manière que les futurs runs A-Physics.

Usage (env torch requis) :
    python main_export_predictions.py --which all
    python main_export_predictions.py --which aseq
    python main_export_predictions.py --which c3
    python main_export_predictions.py --run-dir outputs/models/latent_dynamics_autoencoder_seq/window_w25_heldout_nu

Ne réentraîne rien : lit best_model.pt + standardizers.json + le npz test
(lecture seule) et écrit predictions_test.npz dans le répertoire du run.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "outputs" / "data"
MODELS_ROOT = REPO / "outputs" / "models"

TRAIN_NPZ = DATA_DIR / "train_data_nu_sweep.npz"
VAL_NPZ = DATA_DIR / "val_data_nu_sweep.npz"
TEST_NPZ = DATA_DIR / "test_data_nu_sweep.npz"
HELDOUT_NU = (0.05, 0.2)
HELDOUT_NU005 = (0.01, 0.2)

ASEQ_RE = re.compile(r"window_w(\d+)_(standard|heldout_nu005|heldout_nu)(?:_z(\w+))?(?:_s(\d+))?$")
EC_RE = re.compile(r"(c\d_\w+?)(?:_w(\d+))?_(standard|heldout_nu)$")


def _nuD_est_centered(energy: np.ndarray, dt: float) -> np.ndarray:
    """(E_{t-1} - E_{t+1}) / (2 dt), repli décentré aux bords (même règle que
    NuSweepWindowPairDataset)."""
    n_times = energy.shape[1]
    est = np.empty_like(energy, dtype=float)
    if n_times > 2:
        est[:, 1:-1] = (energy[:, :-2] - energy[:, 2:]) / (2.0 * dt)
    est[:, 0] = (energy[:, 0] - energy[:, 1]) / dt
    est[:, -1] = (energy[:, -2] - energy[:, -1]) / dt
    return est


def export_a_seq_run(run_dir: Path, device: str = "cpu") -> Path:
    import torch
    import ml_train_experiment_a_seq as mt

    m = ASEQ_RE.match(run_dir.name)
    if m is None:
        raise ValueError(f"nom de run A-seq non reconnu: {run_dir.name!r}")
    w = int(m.group(1))
    proto = m.group(2)
    zmode = m.group(3) or "none"
    seed = int(m.group(4) or 42)

    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config") or {})
    cfg.pop("model", None)
    cfg.pop("loss", None)

    bundle = mt.build_a_seq_datasets(
        TRAIN_NPZ, VAL_NPZ, TEST_NPZ,
        window_size=w,
        nu_filter_train=(HELDOUT_NU005 if proto == "heldout_nu005" else HELDOUT_NU) if proto.startswith("heldout") else None,
    )
    ec_models = mt._load_experiment_c_models_module()
    model = ec_models.build_window_dynamics_autoencoder(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    pred = mt.predict_a_seq(model, bundle["test"], device=torch.device(device))
    z = pred["z"]
    # z en unités physiques si le run avait standardisé la cible
    std_path = run_dir / "standardizers.json"
    if std_path.exists() and zmode != "none":
        std = json.loads(std_path.read_text(encoding="utf-8")).get("z_target_standardizer")
        if std:
            z = z * float(std["std"]) + float(std["mean"])

    out = mt.save_predictions_npz(
        run_dir / "predictions_test.npz",
        z=z, h=pred["h"], meta=pred["meta"], u=pred["u"], dx=bundle["test"].dx,
        split="test", protocol=proto, seed=seed, window_size=w,
        z_target_mode=zmode, run_name=run_dir.name,
        extra={
            "source": np.array("window_a"),
            "checkpoint_path": np.array(str(run_dir / "best_model.pt")),
        },
    )
    print(f"[aseq] {run_dir.name} -> {out}")
    return out


def export_c_run(run_dir: Path, device: str = "cpu") -> Path:
    import torch
    import ml_train_experiment_a_seq as mt  # pour save_predictions_npz (sans réentraîner)
    import ml_train_experiment_c as mtc

    m = EC_RE.match(run_dir.name)
    if m is None:
        raise ValueError(f"nom de run Experiment C non reconnu: {run_dir.name!r}")
    variant = m.group(1)
    w = int(m.group(2) or 5)
    proto = m.group(3)

    ec_datasets = mtc._load_experiment_c_datasets_module()
    ec_models = mtc._load_experiment_c_models_module()
    bundle = ec_datasets.build_experiment_c_datasets(
        TRAIN_NPZ, VAL_NPZ, TEST_NPZ,
        window_size=w,
        nu_filter_train=(HELDOUT_NU005 if proto == "heldout_nu005" else HELDOUT_NU) if proto.startswith("heldout") else None,
    )
    test_win = bundle["test"]["window"]
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model = ec_models.build_experiment_c_model(variant, nx=test_win.nx, window_size=w)
    model.load_state_dict(ckpt["model_state_dict"])

    out = mtc.predict_dataset_experiment_c(
        model, test_win, bundle["target_standardizer"], input_key="window",
        device=torch.device(device),
    )
    names = list(ec_datasets.EXPERIMENT_C_TARGET_NAMES)
    j = names.index("nu_dissipation")
    z = np.asarray(out["preds"][:, j], dtype=float)  # déjà en unités physiques

    # métadonnées enrichies depuis le npz test (lecture seule)
    raw = np.load(TEST_NPZ, allow_pickle=True)
    ti = np.asarray(out["meta"]["trajectory_index"], dtype=np.int64)
    tt = np.asarray(out["meta"]["time_index"], dtype=np.int64)
    energy = np.asarray(raw["energy"], dtype=float)
    diss = np.asarray(raw["dissipation"], dtype=float)
    nuD = np.asarray(raw["dissipation_physical"], dtype=float)
    states = np.asarray(raw["states"], dtype=float)
    times = np.asarray(raw["times"], dtype=float)
    dt = float(times[1] - times[0])
    dx = float(np.asarray(raw["dx"]).item())
    nuD_est = _nuD_est_centered(energy, dt)[ti, tt]
    ic = (np.asarray(raw["ic_type_ids"], dtype=np.int64)[ti]
          if "ic_type_ids" in raw else np.full(len(ti), -1, dtype=np.int64))
    meta: Dict[str, np.ndarray] = {
        "nu": np.asarray(out["meta"]["nu"], dtype=float),
        "time_t": np.asarray(out["meta"]["time"], dtype=float),
        "trajectory_index": ti,
        "time_index_t": tt,
        "energy0": np.asarray(out["meta"]["energy0"], dtype=float),
        "energy_t": energy[ti, tt],
        "dissipation_t": diss[ti, tt],
        "nu_dissipation_t": nuD[ti, tt],
        "nuD_est_t": nuD_est,
        "ic_type_id": ic,
    }
    out_path = mt.save_predictions_npz(
        run_dir / "predictions_test.npz",
        z=z, h=np.zeros((len(z), 0), dtype=float), meta=meta,
        u=states[ti, tt], dx=dx,
        split="test", protocol=proto, seed=42, window_size=w,
        z_target_mode="supervised", run_name=run_dir.name,
        extra={
            "source": np.array("experiment_c"),
            "variant": np.array(variant),
            "checkpoint_path": np.array(str(run_dir / "best_model.pt")),
        },
    )
    print(f"[expC] {run_dir.name} -> {out_path}")
    return out_path


def discover(which: str) -> list:
    jobs = []
    if which in ("aseq", "all"):
        root = MODELS_ROOT / "latent_dynamics_autoencoder_seq"
        if root.exists():
            for d in sorted(root.iterdir()):
                if d.is_dir() and ASEQ_RE.match(d.name) and (d / "best_model.pt").exists():
                    jobs.append(("aseq", d))
    if which in ("c3", "all"):
        root = MODELS_ROOT / "experiment_c"
        if root.exists():
            for d in sorted(root.iterdir()):
                if d.is_dir() and EC_RE.match(d.name) and (d / "best_model.pt").exists():
                    if d.name.startswith("c3"):
                        jobs.append(("c3", d))
    return jobs


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--which", choices=["aseq", "c3", "all"], default="all")
    parser.add_argument("--run-dir", type=str, default=None,
                        help="Exporte un seul répertoire de run (chemin relatif ou absolu).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true", help="Réexporte même si predictions_test.npz existe.")
    args = parser.parse_args(argv)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = REPO / run_dir
        kind = "aseq" if ASEQ_RE.match(run_dir.name) else "c3"
        (export_a_seq_run if kind == "aseq" else export_c_run)(run_dir, device=args.device)
        return

    for kind, d in discover(args.which):
        if (d / "predictions_test.npz").exists() and not args.force:
            print(f"[skip] {d.name} (predictions_test.npz existe)")
            continue
        (export_a_seq_run if kind == "aseq" else export_c_run)(d, device=args.device)


if __name__ == "__main__":
    main()
