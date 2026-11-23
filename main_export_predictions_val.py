#!/usr/bin/env python3
"""Point d'entrée ADDITIF — export des prédictions de VALIDATION des runs TD existants.

Pré-enregistrement : `manuscript/prereg_nu002_strate_test.md` §4 (calibration hors
strate sur validation). INFÉRENCE SEULE : lit `best_model.pt` + `standardizers.json`,
aucun réentraînement, aucun `.lock`.

Sécurité :
- n'écrit JAMAIS dans les répertoires de runs (`outputs/models/...`) : exports dans
  `--out-dir` (défaut `outputs/predictions_val/`), un `predictions_val.npz` par run ;
- refus si le fichier de sortie existe (aucune écrasement) ;
- hash SHA-256 de chaque `best_model.pt` AVANT et APRÈS (preuve de lecture seule) +
  manifeste JSON (`--manifest`) avec les hash des exports.

Usage (env torch requis, Windows/PowerShell) :
  python3 main_export_predictions_val.py --run-dir outputs/models/latent_dynamics_autoencoder_seq/td0_heldout_nu005_zenergy_centered_s42
  python3 main_export_predictions_val.py --all-td
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "outputs" / "data"
MODELS_ROOT = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"

TRAIN_NPZ = DATA_DIR / "train_data_nu_sweep.npz"
VAL_NPZ = DATA_DIR / "val_data_nu_sweep.npz"
TEST_NPZ = DATA_DIR / "test_data_nu_sweep.npz"
HELDOUT_NU = (0.05, 0.2)
HELDOUT_NU005 = (0.01, 0.2)

TD_RE = re.compile(r"^(td[0-3])_(heldout_nu005|heldout_nu)(?:_z(\w+))?_s(\d+)$")
WINDOW_SIZE = 25  # prereg td_encoder_prereg.md §3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_td_run_val(run_dir: Path, out_dir: Path, device: str = "cpu") -> dict:
    import torch  # import tardif : --help / py_compile fonctionnent sans torch
    import ml_train_experiment_a_seq as mt

    m = TD_RE.match(run_dir.name)
    if m is None:
        raise ValueError(f"nom de run TD non reconnu: {run_dir.name!r}")
    bras, proto, zmode, seed = m.group(1), m.group(2), (m.group(3) or "none"), int(m.group(4))
    zmode = "energy_centered" if zmode == "zenergy_centered" else zmode

    ckpt_path = run_dir / "best_model.pt"
    sha_before = sha256_file(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = dict(ckpt.get("model_config") or {})
    cfg.pop("model", None)
    cfg.pop("loss", None)

    bundle = mt.build_a_seq_datasets(
        TRAIN_NPZ, VAL_NPZ, TEST_NPZ,
        window_size=WINDOW_SIZE,
        nu_filter_train=(HELDOUT_NU005 if proto == "heldout_nu005" else HELDOUT_NU),
    )
    ec_models = mt._load_experiment_c_models_module()
    model = ec_models.build_window_dynamics_autoencoder(**cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    pred = mt.predict_a_seq(model, bundle["val"], device=torch.device(device))
    z = pred["z"]
    std_path = run_dir / "standardizers.json"
    if std_path.exists() and zmode != "none":
        std = json.loads(std_path.read_text(encoding="utf-8")).get("z_target_standardizer")
        if std:
            z = z * float(std["std"]) + float(std["mean"])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_dir.name}_predictions_val.npz"
    if out_path.exists():
        raise FileExistsError(f"refus d'écrasement : {out_path} existe déjà")
    mt.save_predictions_npz(
        out_path,
        z=z, h=pred["h"], meta=pred["meta"], u=pred["u"], dx=bundle["val"].dx,
        split="val", protocol=proto, seed=seed, window_size=WINDOW_SIZE,
        z_target_mode=zmode, run_name=run_dir.name,
        extra={
            "source": np.array("td_val_export"),
            "checkpoint_path": np.array(str(ckpt_path)),
            "checkpoint_sha256_before": np.array(sha_before),
        },
    )
    sha_after = sha256_file(ckpt_path)
    entry = {
        "run": run_dir.name, "bras": bras, "protocol": proto, "z_target_mode": zmode, "seed": seed,
        "checkpoint_sha256_before": sha_before, "checkpoint_sha256_after": sha_after,
        "checkpoint_unchanged": sha_before == sha_after,
        "export": str(out_path), "export_sha256": sha256_file(out_path),
        "n_windows": int(np.asarray(pred["meta"]["nu"]).shape[0]) if "nu" in pred["meta"] else None,
    }
    print(f"[val-export] {run_dir.name} -> {out_path} (checkpoint inchangé: {entry['checkpoint_unchanged']})")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", default=[],
                    help="répertoire de run TD (répétable)")
    ap.add_argument("--all-td", action="store_true", help="tous les runs td[0-3]_* de MODELS_ROOT")
    ap.add_argument("--out-dir", type=Path, default=REPO / "outputs" / "predictions_val")
    ap.add_argument("--manifest", type=Path, default=REPO / "outputs" / "logs" / "predictions_val_manifest.json")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    run_dirs = list(args.run_dir)
    if args.all_td:
        run_dirs += sorted(p for p in MODELS_ROOT.iterdir() if p.is_dir() and TD_RE.match(p.name))
    if not run_dirs:
        ap.error("préciser --run-dir ... et/ou --all-td")
    if args.manifest.exists():
        print(f"[ABORT] manifeste existe déjà : {args.manifest} (aucune écrasement)")
        return 1

    entries = []
    for rd in run_dirs:
        entries.append(export_td_run_val(rd.resolve(), args.out_dir.resolve(), args.device))
    manifest = {
        "prereg": "manuscript/prereg_nu002_strate_test.md",
        "inference_only": True, "retraining": False,
        "out_dir": str(args.out_dir), "n_runs": len(entries),
        "all_checkpoints_unchanged": all(e["checkpoint_unchanged"] for e in entries),
        "runs": entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[val-export] manifeste -> {args.manifest} ; checkpoints tous inchangés : {manifest['all_checkpoints_unchanged']}")
    return 0 if manifest["all_checkpoints_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
