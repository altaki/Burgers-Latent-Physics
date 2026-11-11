"""Lanceur CLI des 6 runs B1 (protocole heldout_nu005, interpolation nu=0.05).

Pré-enregistrement : outputs/prereg/b1_interpolation_prereg.json (commit f431e66).
AUCUN run n'est lancé par défaut : ce script ne fait que construire la commande.

Commandes exactes (env torch, depuis la racine du repo) :
  python main_train_b1.py --z-target-mode energy_centered --seed 42
  python main_train_b1.py --z-target-mode energy_centered --seed 7
  python main_train_b1.py --z-target-mode energy_centered --seed 123
  python main_train_b1.py --z-target-mode none --seed 42
  python main_train_b1.py --z-target-mode none --seed 7
  python main_train_b1.py --z-target-mode none --seed 123

Smoke test (2 époques, train tronqué à 20 trajectoires, copie temporaire dans /tmp,
aucune écriture dans outputs/) :
  python main_train_b1.py --z-target-mode energy_centered --seed 42 --smoke

Ne réécrit jamais un run existant : output_dir = window_w25_heldout_nu005[_z{mode}]_s{seed}.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
MODELS = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"

PROTOCOL = "heldout_nu005"
HELDOUT_NU005 = (0.01, 0.2)
W = 25
NU_STRAT = 0.05


def run_one(z_target_mode: str, seed: int, smoke: bool = False) -> Path:
    import numpy as np
    import ml_train_experiment_a_seq as mt

    suffix = f"w{W}_{PROTOCOL}" + (f"_z{z_target_mode}" if z_target_mode != "none" else "")
    out = MODELS / f"{suffix}_s{seed}"
    if out.exists() and not smoke:
        raise SystemExit(f"run existant, abandon (pas d'écrasement) : {out}")

    train_path = DATA / "train_data_nu_sweep.npz"
    val_path = DATA / "val_data_nu_sweep.npz"
    test_path = DATA / "test_data_nu_sweep.npz"
    if smoke:
        # copie tronquée (20 premières trajectoires) dans /tmp : les npz de data
        # restent en lecture seule
        tmp = Path(tempfile.mkdtemp(prefix="b1_smoke_"))
        for name, src in (("train", train_path), ("val", val_path), ("test", test_path)):
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
            np.savez(tmp / f"{name}_data_nu_sweep.npz", **d)
        train_path = tmp / "train_data_nu_sweep.npz"
        val_path = tmp / "val_data_nu_sweep.npz"
        test_path = tmp / "test_data_nu_sweep.npz"
        out = Path(tempfile.mkdtemp(prefix="b1_smoke_out_")) / out.name

    mt.train_and_evaluate_a_seq(
        protocol=PROTOCOL,
        train_path=train_path, val_path=val_path, test_path=test_path,
        output_dir=out,
        window_size=W,
        heldout_nu_values=HELDOUT_NU005,
        z_target_mode=z_target_mode,
        z_weight=1.0 if z_target_mode != "none" else 0.0,
        monotonicity_weight=0.0,
        nu_strat=NU_STRAT,
        num_epochs=2 if smoke else 200,
        seed=seed,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--z-target-mode", choices=("energy_centered", "none"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="2 époques, données tronquées (tmp), aucune écriture outputs/")
    args = ap.parse_args()
    out = run_one(args.z_target_mode, args.seed, smoke=args.smoke)
    print("run terminé ->", out)


if __name__ == "__main__":
    main()
