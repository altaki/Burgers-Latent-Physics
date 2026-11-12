"""Lanceur CLI des runs de l'expérience TD (prereg outputs/prereg/td_encoder_prereg.json,
commit 515cf5b — lu avant tout lancement).

Bras : td0 (encodeur actuel, référence) / td1 (différences temporelles explicites) /
td2 (td1 + global average pooling) / td3 (td2 + canal produit u_t ⊙ Δ_1 u, injection
architecturale, diagnostic). Modes : energy_centered (L_z, λ_z=1, λ_mono=0) et none.
Protocoles : heldout_nu (principal, évaluation ν=0.01) et heldout_nu005 (secondaire, ν=0.05).
Graines prereg : 42 7 123 2024 999.

AUCUN run n'est lancé par défaut : ce script construit et exécute UNE commande.
Nommage (jamais d'écrasement, abort si le dossier existe) :
  td{bras}_{protocole}_zenergy_centered_s{seed}   (mode energy_centered)
  td{bras}_{protocole}_s{seed}                    (mode none)

Règles bloquantes (prereg §9) :
  - le working tree doit être propre (hors __pycache__) : le run refuse de démarrer
    sinon (sauf --smoke) ;
  - lancer d'abord `python3 main_determinism_check.py` (exit 0 requis).

Commandes exactes (env torch, python3, racine du repo) :
  python3 main_determinism_check.py
  python3 main_train_td.py --encoder td0 --protocol heldout_nu --z-target-mode energy_centered --seed 42   # TD0 d'abord
  # puis boucle : bras in td0 td1 td2 td3, mode in energy_centered none,
  #               seed in 42 7 123 2024 999, protocole in heldout_nu heldout_nu005
  python3 main_train_td.py --encoder {bras} --protocol {protocole} --z-target-mode {mode} --seed {seed}

Smoke (2 époques, train tronqué à 20 trajectoires, copie temporaire dans le dossier
temp, AUCUNE écriture dans outputs/, working tree non exigé propre) :
  python3 main_train_td.py --encoder td2 --protocol heldout_nu --z-target-mode energy_centered --seed 42 --smoke

Évaluation (lecture seule) : python3 main_evaluate_td.py
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
MODELS = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"

PROTOCOLS = {
    "heldout_nu": {"heldout_nu_values": (0.01,), "nu_strat": 0.01},
    "heldout_nu005": {"heldout_nu_values": (0.01, 0.2), "nu_strat": 0.05},
}
W = 25
ENCODERS = ("td0", "td1", "td2", "td3")
SEEDS_PREREG = (42, 7, 123, 2024, 999)


def run_one(encoder: str, protocol: str, z_target_mode: str, seed: int, smoke: bool = False) -> Path:
    import numpy as np
    import ml_train_experiment_a_seq as mt

    if encoder not in ENCODERS:
        raise SystemExit(f"--encoder doit être dans {ENCODERS}, reçu {encoder!r}")
    if protocol not in PROTOCOLS:
        raise SystemExit(f"--protocol doit être dans {list(PROTOCOLS)}, reçu {protocol!r}")
    if z_target_mode not in ("energy_centered", "none"):
        raise SystemExit("--z-target-mode doit être energy_centered ou none")

    suffix = f"td{encoder[2:]}_{protocol}" + (f"_z{z_target_mode}" if z_target_mode != "none" else "")
    out = MODELS / f"{suffix}_s{seed}"
    if out.exists() and not smoke:
        raise SystemExit(f"run existant, abandon (pas d'écrasement) : {out}")

    train_path = DATA / "train_data_nu_sweep.npz"
    val_path = DATA / "val_data_nu_sweep.npz"
    test_path = DATA / "test_data_nu_sweep.npz"
    if smoke:
        # copie tronquée (20 premières trajectoires) dans le dossier temp :
        # les npz de data restent en lecture seule
        tmp = Path(tempfile.mkdtemp(prefix="td_smoke_"))
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
        out = Path(tempfile.mkdtemp(prefix="td_smoke_out_")) / out.name

    cfg = PROTOCOLS[protocol]
    mt.train_and_evaluate_a_seq(
        protocol=protocol,
        train_path=train_path, val_path=val_path, test_path=test_path,
        output_dir=out,
        window_size=W,
        heldout_nu_values=cfg["heldout_nu_values"],
        z_target_mode=z_target_mode,
        z_weight=1.0 if z_target_mode != "none" else 0.0,
        monotonicity_weight=0.0,
        nu_strat=cfg["nu_strat"],
        num_epochs=2 if smoke else 200,
        seed=seed,
        encoder_mode=encoder,
        run_tag=f"td{encoder[2:]}",
        require_clean_tree=not smoke,
        smoke=smoke,
    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", choices=ENCODERS, required=True)
    ap.add_argument("--protocol", choices=list(PROTOCOLS), required=True)
    ap.add_argument("--z-target-mode", choices=("energy_centered", "none"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="2 époques, données tronquées (tmp), aucune écriture outputs/, working tree non exigé propre")
    args = ap.parse_args()
    out = run_one(args.encoder, args.protocol, args.z_target_mode, args.seed, smoke=args.smoke)
    print("run terminé ->", out)


if __name__ == "__main__":
    main()
