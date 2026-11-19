"""Validation bit-à-bit d'un run de production existant — AUCUN écrasement.

Relance UN run de production complet dans un dossier TEMPORAIRE (aucune écriture
dans outputs/models/, les artefacts existants sont lus uniquement) et compare
bit à bit aux artefacts du run de référence :
  - le metrics JSON (feuilles numériques, hors traceability / chemins) ;
  - tous les tableaux du predictions_test.npz.

Si l'écart max est 0.0, le run de référence est validé (reproductibilité
inter-session sur le même code / même machine / mêmes versions).

Usage (env torch, python3, racine du repo) :
  python3 main_validate_run.py --encoder td0 --protocol heldout_nu005 --z-target-mode energy_centered --seed 42
  # référence par défaut : outputs/models/latent_dynamics_autoencoder_seq/td0_heldout_nu005_zenergy_centered_s42
  python3 main_validate_run.py --encoder td0 --protocol heldout_nu --z-target-mode energy_centered --seed 42 \
      --reference outputs/models/latent_dynamics_autoencoder_seq/td0_heldout_nu_zenergy_centered_s42

Exit 0 si écart max == 0.0 partout ; exit 1 sinon ou si artefacts de référence
manquants. Ce script ne lance aucun entraînement tant qu'il n'est pas exécuté ;
l'entraînement lui-même est lancé par l'utilisateur.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
DATA = REPO / "outputs" / "data"
MODELS = REPO / "outputs" / "models" / "latent_dynamics_autoencoder_seq"

PROTOCOLS = {
    "heldout_nu": {"heldout_nu_values": (0.01,), "nu_strat": 0.01},
    "heldout_nu005": {"heldout_nu_values": (0.01, 0.2), "nu_strat": 0.05},
}
W = 25


def _strip_volatile(obj):
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="td0")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="heldout_nu005")
    ap.add_argument("--z-target-mode", choices=("energy_centered", "none"), default="energy_centered")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference", default=None, help="dossier du run de référence (défaut: nommage standard)")
    args = ap.parse_args()

    suffix = f"td{args.encoder[2:]}_{args.protocol}" + (
        f"_z{args.z_target_mode}" if args.z_target_mode != "none" else "")
    ref_dir = Path(args.reference) if args.reference else MODELS / f"{suffix}_s{args.seed}"
    ref_metrics = next(ref_dir.glob("metrics_window_a_*.json"), None)
    ref_npz = ref_dir / "predictions_test.npz"
    if ref_metrics is None or not ref_npz.exists():
        print(f"ÉCHEC : artefacts de référence manquants dans {ref_dir}")
        return 1
    print(f"référence : {ref_dir}")

    import ml_train_experiment_a_seq as mt

    tmp = Path(tempfile.mkdtemp(prefix="td_validate_"))
    out = tmp / ref_dir.name
    cfg = PROTOCOLS[args.protocol]
    print(f"relance du run dans {out} (dossier temporaire, aucun écrasement) ...")
    mt.train_and_evaluate_a_seq(
        protocol=args.protocol,
        train_path=DATA / "train_data_nu_sweep.npz",
        val_path=DATA / "val_data_nu_sweep.npz",
        test_path=DATA / "test_data_nu_sweep.npz",
        output_dir=out,
        window_size=W,
        heldout_nu_values=cfg["heldout_nu_values"],
        z_target_mode=args.z_target_mode,
        z_weight=1.0 if args.z_target_mode != "none" else 0.0,
        monotonicity_weight=0.0,
        nu_strat=cfg["nu_strat"],
        num_epochs=200,
        seed=args.seed,
        encoder_mode=args.encoder,
        run_tag=f"td{args.encoder[2:]}",
        require_clean_tree=True,
        smoke=False,
    )

    new_metrics = next(out.glob("metrics_window_a_*.json"), None)
    new_npz = out / "predictions_test.npz"
    if new_metrics is None or not new_npz.exists():
        print("ÉCHEC : le run relancé n'a pas produit metrics/predictions")
        return 1

    m_ref = json.load(open(ref_metrics))
    m_new = json.load(open(new_metrics))
    l_ref = dict(_numeric_leaves(_strip_volatile(m_ref)))
    l_new = dict(_numeric_leaves(_strip_volatile(m_new)))
    if set(l_ref) != set(l_new):
        print("ÉCHEC : clés numériques différentes")
        print("  seulement ref :", sorted(set(l_ref) - set(l_new))[:10])
        print("  seulement new :", sorted(set(l_new) - set(l_ref))[:10])
        return 1
    worst, worst_key = 0.0, ""
    for k in l_ref:
        d = abs(l_ref[k] - l_new[k])
        if d > worst:
            worst, worst_key = d, k
    print(f"metrics JSON : {len(l_ref)} valeurs numériques, écart max = {worst:.3e} ({worst_key})")

    p_ref = np.load(ref_npz, allow_pickle=True)
    p_new = np.load(new_npz, allow_pickle=True)
    if set(p_ref.files) != set(p_new.files):
        print("ÉCHEC : clés npz différentes")
        return 1
    worst_npz, worst_arr = 0.0, ""
    for k in p_ref.files:
        a, b = p_ref[k], p_new[k]
        if a.dtype.kind in "fiu":
            if a.shape != b.shape:
                print(f"ÉCHEC : shape différente pour {k}")
                return 1
            d = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
            if d > worst_npz:
                worst_npz, worst_arr = d, k
        else:
            if not np.array_equal(a, b):
                print(f"ÉCHEC : tableau non numérique différent pour {k}")
                return 1
    print(f"predictions_test.npz : {len(p_ref.files)} tableaux, écart max = {worst_npz:.3e} ({worst_arr})")

    ok = worst == 0.0 and worst_npz == 0.0
    print(f"{'BIT-À-BIT IDENTIQUE (exit 0)' if ok else 'ÉCART NON NUL (exit 1)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
