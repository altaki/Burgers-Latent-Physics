"""Validation bit-à-bit d'un run de production existant — AUCUN écrasement.

Deux modes :
  - mode relance (défaut) : relance UN run de production complet dans un dossier
    TEMPORAIRE (aucune écriture dans outputs/models/, les artefacts existants sont
    lus uniquement) puis compare au run de référence ;
  - mode --compare-only <dossier> : compare un dossier de run DÉJÀ produit (sans
    réentraîner) au run de référence. Ne charge pas torch.

Comparaison :
  - metrics JSON : feuilles numériques, hors clés exclues explicites
    (METRICS_EXCLUDED_KEYS : traceability, checkpoint_path, best_model_path) ;
  - predictions_test.npz : TOUS les tableaux, chacun affiché (dtype, shape, écart
    max ou égalité), hors champs exclus explicites (NPZ_EXCLUDED_FIELDS :
    checkpoint_path, chemin absolu nécessairement différent hors du dossier de
    référence). Aucun motif large d'exclusion.

Exit 0 si et seulement si : toutes les feuilles numériques du metrics ont un
écart de 0.0, tous les tableaux numériques du npz ont un écart max de 0.0, et
tous les champs non numériques non exclus sont identiques, sans clé présente d'un
seul côté. Exit 1 sinon ou si artefacts de référence manquants.

Usage (env torch, python3, racine du repo) :
  python3 main_validate_run.py --encoder td0 --protocol heldout_nu005 --z-target-mode energy_centered --seed 42
  python3 main_validate_run.py --encoder td0 --protocol heldout_nu005 --z-target-mode energy_centered --seed 42 \
      --compare-only C:\\Users\\baltaki\\AppData\\Local\\Temp\\td_validate_XXXX\\td0_heldout_nu005_zenergy_centered_s42
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

# Exclusions EXPLICITES et minimales (noms exacts, aucun motif) :
# - metrics JSON : bloc traceability (hash git, horodatages, versions, dirty) et
#   chemins absolus de checkpoint ;
# - npz : checkpoint_path (champ 'extra' ajouté par save_predictions_npz) = chemin
#   absolu du best_model.pt, nécessairement différent quand le run est relancé
#   dans un dossier temporaire.
METRICS_EXCLUDED_KEYS = ("traceability", "checkpoint_path", "best_model_path")
NPZ_EXCLUDED_FIELDS = ("checkpoint_path",)

import td_protocols as tp  # source de vérité unique des protocoles (correction bug filtre)

PROTOCOLS = tp.PROTOCOLS  # {nom: {"train_val_nu": ..., "nu_strat": ...}}
W = 25


def _strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in METRICS_EXCLUDED_KEYS}
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


def _compare_metrics(ref_path: Path, new_path: Path):
    """Compare les feuilles numériques du metrics JSON. Retourne (ok, bilan)."""
    m_ref = json.load(open(ref_path))
    m_new = json.load(open(new_path))
    l_ref = dict(_numeric_leaves(_strip_volatile(m_ref)))
    l_new = dict(_numeric_leaves(_strip_volatile(m_new)))
    only_ref = sorted(set(l_ref) - set(l_new))
    only_new = sorted(set(l_new) - set(l_ref))
    if only_ref or only_new:
        print(f"  metrics : clés présentes d'un seul côté — ref: {only_ref[:10]} | new: {only_new[:10]}")
    diffs = []
    for k in sorted(set(l_ref) & set(l_new)):
        a, b = l_ref[k], l_new[k]
        if np.isnan(a) and np.isnan(b):
            continue
        d = abs(a - b)
        if not (d == 0.0):
            diffs.append((d, k))
    diffs.sort(reverse=True)
    worst, worst_key = diffs[0] if diffs else (0.0, "")
    n = len(set(l_ref) & set(l_new))
    print(f"  metrics JSON : {n} valeurs numériques comparées, {len(diffs)} écart(s) non nul(s), "
          f"écart max = {worst:.3e} ({worst_key})")
    for d, k in diffs[:10]:
        print(f"    {d:.3e}  {k}")
    ok = not only_ref and not only_new and not diffs
    return ok


def _num_max_diff(a: np.ndarray, b: np.ndarray):
    """Écart max en considérant NaN==NaN à la même position. None si positions NaN différentes."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = np.isnan(a), np.isnan(b)
    if not np.array_equal(na, nb):
        return None
    m = ~na
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m])))


def _compare_npz(ref_path: Path, new_path: Path):
    """Compare TOUS les tableaux du npz, un par un. Retourne (ok, bilan)."""
    p_ref = np.load(ref_path, allow_pickle=True)
    p_new = np.load(new_path, allow_pickle=True)
    only_ref = [k for k in p_ref.files if k not in p_new.files]
    only_new = [k for k in p_new.files if k not in p_ref.files]
    if only_ref or only_new:
        print(f"  npz : clés présentes d'un seul côté — ref: {only_ref} | new: {only_new}")
    excluded = [k for k in p_ref.files if k in NPZ_EXCLUDED_FIELDS]
    print(f"  npz : {len(p_ref.files)} tableaux, exclus explicites : {excluded}")
    n_num, n_str, worst, worst_key, ok = 0, 0, 0.0, "", True
    for k in p_ref.files:
        if k in NPZ_EXCLUDED_FIELDS:
            print(f"    {k:20s} {str(p_ref[k].dtype):8s} {str(p_ref[k].shape):12s} EXCLU (chemin)")
            continue
        if k not in p_new.files:
            print(f"    {k:20s} ABSENT côté nouveau")
            ok = False
            continue
        a, b = p_ref[k], p_new[k]
        if a.dtype.kind in "fiu":
            n_num += 1
            if a.shape != b.shape:
                print(f"    {k:20s} {str(a.dtype):8s} shape ref={a.shape} new={b.shape}  ÉCHEC SHAPE")
                ok = False
                continue
            d = _num_max_diff(a, b)
            if d is None:
                print(f"    {k:20s} {str(a.dtype):8s} {str(a.shape):12s} positions NaN différentes  ÉCHEC")
                ok = False
                continue
            if d > worst:
                worst, worst_key = d, k
            print(f"    {k:20s} {str(a.dtype):8s} {str(a.shape):12s} max|diff|={d:.3e}")
            if not (d == 0.0):
                ok = False
        else:
            n_str += 1
            eq = a.shape == b.shape and np.array_equal(a, b)
            print(f"    {k:20s} {str(a.dtype):8s} {str(a.shape):12s} {'identique' if eq else 'DIFFÉRENT'}")
            if not eq:
                print(f"      ref={a.item()!r}")
                print(f"      new={b.item()!r}")
                ok = False
    for k in only_new:
        print(f"    {k:20s} présent uniquement côté nouveau")
    print(f"  npz : bilan {n_num} tableaux numériques (écart max = {worst:.3e} sur {worst_key}), "
          f"{n_str} non numériques, {len(excluded)} exclu(s)")
    if only_ref or only_new:
        ok = False
    return ok


def _find_artifacts(run_dir: Path):
    m = next(run_dir.glob("metrics_window_a_*.json"), None)
    n = run_dir / "predictions_test.npz"
    return m, (n if n.exists() else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--encoder", default="td0")
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="heldout_nu005")
    ap.add_argument("--z-target-mode", choices=("energy_centered", "none"), default="energy_centered")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reference", default=None, help="dossier du run de référence (défaut: nommage standard)")
    ap.add_argument("--compare-only", default=None, metavar="DOSSIER",
                    help="compare ce dossier de run déjà produit, sans réentraîner (aucun import torch)")
    args = ap.parse_args()

    suffix = f"td{args.encoder[2:]}_{args.protocol}" + (
        f"_z{args.z_target_mode}" if args.z_target_mode != "none" else "")
    ref_dir = Path(args.reference) if args.reference else MODELS / f"{suffix}_s{args.seed}"
    ref_metrics, ref_npz = _find_artifacts(ref_dir)
    if ref_metrics is None or ref_npz is None:
        print(f"ÉCHEC : artefacts de référence manquants dans {ref_dir}")
        return 1
    print(f"référence : {ref_dir}")

    if args.compare_only:
        new_dir = Path(args.compare_only)
        print(f"comparaison sans réentraînement : {new_dir}")
    else:
        import ml_train_experiment_a_seq as mt  # import torch uniquement en mode relance

        tmp = Path(tempfile.mkdtemp(prefix="td_validate_"))
        new_dir = tmp / ref_dir.name
        cfg = PROTOCOLS[args.protocol]
        tp.check_against_prereg()
        print(f"relance du run dans {new_dir} (dossier temporaire, aucun écrasement) ...")
        mt.train_and_evaluate_a_seq(
            protocol=args.protocol,
            train_path=DATA / "train_data_nu_sweep.npz",
            val_path=DATA / "val_data_nu_sweep.npz",
            test_path=DATA / "test_data_nu_sweep.npz",
            output_dir=new_dir,
            window_size=W,
            heldout_nu_values=cfg["train_val_nu"],
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
            expected_train_nu=cfg["train_val_nu"],
        )

    new_metrics, new_npz = _find_artifacts(new_dir)
    if new_metrics is None or new_npz is None:
        print(f"ÉCHEC : artefacts manquants dans {new_dir}")
        return 1

    print("comparaison metrics JSON :")
    ok_m = _compare_metrics(ref_metrics, new_metrics)
    print("comparaison predictions_test.npz :")
    ok_n = _compare_npz(ref_npz, new_npz)

    ok = ok_m and ok_n
    print(f"{'BIT-À-BIT IDENTIQUE (exit 0)' if ok else 'ÉCART NON NUL OU CHAMP EXCLU DIFFÉRENT (exit 1)'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
