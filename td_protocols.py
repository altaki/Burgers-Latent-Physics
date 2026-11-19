"""Source de vérité UNIQUE des protocoles de l'expérience TD.

Correction de bug (filtre heldout_nu inversé : (0.01,) au lieu de (0.05, 0.2),
main_train_td.py:48) : les dictionnaires de protocoles dupliqués dans
main_train_td.py, main_validate_run.py et main_determinism_check.py sont
supprimés ; tous les scripts importent ce module. La référence normative est le
pré-enregistrement outputs/prereg/td_encoder_prereg.json (commit 515cf5b) :
`train_val_nu` par protocole = ensemble ν RETENU pour train/val (inclusion,
sémantique de ml_datasets_experiment_c.py:_build_nu_mask).

Test automatique : `python3 td_protocols.py --check`
  1. PROTOCOLS == prereg JSON (train_val_nu et ν d'évaluation) ;
  2. pour chaque protocole, le filtrage du VRAI train npz (lecture seule) par
     train_val_nu produit exactement l'ensemble du prereg.
Exit 0 si conforme, exit 1 sinon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
PREREG = REPO / "outputs" / "prereg" / "td_encoder_prereg.json"
TRAIN_NPZ = REPO / "outputs" / "data" / "train_data_nu_sweep.npz"

PROTOCOLS = {
    "heldout_nu": {"train_val_nu": (0.05, 0.2), "nu_strat": 0.01},
    "heldout_nu005": {"train_val_nu": (0.01, 0.2), "nu_strat": 0.05},
}


def prereg_protocols() -> dict:
    p = json.load(open(PREREG))
    return {
        q["name"]: {
            "train_val_nu": tuple(float(v) for v in q["train_val_nu"]),
            "nu_eval_principal": float(q["nu_eval_principal"]),
        }
        for q in p["protocoles"]
    }


def check_against_prereg() -> None:
    pr = prereg_protocols()
    if set(pr) != set(PROTOCOLS):
        raise SystemExit(f"protocoles module {sorted(PROTOCOLS)} != prereg {sorted(pr)}")
    for name, cfg in PROTOCOLS.items():
        if tuple(float(v) for v in cfg["train_val_nu"]) != pr[name]["train_val_nu"]:
            raise SystemExit(
                f"train_val_nu incohérent pour {name} : module={cfg['train_val_nu']} "
                f"prereg={pr[name]['train_val_nu']}"
            )
        if float(cfg["nu_strat"]) != pr[name]["nu_eval_principal"]:
            raise SystemExit(
                f"nu_eval incohérent pour {name} : module={cfg['nu_strat']} "
                f"prereg={pr[name]['nu_eval_principal']}"
            )


def effective_train_nu(train_val_nu) -> list:
    """Ensemble ν réellement retenu par le filtre sur le vrai train npz
    (même sémantique que _build_nu_mask : inclusion par isclose)."""
    nu = np.load(TRAIN_NPZ, allow_pickle=True)["nu_values"].astype(float)
    allowed = np.asarray([float(v) for v in train_val_nu], dtype=float)
    mask = np.any(np.isclose(nu[:, None], allowed[None, :]), axis=1)
    return sorted({float(v) for v in nu[mask]})


def check_filter_on_data() -> None:
    pr = prereg_protocols()
    for name, cfg in PROTOCOLS.items():
        eff = effective_train_nu(cfg["train_val_nu"])
        exp = sorted(pr[name]["train_val_nu"])
        if not (len(eff) == len(exp) and all(np.isclose(a, b) for a, b in zip(eff, exp))):
            raise SystemExit(f"filtrage incohérent pour {name} : effectif={eff} attendu={exp}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_against_prereg()
        check_filter_on_data()
        for name, cfg in PROTOCOLS.items():
            print(f"OK {name}: train_val_nu={cfg['train_val_nu']} nu_eval={cfg['nu_strat']} "
                  "(module == prereg == filtrage effectif du train npz)")
        sys.exit(0)
    print(__doc__)
    sys.exit(0)
