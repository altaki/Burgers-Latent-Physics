"""
main_experiment_c2.py
=====================

Experiment C2 : fenêtres temporelles longues pour la variante `c3_window`
(w = 5, 25, 50) sur le dataset `nu_sweep_v1`, avec les mêmes cibles, protocoles
et métriques que l'Experiment C.

Objectif : tester si l'information sur ν — qui semble principalement temporelle
(résultats C : C3 > C2 > C1, et dE/dt = −νD exact dans les données) — devient
identifiable quand le contexte temporel s'allonge. On compare w = 5 (baseline C3,
0.08 s), w = 25 (0.48 s) et w = 50 (0.98 s, moitié de T=2).

Usage
-----
Depuis la racine du projet (environnement torch) :

    python main_experiment_c2.py                          # w = 5 25 50, standard + heldout_nu
    python main_experiment_c2.py --window-sizes 25 50     # sous-ensemble de fenêtres
    python main_experiment_c2.py --protocols standard     # un seul protocole

Implémentation : réutilise intégralement l'infrastructure Experiment C
(`main_experiment_c.run_experiment_c` avec `window_sizes=[...]`). Aucun module
d'entraînement n'est modifié : `window_size` était déjà câblé de bout en bout
(datasets fenêtre -> modèle c3_window -> métriques -> verdict).

Sorties :
- modèles/métriques : `outputs/models/experiment_c/c3_window_w{w}_{protocole}/`
  (labels explicites pour ne pas écraser les runs C historiques `c3_window_*`),
- résumé + verdicts : `outputs/logs/experiment_c_summary_c2.json` et
  `experiment_c_config_snapshot_c2.json` (tag "c2" ; les rapports C d'origine
  `experiment_c_summary.json` ne sont pas touchés).

Prédictions à falsifier (voir README, Step 4 — Experiment C2) : si R² νD et
rel_rmse à ν=0.01 held-out ne s'améliorent pas quand w croît de 5 à 50, alors
l'hypothèse « l'information sur ν est temporelle » est réfutée et le bloqueur
est le dataset (3 strates de ν, held-out extrême), pas la taille de fenêtre.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Sequence

import main_experiment_c as mec


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment C2 : fenêtres temporelles longues (c3_window, w=5/25/50) sur nu_sweep_v1."
    )
    parser.add_argument(
        "--window-sizes", nargs="*", type=int, default=[5, 25, 50], metavar="W",
        help="Tailles de fenêtre testées (défaut: 5 25 50).",
    )
    parser.add_argument(
        "--protocols", nargs="*", default=["standard", "heldout_nu"],
        choices=["standard", "heldout_nu"],
        help="Protocoles à exécuter (défaut: standard heldout_nu).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Lance l'Experiment C2 via l'infrastructure Experiment C (tag 'c2')."""
    args = _parse_args(argv)
    config_module = mec._load_config_module()
    config = config_module.get_default_config(create_dirs=True)
    return mec.run_experiment_c(
        config,
        variants=["c3_window"],
        protocols=args.protocols,
        window_sizes=args.window_sizes,
        tag="c2",
        verbose=True,
    )


if __name__ == "__main__":
    main()
