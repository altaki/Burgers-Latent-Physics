"""
main_experiment_c.py
====================

Point d'entrée de l'expérience C : baseline supervisée de séparabilité des
observables physiques sur le dataset `nu_sweep_v1`, avant tout modèle latent.

Usage
-----
Depuis la racine du projet :

    python main_experiment_c.py                          # toutes les variantes x protocoles
    python main_experiment_c.py --variants c1_mlp        # première run de production
    python main_experiment_c.py --variants c1_mlp --protocols standard heldout_nu

Pour chaque variante (c1_mlp / c2_cnn / c3_window) et chaque protocole
(standard / heldout_nu), le script :

1. entraîne le modèle multi-tâches (5 cibles : E, nu*D, D, TV, max|u_x|),
2. évalue sur le test (global, par nu, par temps, par quartile de E(0),
   agrégé par trajectoire) + baselines triviales,
3. sauvegarde métriques JSON + CSV long format + checkpoint,
4. calcule le verdict d'identifiabilité de nu*D à partir du protocole
   heldout_nu (r_holdout, rho, equiv) et écrit `experiment_c_summary.json`.

Ce script est additif : il ne modifie aucun point d'entrée existant et ne
change aucun comportement par défaut des autres scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import importlib.util
import json


# =============================================================================
# 1. Chargement flexible des modules du projet
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: List[str]):
    """
    Charge dynamiquement un module depuis une liste de chemins candidats.
    """
    try:
        return __import__(module_name, fromlist=['*'])
    except ModuleNotFoundError:
        pass

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        f"Impossible de charger le module {module_name!r}. "
        f"Candidats testés : {candidates}"
    )


def _load_config_module():
    return _load_module_from_candidates('config', ['config.py'])


def _load_train_module():
    return _load_module_from_candidates(
        'ml_train_experiment_c', ['ml_train_experiment_c.py'])


# =============================================================================
# 2. Utilitaires
# =============================================================================

def ensure_dir(path: str | Path) -> Path:
    """Crée un dossier (et ses parents) s'il n'existe pas."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Sauvegarde un dictionnaire en JSON (indenté, UTF-8)."""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def print_section(title: str) -> None:
    """Affiche un titre de section lisible."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# 3. Verdict d'identifiabilité de nu*D
# =============================================================================

def evaluate_identifiability(
    standard_results: Dict[str, Any],
    heldout_results: Dict[str, Any],
    thresholds: Dict[str, float],
    heldout_nu_values: Sequence[float],
    target: str = "nu_dissipation",
) -> Dict[str, Any]:
    """
    Calcule le verdict d'identifiabilité de la cible `target` (nu*D) à
    partir des résultats des protocoles "standard" et "heldout_nu".

    Critères (sous-ensemble "nu=0.01" du protocole heldout_nu) :
    - r_holdout : Spearman(pred, cible) sur la strate nu=0.01 held-out ;
    - rel_rmse_holdout : rel_rmse_mean sur nu=0.01 held-out ;
    - rho = rel_rmse_holdout / max(rel_rmse_mean in-distribution sur les
      strates `heldout_nu_values`, eps) ;
    - equiv : R² global entre les prédictions des têtes D et nu*D
      (1.0 = têtes indiscernables) ;
    - beats_nu_oracle : le modèle bat la baseline "moyenne par nu" sur
      nu=0.01 held-out.

    Verdict :
    - IDENTIFIABLE     si r_holdout >= r_holdout_min, rho <= rho_max,
                       equiv < equiv_max et beats_nu_oracle ;
    - NOT IDENTIFIABLE si r_holdout < r_holdout_fail ou
                       rel_rmse_holdout > rel_rmse_fail ;
    - PARTIAL          sinon.
    """
    subset_heldout = "nu=0.01"

    r_holdout = float(
        heldout_results["metrics"][subset_heldout][target]["spearman"])
    rel_rmse_holdout = float(
        heldout_results["metrics"][subset_heldout][target]["rel_rmse_mean"])

    rel_rmse_indist = max(
        float(standard_results["metrics"][f"nu={float(nu):g}"][target]["rel_rmse_mean"])
        for nu in heldout_nu_values
    )
    rho = rel_rmse_holdout / max(rel_rmse_indist, 1.0e-30)

    equiv = float(standard_results["separability"]["r2_pred_D_vs_pred_nuD_global"])

    baseline_rel_rmse = float(
        heldout_results["baselines"]["nu_oracle"][subset_heldout][target]["rel_rmse_mean"])
    beats_nu_oracle = bool(rel_rmse_holdout < baseline_rel_rmse)

    if (r_holdout >= thresholds["r_holdout_min"]
            and rho <= thresholds["rho_max"]
            and equiv < thresholds["equiv_max"]
            and beats_nu_oracle):
        verdict = "IDENTIFIABLE"
    elif (r_holdout < thresholds["r_holdout_fail"]
          or rel_rmse_holdout > thresholds["rel_rmse_fail"]):
        verdict = "NOT IDENTIFIABLE"
    else:
        verdict = "PARTIAL"

    return {
        "target": target,
        "r_holdout": r_holdout,
        "rel_rmse_holdout_nu001": rel_rmse_holdout,
        "rel_rmse_indist_max": rel_rmse_indist,
        "rho": rho,
        "equiv_r2_pred_D_vs_pred_nuD": equiv,
        "baseline_nu_oracle_rel_rmse_nu001": baseline_rel_rmse,
        "beats_nu_oracle": beats_nu_oracle,
        "verdict": verdict,
    }


def evaluate_sanity(
    standard_results: Dict[str, Any],
    sanity_r2_energy: float,
    sanity_r2_other_targets: float,
) -> Dict[str, Any]:
    """
    Sanity du pipeline : R² global (test) de chaque cible en protocole
    standard. E doit dépasser `sanity_r2_energy`, les autres cibles
    `sanity_r2_other_targets`.
    """
    global_metrics = standard_results["metrics"]["global"]
    r2_energy = float(global_metrics["energy"]["r2"])
    r2_others = {
        name: float(m["r2"])
        for name, m in global_metrics.items() if name != "energy"
    }
    return {
        "r2_energy": r2_energy,
        "r2_other_targets": r2_others,
        "r2_others_min": min(r2_others.values()) if r2_others else 0.0,
        "sanity_ok": bool(
            r2_energy >= sanity_r2_energy
            and (min(r2_others.values()) if r2_others else 0.0) >= sanity_r2_other_targets
        ),
    }


# =============================================================================
# 4. Pipeline complet
# =============================================================================

def run_experiment_c(
    config: Any,
    variants: Optional[Sequence[str]] = None,
    protocols: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Exécute l'expérience C pour les variantes x protocoles demandées.

    Paramètres
    ----------
    config : ProjectConfig
        Configuration du projet (utilise `config.experiment_c`).
    variants : séquence de str ou None
        Variantes à exécuter (None = `config.experiment_c.variants`).
    protocols : séquence de str ou None
        Protocoles à exécuter (None = `config.experiment_c.protocols`).

    Retour
    ------
    Dict[str, Any]
        Résumé complet, aussi sauvegardé dans
        `outputs/logs/experiment_c_summary.json`.
    """
    mt = _load_train_module()

    ec = config.experiment_c
    variants = list(variants) if variants is not None else list(ec.variants)
    protocols = list(protocols) if protocols is not None else list(ec.protocols)

    data_dir = Path(config.paths.data_dir)
    train_path = data_dir / ec.train_filename
    val_path = data_dir / ec.val_filename
    test_path = data_dir / ec.test_filename

    models_root = Path(config.paths.models_dir) / ec.models_subdir
    logs_dir = ensure_dir(config.paths.logs_dir)

    if verbose:
        print_section("Expérience C : baseline supervisée de séparabilité (nu_sweep_v1)")
        print(f"- variantes  : {variants}")
        print(f"- protocoles : {protocols}")
        print(f"- données    : {data_dir}")
        print(f"- sorties    : {models_root}")

    runs: Dict[str, Dict[str, Any]] = {v: {} for v in variants}

    for variant in variants:
        for protocol in protocols:
            output_dir = ensure_dir(models_root / f"{variant}_{protocol}")
            if verbose:
                print_section(f"Variante {variant} | protocole {protocol}")

            results = mt.train_and_evaluate_experiment_c(
                variant=variant,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path,
                output_dir=output_dir,
                protocol=protocol,
                heldout_nu_values=ec.heldout_nu_values,
                window_size=ec.window_size,
                batch_size=ec.batch_size,
                learning_rate=ec.learning_rate,
                num_epochs=ec.num_epochs,
                patience=ec.patience,
                weight_decay=ec.weight_decay,
                latent_dim=ec.latent_dim,
                encoder_hidden_dims=ec.encoder_hidden_dims,
                conv_channels=ec.conv_channels,
                kernel_size=ec.kernel_size,
                conv_stride=ec.conv_stride,
                trunk_dims=ec.trunk_dims,
                head_hidden_dims=ec.head_hidden_dims,
                activation=ec.activation,
                dropout=ec.dropout,
                eval_times=ec.eval_times,
                seed=int(config.reproducibility.seed),
                device=config.ml.device,
                num_workers=0,
                print_every=10,
                verbose=verbose,
            )

            runs[variant][protocol] = {
                "output_dir": str(output_dir),
                "checkpoint_path": results["checkpoint_path"],
                "metrics_json": str(output_dir / f"metrics_{variant}_{protocol}.json"),
                "metrics_csv": str(output_dir / f"metrics_long_{variant}_{protocol}.csv"),
                "best_epoch": results["best_epoch"],
                "best_val_loss": results["best_val_loss"],
                "stopped_early": results["stopped_early"],
                "results": results,
            }

    # sanity + verdict d'identifiabilité (nécessite les deux protocoles)
    thresholds = {
        "r_holdout_min": ec.r_holdout_min,
        "rho_max": ec.rho_max,
        "equiv_max": ec.equiv_max,
        "r_holdout_fail": ec.r_holdout_fail,
        "rel_rmse_fail": ec.rel_rmse_fail,
    }

    sanity: Dict[str, Any] = {}
    identifiability: Dict[str, Any] = {}

    for variant in variants:
        if "standard" in runs[variant]:
            sanity[variant] = evaluate_sanity(
                runs[variant]["standard"]["results"],
                ec.sanity_r2_energy,
                ec.sanity_r2_other_targets,
            )
        if "standard" in runs[variant] and "heldout_nu" in runs[variant]:
            identifiability[variant] = evaluate_identifiability(
                runs[variant]["standard"]["results"],
                runs[variant]["heldout_nu"]["results"],
                thresholds,
                ec.heldout_nu_values,
            )

    summary: Dict[str, Any] = {
        "experiment": "experiment_c",
        "dataset_version": "nu_sweep_v1",
        "seed": int(config.reproducibility.seed),
        "variants": variants,
        "protocols": protocols,
        "runs": {
            v: {p: {k: val for k, val in info.items() if k != "results"}
                for p, info in runs[v].items()}
            for v in variants
        },
        "sanity": sanity,
        "identifiability": identifiability,
        "thresholds": thresholds,
    }

    summary_path = logs_dir / ec.summary_filename
    save_json(summary_path, summary)
    save_json(logs_dir / ec.config_snapshot_filename, config.to_dict())

    if verbose:
        print_section("Résumé de l'expérience C")
        for variant in variants:
            if variant in sanity:
                s = sanity[variant]
                print(f"- {variant:10s} sanity_ok={s['sanity_ok']} "
                      f"(R2 E={s['r2_energy']:.4f}, min autres={s['r2_others_min']:.4f})")
            if variant in identifiability:
                d = identifiability[variant]
                print(f"- {variant:10s} verdict nu*D = {d['verdict']:16s} "
                      f"(r_holdout={d['r_holdout']:.3f}, rho={d['rho']:.2f}, "
                      f"equiv={d['equiv_r2_pred_D_vs_pred_nuD']:.4f}, "
                      f"beats_nu_oracle={d['beats_nu_oracle']})")
        print(f"Résumé sauvegardé : {summary_path}")

    return summary


# =============================================================================
# 5. Arguments de ligne de commande
# =============================================================================

def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse les arguments : --variants / --protocols (listes optionnelles).

    Sans argument, exécute toutes les variantes x tous les protocoles de
    la configuration. Ex. première run de production :
        python main_experiment_c.py --variants c1_mlp
    """
    parser = argparse.ArgumentParser(
        description="Expérience C : baseline supervisée de séparabilité sur nu_sweep_v1."
    )
    parser.add_argument(
        "--variants", nargs="*", default=None,
        choices=["c1_mlp", "c2_cnn", "c3_window"],
        help="Variantes de modèles à exécuter (défaut: toutes, depuis la config).",
    )
    parser.add_argument(
        "--protocols", nargs="*", default=None,
        choices=["standard", "heldout_nu"],
        help="Protocoles à exécuter (défaut: tous, depuis la config).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Point d'entrée : charge la config et lance l'expérience C."""
    args = _parse_args(argv)
    config_module = _load_config_module()
    config = config_module.get_default_config(create_dirs=True)
    return run_experiment_c(
        config,
        variants=args.variants,
        protocols=args.protocols,
        verbose=True,
    )


if __name__ == "__main__":
    main()
