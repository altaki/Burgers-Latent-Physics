"""
main_generate_data.py
=====================

Script principal pour générer les datasets train / val / test du projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce script sert de point d'entrée simple pour :

1. charger la configuration globale du projet,
2. initialiser la reproductibilité,
3. générer les splits train / validation / test,
4. sauvegarder les fichiers de données au format .npz,
5. afficher un résumé et quelques vérifications de cohérence.

Philosophie
-----------
- script clair, lisible et abondamment commenté,
- peu d'hypothèses implicites,
- compatible avec le prototype déjà construit,
- robuste même si certains modules ne sont pas encore importables
  comme un package Python complet.

Fonctionnement pratique
-----------------------
Le script essaie d'abord d'utiliser `config.py` si ce fichier est disponible.
Si ce n'est pas le cas, il utilise une configuration locale de secours, avec
les mêmes paramètres par défaut que ceux définis précédemment dans le projet.

Même logique pour le builder de dataset :
- priorité à `pde_dataset_builder.py` si disponible,
- fallback sur `pde/dataset_builder.py` si le package `pde` est utilisable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field, asdict
import importlib.util
import json
import random
import sys

import numpy as np


# =============================================================================
# 1. Fallback local pour la configuration
# =============================================================================

@dataclass
class _PathsConfig:
    output_root: Path = Path("outputs")
    data_dir: Path = field(default_factory=lambda: Path("outputs") / "data")
    models_dir: Path = field(default_factory=lambda: Path("outputs") / "models")
    figures_dir: Path = field(default_factory=lambda: Path("outputs") / "figures")
    logs_dir: Path = field(default_factory=lambda: Path("outputs") / "logs")

    def create_directories(self) -> None:
        for path in [self.output_root, self.data_dir, self.models_dir, self.figures_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class _PDEConfig:
    L: float = 2.0 * np.pi
    Nx: int = 128
    nu: float = 0.05
    T: float = 2.0
    dt: float = 1.0e-3
    save_every: int = 20
    boundary_condition: str = "periodic"
    spatial_scheme: str = "finite_difference"
    time_scheme: str = "rk4"


@dataclass
class _DatasetConfig:
    n_train: int = 500
    n_val: int = 100
    n_test: int = 100
    ic_types: list[str] = field(default_factory=lambda: ["fourier", "gaussian_bumps", "random_smooth"])
    max_fourier_mode: int = 5
    fourier_amplitude_scale: float = 1.0
    n_gaussian_bumps: int = 2
    spectral_decay: float = 2.0
    train_filename: str = "train_data.npz"
    val_filename: str = "val_data.npz"
    test_filename: str = "test_data.npz"


@dataclass
class _ReproConfig:
    seed: int = 42
    deterministic_torch: bool = False


@dataclass
class _ProjectConfig:
    project_name: str = "burgers_latent_physics"
    experiment_name: str = "baseline_v1"
    paths: _PathsConfig = field(default_factory=_PathsConfig)
    pde: _PDEConfig = field(default_factory=_PDEConfig)
    dataset: _DatasetConfig = field(default_factory=_DatasetConfig)
    reproducibility: _ReproConfig = field(default_factory=_ReproConfig)

    def create_directories(self) -> None:
        self.paths.create_directories()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        def convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(data)


# =============================================================================
# 2. Chargement robuste de la configuration
# =============================================================================

def _load_config_function():
    """
    Essaie de récupérer `get_default_config` depuis `config.py`.

    Stratégie :
    1. import Python classique si possible,
    2. chargement via chemin local `config.py` si le fichier existe,
    3. fallback sur une configuration locale intégrée.
    """
    # 1) Import classique
    try:
        from config import get_default_config  # type: ignore
        return get_default_config
    except ModuleNotFoundError:
        pass

    # 2) Chargement depuis un fichier local si présent.
    config_path = Path("config.py")
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("config", str(config_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if hasattr(module, "get_default_config"):
            return module.get_default_config

    # 3) Fallback local.
    def _fallback_get_default_config(create_dirs: bool = True) -> _ProjectConfig:
        cfg = _ProjectConfig()
        if create_dirs:
            cfg.create_directories()
        return cfg

    return _fallback_get_default_config


get_default_config = _load_config_function()


# =============================================================================
# 3. Chargement robuste du builder de dataset
# =============================================================================

def _load_dataset_builder_symbols():
    """
    Charge les fonctions principales du builder de dataset.

    Ordre de priorité :
    1. import de `pde_dataset_builder`
    2. chargement par chemin `pde_dataset_builder.py`
    3. import de `pde.dataset_builder`
    4. chargement par chemin `pde/dataset_builder.py`
    """
    symbol_names = [
        "build_and_save_split_from_config",
        "summarize_dataset",
        "basic_dataset_sanity_checks",
    ]

    # 1) Import classique du fichier racine autoporté
    try:
        import pde_dataset_builder as module  # type: ignore
        return tuple(getattr(module, name) for name in symbol_names)
    except ModuleNotFoundError:
        pass

    # 2) Chargement via chemin local
    root_builder = Path("pde_dataset_builder.py")
    if root_builder.exists():
        spec = importlib.util.spec_from_file_location("pde_dataset_builder", str(root_builder))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return tuple(getattr(module, name) for name in symbol_names)

    # 3) Import package
    try:
        from pde.dataset_builder import (  # type: ignore
            build_and_save_split_from_config,
            summarize_dataset,
            basic_dataset_sanity_checks,
        )
        return build_and_save_split_from_config, summarize_dataset, basic_dataset_sanity_checks
    except ModuleNotFoundError:
        pass

    # 4) Chargement via chemin package
    pkg_builder = Path("pde") / "dataset_builder.py"
    if pkg_builder.exists():
        spec = importlib.util.spec_from_file_location("pde.dataset_builder", str(pkg_builder))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return tuple(getattr(module, name) for name in symbol_names)

    raise ModuleNotFoundError(
        "Impossible de charger le builder de dataset. Vérifiez la présence de "
        "`pde_dataset_builder.py` ou `pde/dataset_builder.py`."
    )


(
    build_and_save_split_from_config,
    summarize_dataset,
    basic_dataset_sanity_checks,
) = _load_dataset_builder_symbols()


# =============================================================================
# 4. Reproductibilité
# =============================================================================

def set_global_seed(seed: int) -> None:
    """
    Fixe les graines aléatoires principales pour la reproductibilité.
    """
    np.random.seed(seed)
    random.seed(seed)

    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


# =============================================================================
# 5. Utilitaires de logging
# =============================================================================

def print_section(title: str) -> None:
    """Affiche un titre de section lisible dans le terminal."""
    line = "=" * len(title)
    print(f"\n{title}\n{line}")



def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Sauvegarde un dictionnaire dans un fichier JSON lisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# =============================================================================
# 6. Génération d'un split avec résumé
# =============================================================================

def generate_and_report_split(config: Any, split: str, seed_offset: int = 0) -> Dict[str, Any]:
    """
    Génère un split, le sauvegarde, puis affiche un résumé et des sanity checks.
    """
    base_seed = int(config.reproducibility.seed)
    rng = np.random.default_rng(base_seed + seed_offset)

    print_section(f"Génération du split : {split}")

    dataset = build_and_save_split_from_config(
        config=config,
        split=split,
        rng=rng,
        verbose=True,
    )

    summary = summarize_dataset(dataset)
    checks = basic_dataset_sanity_checks(dataset)

    print("\nRésumé :")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("\nSanity checks :")
    for key, value in checks.items():
        print(f"- {key}: {value}")

    report = {
        "split": split,
        "summary": summary,
        "sanity_checks": checks,
    }
    report_path = Path(config.paths.logs_dir) / f"data_generation_report_{split}.json"
    save_json(report_path, report)
    print(f"\nRapport sauvegardé dans : {report_path}")

    return dataset


# =============================================================================
# 7. Point d'entrée principal
# =============================================================================

def main() -> None:
    """Point d'entrée principal du script."""
    print_section("Chargement de la configuration")
    config = get_default_config(create_dirs=True)

    print("Configuration chargée avec succès.")
    print(f"- Projet              : {config.project_name}")
    print(f"- Expérience          : {config.experiment_name}")
    print(f"- Dossier de données  : {config.paths.data_dir}")
    print(f"- Dossier de logs     : {config.paths.logs_dir}")

    print_section("Initialisation de la reproductibilité")
    seed = int(config.reproducibility.seed)
    set_global_seed(seed)
    print(f"Seed globale fixée à : {seed}")

    print_section("Paramètres principaux de génération")
    print(f"- Domaine L                 : {config.pde.L}")
    print(f"- Nombre de points Nx       : {config.pde.Nx}")
    print(f"- Viscosité nu              : {config.pde.nu}")
    print(f"- Temps final T             : {config.pde.T}")
    print(f"- Pas de temps dt           : {config.pde.dt}")
    print(f"- Sauvegarde tous les       : {config.pde.save_every} pas")
    print(f"- Schéma temporel           : {config.pde.time_scheme}")
    print(f"- CI autorisées             : {list(config.dataset.ic_types)}")
    print(
        f"- n_train / n_val / n_test  : "
        f"{config.dataset.n_train} / {config.dataset.n_val} / {config.dataset.n_test}"
    )

    print_section("Sauvegarde de la configuration")
    config_json_path = Path(config.paths.logs_dir) / "config_snapshot.json"
    save_json(config_json_path, config.to_dict())
    print(f"Configuration sauvegardée dans : {config_json_path}")

    # Génération des trois splits
    generate_and_report_split(config, split="train", seed_offset=0)
    generate_and_report_split(config, split="val", seed_offset=1)
    generate_and_report_split(config, split="test", seed_offset=2)

    print_section("Terminé")
    print("La génération des datasets train / val / test est terminée.")
    print("Vous pouvez maintenant passer à l'étape suivante :")
    print("- visualisation des trajectoires,")
    print("- chargement des données dans PyTorch,")
    print("- entraînement des modèles supervisés ou latents.")


# =============================================================================
# 7bis. Protocole nu_sweep_v1 : main_nu_sweep() et dispatch par argument
# =============================================================================

def _load_nu_sweep_builder_symbols():
    """
    Charge les symboles nu_sweep_v1 du builder de dataset.

    Chargement PARESSEUX (appelé uniquement depuis `main_nu_sweep`) : le
    chemin legacy sans argument ne doit pas dépendre des symboles nu_sweep.
    Nécessite le fichier racine `pde_dataset_builder.py` à jour (les symboles
    nu_sweep_v1 n'existent pas dans l'ancien package `pde/`).
    """
    symbol_names = [
        "build_and_save_split_from_config_nu_sweep",
        "summarize_dataset_nu_sweep",
        "nu_sweep_dataset_sanity_checks",
        "evaluate_nu_sweep_gates",
        "run_nu_sweep_pilot",
    ]

    modules = []
    try:
        import pde_dataset_builder as module  # type: ignore
        modules.append(module)
    except ModuleNotFoundError:
        pass

    root_builder = Path("pde_dataset_builder.py")
    if root_builder.exists():
        spec = importlib.util.spec_from_file_location(
            "pde_dataset_builder_nu_sweep", str(root_builder)
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        modules.append(module)

    for module in modules:
        if all(hasattr(module, name) for name in symbol_names):
            return tuple(getattr(module, name) for name in symbol_names)

    raise ModuleNotFoundError(
        "Symboles nu_sweep_v1 introuvables : il faut le fichier racine "
        "`pde_dataset_builder.py` à jour (fonctions : "
        + ", ".join(symbol_names)
        + ")."
    )


def _parse_protocol_arg(argv: list[str]) -> str | None:
    """
    Interprète les arguments de ligne de commande.

    Retourne None sans argument (chemin legacy inchangé) et "nu_sweep" pour
    `--protocol nu_sweep`. Tout autre argument lève SystemExit avec un
    message d'usage (on ne retombe jamais silencieusement sur le legacy).
    """
    if not argv:
        return None
    if argv == ["--protocol", "nu_sweep"]:
        return "nu_sweep"
    raise SystemExit(
        "Usage : python main_generate_data.py [--protocol nu_sweep]\n"
        f"Arguments non reconnus : {argv!r}"
    )


def main_nu_sweep() -> None:
    """
    Point d'entrée du protocole nu_sweep_v1 (dataset à balayage de viscosité).

    Étapes :
    1. charger la configuration globale (bloc `config.nu_sweep`),
    2. fixer la graine globale,
    3. sauvegarder le snapshot de configuration (`config_snapshot_nu_sweep.json`),
    4. exécuter le pilote `run_nu_sweep_pilot` et sauvegarder son rapport
       (`nu_sweep_pilot_report.json`) ; ABANDON si tous les niveaux du
       ladder échouent (aucun split n'est alors généré),
    5. générer et sauvegarder train / val / test
       (`{train,val,test}_data_nu_sweep.npz` dans `outputs/data`),
    6. écrire les rapports par split
       (`data_generation_report_nu_sweep_{split}.json`) et le résumé global
       (`nu_sweep_generation_summary.json`) dans `outputs/logs`.

    Usage : python main_generate_data.py --protocol nu_sweep
    """
    (
        build_and_save_split_from_config_nu_sweep,
        summarize_dataset_nu_sweep,
        nu_sweep_dataset_sanity_checks,
        evaluate_nu_sweep_gates,
        run_nu_sweep_pilot,
    ) = _load_nu_sweep_builder_symbols()

    print_section("Protocole nu_sweep_v1 : chargement de la configuration")
    config = get_default_config(create_dirs=True)
    ns = config.nu_sweep

    print("Configuration chargée avec succès.")
    print(f"- Projet               : {config.project_name}")
    print(f"- Version du dataset   : {ns.dataset_version}")
    print(f"- Viscosités nu        : {list(ns.nu_values)}")
    print(f"- Nx / dt / save_every : {ns.Nx} / {ns.dt} / {ns.save_every}")
    print(f"- L / T / schéma       : {ns.L} / {ns.T} / {ns.time_scheme}")
    print(f"- Splits               : train={ns.n_train}, val={ns.n_val}, test={ns.n_test}")
    print(f"- Dossier de données   : {config.paths.data_dir}")
    print(f"- Dossier de logs      : {config.paths.logs_dir}")

    print_section("Protocole nu_sweep_v1 : reproductibilité")
    seed = int(config.reproducibility.seed)
    set_global_seed(seed)
    print(f"Seed globale fixée à : {seed}")

    print_section("Protocole nu_sweep_v1 : snapshot de configuration")
    config_json_path = Path(config.paths.logs_dir) / "config_snapshot_nu_sweep.json"
    save_json(config_json_path, config.to_dict())
    print(f"Configuration sauvegardée dans : {config_json_path}")

    print_section("Protocole nu_sweep_v1 : pilote (ladder + portes numériques)")
    pilot_report = run_nu_sweep_pilot(config, verbose=True)
    pilot_report_path = Path(config.paths.logs_dir) / "nu_sweep_pilot_report.json"
    save_json(pilot_report_path, pilot_report)
    print(f"Rapport du pilote sauvegardé dans : {pilot_report_path}")
    print(f"- overall_pass  : {pilot_report['overall_pass']}")
    print(f"- niveau retenu : {pilot_report['selected_level']}")
    if not pilot_report["overall_pass"]:
        print_section("Protocole nu_sweep_v1 : ABANDON")
        print("Toutes les portes ont échoué à tous les niveaux du ladder :")
        print("aucun split n'est généré (voir le rapport du pilote).")
        raise SystemExit(1)

    print_section("Protocole nu_sweep_v1 : génération des splits train / val / test")
    split_reports = {}
    for split, seed_offset in [("train", 0), ("val", 1), ("test", 2)]:
        rng = np.random.default_rng(seed + seed_offset)
        print_section(f"Génération du split nu_sweep : {split}")

        dataset = build_and_save_split_from_config_nu_sweep(
            config=config,
            split=split,
            rng=rng,
            verbose=True,
        )

        summary = summarize_dataset_nu_sweep(dataset)
        checks = nu_sweep_dataset_sanity_checks(
            dataset,
            nu_values=list(ns.nu_values),
            target_norm_range=list(ns.target_norm_range),
        )
        gates = evaluate_nu_sweep_gates(dataset)

        print("\nRésumé :")
        for key, value in summary.items():
            print(f"- {key}: {value}")
        print("\nSanity checks :")
        for key, value in checks.items():
            print(f"- {key}: {value}")
        print("\nPortes numériques :")
        print(f"- gates_passed : {gates['gates_passed']} ({gates['checks']})")

        report = {
            "split": split,
            "dataset_version": str(ns.dataset_version),
            "summary": summary,
            "sanity_checks": checks,
            "gates": gates,
        }
        report_path = Path(config.paths.logs_dir) / f"data_generation_report_nu_sweep_{split}.json"
        save_json(report_path, report)
        print(f"\nRapport sauvegardé dans : {report_path}")

        split_reports[split] = {
            "filename": str(getattr(ns, f"{split}_filename")),
            "summary": summary,
            "sanity_checks_all_true": bool(all(checks.values())),
            "gates_passed": bool(gates["gates_passed"]),
        }

    print_section("Protocole nu_sweep_v1 : résumé global")
    generation_summary = {
        "dataset_version": str(ns.dataset_version),
        "protocol": str(ns.protocol),
        "nu_values": list(ns.nu_values),
        "numerical": {
            "Nx": int(ns.Nx),
            "dt": float(ns.dt),
            "save_every": int(ns.save_every),
            "L": float(ns.L),
            "T": float(ns.T),
            "time_scheme": str(ns.time_scheme),
        },
        "splits": {"train": int(ns.n_train), "val": int(ns.n_val), "test": int(ns.n_test)},
        "pilot": {
            "overall_pass": bool(pilot_report["overall_pass"]),
            "selected_level": pilot_report["selected_level"],
            "n_levels_tested": len(pilot_report["levels"]),
        },
        "split_reports": split_reports,
    }
    summary_path = Path(config.paths.logs_dir) / "nu_sweep_generation_summary.json"
    save_json(summary_path, generation_summary)
    print(f"Résumé global sauvegardé dans : {summary_path}")

    print_section("Terminé")
    print("La génération du dataset nu_sweep_v1 est terminée :")
    print(f"- {ns.train_filename}")
    print(f"- {ns.val_filename}")
    print(f"- {ns.test_filename}")


if __name__ == "__main__":
    # Dispatch additif du protocole nu_sweep_v1. Sans argument, ce bloc ne
    # fait rien : le bloc `__main__` legacy de la section 8 ci-dessous exécute
    # alors main() exactement comme avant. Avec `--protocol nu_sweep`, on
    # exécute main_nu_sweep() puis on quitte (sys.exit) : le bloc legacy ne
    # s'exécute pas.
    _argv = sys.argv[1:]
    if _argv:
        _protocol = _parse_protocol_arg(_argv)
        if _protocol == "nu_sweep":
            main_nu_sweep()
            sys.exit(0)


# =============================================================================
# 8. Exécution directe
# =============================================================================

if __name__ == "__main__":
    main()
