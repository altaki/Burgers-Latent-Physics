"""
main_identify_latent_variable.py
================================

Script principal pour identifier, parmi une bibliothèque d'observables candidates,
quelle quantité est la plus proche de la variable latente scalaire apprise par le modèle.

Objectif
--------
Avant de passer à une deuxième étape de modélisation, ce script permet de répondre à :

> Si la variable latente z n'est pas l'énergie, pourrait-elle ressembler à une autre
> observable intéressante (masse, norme L1/L∞, gradient, localisation de choc,
> contenu spectral, etc.) ?

Le script :
1. charge le dataset de test,
2. charge les checkpoints latents disponibles,
3. reconstruit la variable scalaire z prédite par les modèles,
4. calcule une bibliothèque d'observables candidates à partir de u(x,t),
5. compare z à chaque observable par corrélation et calibration affine,
6. sauvegarde un classement complet (CSV + JSON) et un résumé court.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import importlib.util
import json
import sys
import pandas as pd
import numpy as np


def _build_fallback_config() -> Dict[str, Any]:
    return {
        'paths': {
            'data_dir': 'outputs/data',
            'logs_dir': 'outputs/logs',
            'models_dir': 'outputs/models',
        },
        'dataset': {
            'test_filename': 'test_data.npz',
        },
        'ml': {
            'device': 'cpu',
        },
        'pde': {
            'nu': 0.05,
        },
    }


def _load_module_from_candidates(module_name: str, candidates: list[str]):
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
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
    raise ModuleNotFoundError(f"Impossible de charger le module {module_name!r}. Candidats testés : {candidates}")


def load_config_object():
    try:
        from config import get_default_config  # type: ignore
        return get_default_config(create_dirs=True)
    except ModuleNotFoundError:
        pass

    config_path = Path('config.py')
    if config_path.exists():
        spec = importlib.util.spec_from_file_location('config', str(config_path))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules['config'] = module
        spec.loader.exec_module(module)
        if hasattr(module, 'get_default_config'):
            return module.get_default_config(create_dirs=True)

    cfg = _build_fallback_config()
    for folder in cfg['paths'].values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    return cfg


def _get_value(config: Any, dotted_key: str, default: Any = None) -> Any:
    parts = dotted_key.split('.')
    current = config
    for part in parts:
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def print_section(title: str) -> None:
    line = '=' * len(title)
    print(f"\n{title}\n{line}")


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def compute_per_trajectory_rankings(
    z: np.ndarray,
    trajectory_index: np.ndarray,
    candidate_values: Dict[str, np.ndarray],
    candidate_module,
    min_std: float = 1.0e-10,
):
    """
    Calcule un classement des candidats pour chaque trajectoire séparément,
    en excluant les candidats quasi constants sur la trajectoire, puis agrège
    les résultats pour mesurer la cohérence du meilleur candidat.

    Paramètres
    ----------
    z : np.ndarray
        Variable latente scalaire, de forme (n,) ou (n,1).
    trajectory_index : np.ndarray
        Indice de trajectoire pour chaque échantillon.
    candidate_values : Dict[str, np.ndarray]
        Dictionnaire {nom_candidat: valeurs_aplatis}.
    candidate_module : module
        Module contenant rank_candidates_against_latent(...)
    min_std : float
        Seuil minimal d'écart-type pour considérer un candidat comme informatif
        sur une trajectoire donnée.

    Retour
    ------
    tuple :
        - all_rankings_df : classement complet pour chaque trajectoire
        - best_per_traj_df : meilleur candidat par trajectoire
        - aggregated_df : agrégation par candidat
        - consistency_summary : résumé court
    """
    z = np.asarray(z)
    trajectory_index = np.asarray(trajectory_index, dtype=int)

    unique_traj = np.unique(trajectory_index)

    all_rankings = []
    best_rows = []

    skipped_constant_counts = {}

    for traj in unique_traj:
        mask = trajectory_index == traj
        z_traj = z[mask]

        # Si z lui-même est quasi constant sur la trajectoire, on ignore
        if np.std(z_traj) <= min_std:
            continue

        filtered_candidate_values = {}
        skipped_here = []

        for name, values in candidate_values.items():
            values_traj = np.asarray(values)[mask]

            # Exclure les candidats quasi constants
            if np.std(values_traj) <= min_std:
                skipped_here.append(name)
                skipped_constant_counts[name] = skipped_constant_counts.get(name, 0) + 1
                continue

            filtered_candidate_values[name] = values_traj

        # Si tous les candidats ont été exclus, on passe à la trajectoire suivante
        if len(filtered_candidate_values) == 0:
            continue

        df_traj = candidate_module.rank_candidates_against_latent(
            z_traj,
            filtered_candidate_values,
        )

        df_traj["trajectory_index"] = int(traj)
        df_traj["n_candidates_kept"] = int(len(filtered_candidate_values))
        df_traj["n_candidates_skipped"] = int(len(skipped_here))

        all_rankings.append(df_traj)

        best_row = df_traj.iloc[0].to_dict()
        best_rows.append(best_row)

    # Si aucune trajectoire n'est exploitable
    if len(all_rankings) == 0:
        import pandas as pd

        empty_df = pd.DataFrame()
        consistency_summary = {
            "n_trajectories_total": int(len(unique_traj)),
            "n_trajectories_used": 0,
            "most_frequent_best_candidate": None,
            "wins_of_most_frequent_candidate": 0,
            "fraction_trajectories_same_best_candidate": 0.0,
            "skipped_constant_counts": skipped_constant_counts,
            "min_std_threshold": float(min_std),
        }
        return empty_df, empty_df, empty_df, consistency_summary

    import pandas as pd

    all_rankings_df = pd.concat(all_rankings, ignore_index=True)
    best_per_traj_df = pd.DataFrame(best_rows)

    aggregated_df = (
        best_per_traj_df
        .groupby("candidate", as_index=False)
        .agg(
            wins=("candidate", "size"),
            mean_abs_spearman=("abs_spearman", "mean"),
            std_abs_spearman=("abs_spearman", "std"),
            mean_abs_pearson=("abs_pearson", "mean"),
            std_abs_pearson=("abs_pearson", "std"),
            mean_affine_rmse=("affine_rmse", "mean"),
            std_affine_rmse=("affine_rmse", "std"),
            mean_affine_r2=("affine_r2", "mean"),
            std_affine_r2=("affine_r2", "std"),
        )
        .sort_values(
            by=["wins", "mean_abs_spearman", "mean_affine_rmse"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    top_candidate = str(aggregated_df.iloc[0]["candidate"])
    top_wins = int(aggregated_df.iloc[0]["wins"])
    n_used = int(best_per_traj_df["trajectory_index"].nunique())
    n_total = int(len(unique_traj))

    consistency_summary = {
        "n_trajectories_total": n_total,
        "n_trajectories_used": n_used,
        "most_frequent_best_candidate": top_candidate,
        "wins_of_most_frequent_candidate": top_wins,
        "fraction_trajectories_same_best_candidate": float(top_wins / max(n_used, 1)),
        "skipped_constant_counts": skipped_constant_counts,
        "min_std_threshold": float(min_std),
    }

    return all_rankings_df, best_per_traj_df, aggregated_df, consistency_summary



def compute_per_dim_candidate_rankings(
    h: np.ndarray,
    trajectory_index: np.ndarray,
    time_index: np.ndarray,
    candidate_tensor: Dict[str, np.ndarray],
    candidate_module,
    static_threshold: float = 0.7,
    dynamic_threshold: float = 0.3,
    interpretable_abs_spearman: float = 0.5,
):
    """
    Pour chaque dimension latente h_j, classe TOUS les observables candidats
    (Pearson, Spearman, R² affine via `rank_candidates_against_latent`), puis
    produit un résumé par dimension : meilleur candidat, métriques, fraction de
    variance entre trajectoires, et classification (static / dynamic / mixed).

    Paramètres
    ----------
    h : np.ndarray
        Latents de forme (n_samples, latent_dim), ex. h_t extrait du test set.
    trajectory_index, time_index : np.ndarray
        Indices (trajectoire, temps) alignés avec les lignes de h.
    candidate_tensor : Dict[str, np.ndarray]
        Observables candidates de forme (n_traj, n_times) pour chaque nom.
    candidate_module : module
        Module `physics_candidates` (pour extract_candidate_values_from_indices
        et rank_candidates_against_latent).
    static_threshold : float
        Si la fraction de variance entre trajectoires >= ce seuil, la dimension
        est classée 'static_label' (label statique de trajectoire).
    dynamic_threshold : float
        Si la fraction entre trajectoires <= ce seuil, la dimension est classée
        'dynamic' (elle varie en temps à l'intérieur des trajectoires).
    interpretable_abs_spearman : float
        Seuil sur |Spearman| du meilleur candidat pour déclarer la dimension
        'interpretable' (seuil 0.5, cf critères du README).

    Retour
    ------
    tuple (per_dim_df, per_dim_summary)
        - per_dim_df : DataFrame concaténé, colonne 'latent_dim' en tête,
          toutes les lignes de classement pour chaque dimension,
        - per_dim_summary : liste de dicts (un par latent_dim) avec best_candidate,
          pearson, spearman, abs_spearman, affine_r2, affine_rmse,
          between_traj_var_fraction, mean_cross_sectional_abs_spearman_best,
          classification, interpretable.
    """
    h = np.asarray(h, dtype=float)
    if h.ndim != 2:
        raise ValueError(f"`h` doit être 2D (n_samples, latent_dim), obtenu {h.shape}")
    n_samples, latent_dim = h.shape
    trajectory_index = np.asarray(trajectory_index, dtype=int)
    time_index = np.asarray(time_index, dtype=int)
    if not (h.shape[0] == trajectory_index.shape[0] == time_index.shape[0]):
        raise ValueError("h, trajectory_index et time_index doivent avoir le même nombre de lignes.")

    candidate_values = candidate_module.extract_candidate_values_from_indices(
        candidate_tensor, trajectory_index, time_index
    )

    frames = []
    summary_rows = []

    for j in range(latent_dim):
        h_j = h[:, j]
        df_j = candidate_module.rank_candidates_against_latent(h_j, candidate_values)
        df_j.insert(0, 'latent_dim', j)
        frames.append(df_j)

        best = df_j.iloc[0].to_dict()

        # Fraction de variance entre trajectoires : Var(moyennes par trajectoire) / Var(h_j)
        # (forte valeur => la dimension est surtout un label statique de trajectoire).
        traj_means = pd.DataFrame({'traj': trajectory_index, 'h': h_j}).groupby('traj')['h'].mean()
        var_total = float(np.var(h_j))
        var_between = float(np.var(traj_means.to_numpy())) if len(traj_means) > 1 else 0.0
        between_frac = var_between / var_total if var_total > 1.0e-14 else 0.0

        if between_frac >= static_threshold:
            classification = 'static_label'
        elif between_frac <= dynamic_threshold:
            classification = 'dynamic'
        else:
            classification = 'mixed'

        # Corrélation transversale (à temps fixé) avec le meilleur candidat :
        # forte et stable à chaque bin de temps => alignement statique ; sinon dynamique.
        best_name = str(best['candidate'])
        best_vals = np.asarray(candidate_values[best_name], dtype=float)
        cross_corr = []
        for t in np.unique(time_index):
            mask = time_index == t
            if np.sum(mask) >= 3 and np.std(h_j[mask]) > 1.0e-14 and np.std(best_vals[mask]) > 1.0e-14:
                cross_corr.append(abs(candidate_module.spearman_correlation(h_j[mask], best_vals[mask])))
        mean_cross = float(np.mean(cross_corr)) if len(cross_corr) > 0 else 0.0

        summary_rows.append({
            'latent_dim': int(j),
            'best_candidate': best_name,
            'pearson': float(best['pearson']),
            'spearman': float(best['spearman']),
            'abs_spearman': float(best['abs_spearman']),
            'affine_r2': float(best['affine_r2']),
            'affine_rmse': float(best['affine_rmse']),
            'var_total': var_total,
            'between_traj_var_fraction': float(between_frac),
            'mean_cross_sectional_abs_spearman_best': mean_cross,
            'classification': classification,
            'interpretable': bool(float(best['abs_spearman']) >= interpretable_abs_spearman),
        })

    per_dim_df = pd.concat(frames, ignore_index=True)
    return per_dim_df, summary_rows



def main() -> Dict[str, Any]:
    config = load_config_object()

    eval_module = _load_module_from_candidates('ml_evaluate', ['ml_evaluate.py', 'ml/evaluate.py'])
    candidate_module = _load_module_from_candidates('physics_candidates', ['physics_candidates.py'])

    data_dir = Path(_get_value(config, 'paths.data_dir', 'outputs/data'))
    logs_dir = Path(_get_value(config, 'paths.logs_dir', 'outputs/logs'))
    models_dir = Path(_get_value(config, 'paths.models_dir', 'outputs/models'))
    dataset_path = data_dir / str(_get_value(config, 'dataset.test_filename', 'test_data.npz'))
    device = _get_value(config, 'ml.device', None)
    nu = float(_get_value(config, 'pde.nu', 0.05))

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")

    print_section('Chargement du dataset de test')
    load_npz_dataset = _load_module_from_candidates('utils_plotting', ['utils_plotting.py', 'utils/plotting.py']).load_npz_dataset
    dataset = load_npz_dataset(dataset_path)
    states = np.asarray(dataset['states'], dtype=float)
    x = np.asarray(dataset['x'], dtype=float)
    dx = float(np.asarray(dataset['dx']).item())
    print(f'- Dataset             : {dataset_path}')
    print(f'- states shape        : {states.shape}')
    print(f'- nu                  : {nu}')

    print_section('Calcul des observables candidates sur tout le dataset')
    candidate_tensor = candidate_module.compute_candidate_tensor_from_dataset(states=states, x=x, dx=dx, nu=nu)
    print(f'- Nombre de candidats : {len(candidate_tensor)}')

    summary: Dict[str, Any] = {
        'dataset_path': str(dataset_path),
        'models': {},
    }

    # ------------------------------------------------------------------
    # 1) Modèle de dynamique latente
    # ------------------------------------------------------------------
    latent_dyn_ckpt = models_dir / 'latent_dynamics' / 'best_latent_dynamics_model.pt'
    if latent_dyn_ckpt.exists():
        print_section('Identification pour le modèle latent_dynamics')
        results = eval_module.evaluate_saved_latent_dynamics_model(
            checkpoint_path=latent_dyn_ckpt,
            dataset_path=dataset_path,
            batch_size=128,
            stride=None,
            device=device,
            output_json_path=None,
        )
        z = results['extracted']['z_t']
        traj_idx = results['extracted']['trajectory_index']
        time_idx = results['extracted']['time_index_t']
        candidate_values = candidate_module.extract_candidate_values_from_indices(candidate_tensor, traj_idx, time_idx)
        # Classement global sur tout le dataset
        df = candidate_module.rank_candidates_against_latent(z, candidate_values)
        saved = candidate_module.save_rankings(df, logs_dir, prefix='latent_dynamics')
        top5 = df.head(5).to_dict(orient='records')

        # Classement trajectoire par trajectoire
        all_rankings_df, best_per_traj_df, aggregated_df, consistency_summary = compute_per_trajectory_rankings(
            z=z,
            trajectory_index=traj_idx,
            candidate_values=candidate_values,
            candidate_module=candidate_module,
        )

        per_traj_csv = logs_dir / "latent_dynamics_candidate_rankings_per_trajectory.csv"
        best_per_traj_csv = logs_dir / "latent_dynamics_best_candidate_per_trajectory.csv"
        aggregated_csv = logs_dir / "latent_dynamics_candidate_rankings_aggregated.csv"

        all_rankings_df.to_csv(per_traj_csv, index=False)
        best_per_traj_df.to_csv(best_per_traj_csv, index=False)
        aggregated_df.to_csv(aggregated_csv, index=False)

        summary['models']['latent_dynamics'] = {
            'checkpoint_path': str(latent_dyn_ckpt),
            'top5_candidates': top5,
            'saved_rankings': saved,
            'per_trajectory_rankings': {
                'all_rankings_csv': str(per_traj_csv),
                'best_per_trajectory_csv': str(best_per_traj_csv),
                'aggregated_csv': str(aggregated_csv),
            },
            'trajectory_consistency': consistency_summary,
            'top5_aggregated_candidates': aggregated_df.head(5).to_dict(orient='records'),
        }

        print("Top 5 global candidates:")
        for k, row in enumerate(top5, 1):
            print(
                f"{k}. {row['candidate']} | "
                f"abs_spearman={row['abs_spearman']:.4f} | "
                f"affine_rmse={row['affine_rmse']:.4f} | "
                f"pearson={row['pearson']:.4f}"
            )

        print("\\nTop 5 aggregated candidates across trajectories:")
        for k, row in enumerate(aggregated_df.head(5).to_dict(orient='records'), 1):
            print(
                f"{k}. {row['candidate']} | "
                f"wins={int(row['wins'])} | "
                f"mean_abs_spearman={row['mean_abs_spearman']:.4f} | "
                f"mean_affine_rmse={row['mean_affine_rmse']:.4f}"
            )

        print("\\nTrajectory consistency summary:")
        print(
            f"- most frequent best candidate : {consistency_summary['most_frequent_best_candidate']}"
        )
        print(f"- fraction of trajectories with same best candidate : "f"{consistency_summary['fraction_trajectories_same_best_candidate']:.4f}")

    else:
        print_section('Aucun checkpoint latent_dynamics trouvé')

    # ------------------------------------------------------------------
    # 2) Modèle autoencodeur latent
    # ------------------------------------------------------------------
    latent_ae_ckpt = models_dir / 'latent_autoencoder' / 'best_latent_autoencoder.pt'
    if latent_ae_ckpt.exists():
        print_section('Identification pour le modèle latent_autoencoder')
        results = eval_module.evaluate_saved_latent_autoencoder(
            checkpoint_path=latent_ae_ckpt,
            dataset_path=dataset_path,
            batch_size=128,
            device=device,
            output_json_path=None,
        )
        z = results['extracted']['scalar']
        traj_idx = results['extracted']['trajectory_index']
        time_idx = results['extracted']['time_index']
        candidate_values = candidate_module.extract_candidate_values_from_indices(candidate_tensor, traj_idx, time_idx)
        df = candidate_module.rank_candidates_against_latent(z, candidate_values)
        saved = candidate_module.save_rankings(df, logs_dir, prefix='latent_autoencoder')
        top5 = df.head(5).to_dict(orient='records')
        summary['models']['latent_autoencoder'] = {
            'checkpoint_path': str(latent_ae_ckpt),
            'top5_candidates': top5,
            'saved_rankings': saved,
        }
        for k, row in enumerate(top5, 1):
            print(f"{k}. {row['candidate']} | abs_spearman={row['abs_spearman']:.4f} | affine_rmse={row['affine_rmse']:.4f} | pearson={row['pearson']:.4f}")
    else:
        print_section('Aucun checkpoint latent_autoencoder trouvé')

    # ------------------------------------------------------------------
    # 3) Autoencodeur de dynamique latente (Architecture A) : analyse PAR DIMENSION
    # ------------------------------------------------------------------
    latent_dyn_ae_ckpt = models_dir / 'latent_dynamics_autoencoder' / 'best_latent_dynamics_autoencoder.pt'
    if latent_dyn_ae_ckpt.exists():
        print_section('Identification par dimension pour latent_dynamics_autoencoder (Architecture A)')
        results = eval_module.evaluate_saved_latent_dynamics_autoencoder(
            checkpoint_path=latent_dyn_ae_ckpt,
            dataset_path=dataset_path,
            batch_size=128,
            stride=None,
            device=device,
            output_json_path=None,
        )
        h = np.asarray(results['extracted']['h_t'], dtype=float)
        traj_idx = results['extracted']['trajectory_index']
        time_idx = results['extracted']['time_index_t']
        print(f'- h_t shape           : {h.shape}')

        # Candidats supplémentaires liés à la condition initiale / nu si disponibles
        if 'ic_type_ids' in dataset:
            candidate_tensor['ic_type_id'] = (
                np.asarray(dataset['ic_type_ids'], dtype=float)[:, None]
                * np.ones((1, states.shape[1]))
            )
        if 'nu_values' in dataset:
            candidate_tensor['nu'] = (
                np.asarray(dataset['nu_values'], dtype=float)[:, None]
                * np.ones((1, states.shape[1]))
            )

        per_dim_df, per_dim_summary = compute_per_dim_candidate_rankings(
            h=h,
            trajectory_index=traj_idx,
            time_index=time_idx,
            candidate_tensor=candidate_tensor,
            candidate_module=candidate_module,
        )

        per_dim_csv = logs_dir / 'latent_dynamics_autoencoder_candidate_rankings_per_dim.csv'
        per_dim_df.to_csv(per_dim_csv, index=False)
        per_dim_summary_json = logs_dir / 'latent_dynamics_autoencoder_per_dim_summary.json'
        save_json(per_dim_summary_json, {
            'checkpoint_path': str(latent_dyn_ae_ckpt),
            'latent_dim': int(h.shape[1]),
            'latent_dims': per_dim_summary,
        })

        summary['models']['latent_dynamics_autoencoder'] = {
            'checkpoint_path': str(latent_dyn_ae_ckpt),
            'latent_dim': int(h.shape[1]),
            'per_dim_rankings_csv': str(per_dim_csv),
            'per_dim_summary_json': str(per_dim_summary_json),
            'per_dim_summary': per_dim_summary,
        }

        print('Résumé par dimension latente (h_j -> meilleur candidat) :')
        for row in per_dim_summary:
            print(
                f"h{row['latent_dim']} -> {row['best_candidate']} | "
                f"abs_spearman={row['abs_spearman']:.4f} | "
                f"pearson={row['pearson']:.4f} | "
                f"affine_r2={row['affine_r2']:.4f} | "
                f"between_frac={row['between_traj_var_fraction']:.3f} | "
                f"cross_t={row['mean_cross_sectional_abs_spearman_best']:.3f} | "
                f"{row['classification']} | interpretable={row['interpretable']}"
            )
    else:
        print_section('Aucun checkpoint latent_dynamics_autoencoder trouvé')

    summary_path = logs_dir / 'main_identify_latent_variable_summary.json'
    save_json(summary_path, summary)

    print_section('Terminé')
    print(f'- Résumé sauvegardé : {summary_path}')
    print('- Les classements détaillés CSV/JSON sont dans outputs/logs/')
    return summary


if __name__ == '__main__':
    main()
