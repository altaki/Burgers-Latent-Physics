"""
main_compare_z_and_dissipation.py
=================================

Script pour comparer la variable latente scalaire z apprise par le modèle
de dynamique latente avec une observable physique candidate de type dissipation :

    dissipation(u) = nu * ∫ |u_x|^2 dx

Le script :
1. charge le dataset de test,
2. charge le checkpoint latent_dynamics,
3. reconstruit z_t via ml_evaluate.py,
4. calcule la dissipation via physics_candidates.py,
5. compare plusieurs transformations :
   - z
   - -z
   - log(dissipation + eps)
   - sqrt(dissipation)
6. sauvegarde métriques + figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import importlib.util
import json
import sys

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Chargement robuste des modules du projet
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: list[str]):
    """
    Charge dynamiquement un module depuis plusieurs chemins possibles.
    """
    try:
        return __import__(module_name, fromlist=["*"])
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

    raise ModuleNotFoundError(
        f"Impossible de charger le module {module_name!r}. "
        f"Candidats testés : {candidates}"
    )


# Chargement des modules existants du projet
eval_module = _load_module_from_candidates("ml_evaluate", ["ml_evaluate.py", "ml/evaluate.py"])
candidate_module = _load_module_from_candidates("physics_candidates", ["physics_candidates.py"])
plot_module = _load_module_from_candidates("utils_plotting", ["utils_plotting.py", "utils/plotting.py"])


# =============================================================================
# 2. Chargement de la configuration
# =============================================================================

def _build_fallback_config() -> Dict[str, Any]:
    """
    Configuration minimale si config.py n'est pas importable.
    """
    return {
        "paths": {
            "data_dir": "outputs/data",
            "models_dir": "outputs/models",
            "logs_dir": "outputs/logs",
            "figures_dir": "outputs/figures",
        },
        "dataset": {
            "test_filename": "test_data.npz",
        },
        "ml": {
            "device": "cpu",
        },
        "pde": {
            "nu": 0.05,
        },
    }


def load_config_object():
    """
    Charge config.py si disponible, sinon fallback.
    """
    try:
        from config import get_default_config  # type: ignore
        return get_default_config(create_dirs=True)
    except ModuleNotFoundError:
        pass

    cfg = _build_fallback_config()
    for folder in cfg["paths"].values():
        Path(folder).mkdir(parents=True, exist_ok=True)
    return cfg


def _get_value(config: Any, dotted_key: str, default: Any = None) -> Any:
    """
    Accède à une valeur dans un objet config ou un dictionnaire.
    """
    parts = dotted_key.split(".")
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


# =============================================================================
# 3. Utilitaires
# =============================================================================

def print_section(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{title}\n{line}")


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def flatten_scalar(a: np.ndarray) -> np.ndarray:
    """
    Aplati un tableau (n,) ou (n,1) en (n,).
    """
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    raise ValueError(f"Tableau scalaire attendu de forme (n,) ou (n,1), obtenu {a.shape}")


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    MSE / RMSE / MAE / Pearson / Spearman.
    """
    pred = flatten_scalar(np.asarray(pred, dtype=float))
    target = flatten_scalar(np.asarray(target, dtype=float))

    if pred.shape != target.shape:
        raise ValueError(f"Formes incompatibles : {pred.shape} vs {target.shape}")

    err = pred - target
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))

    # Pearson
    sx = np.std(pred)
    sy = np.std(target)
    if sx < 1e-14 or sy < 1e-14:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(pred, target)[0, 1])

    # Spearman simple
    def rankdata_average(a: np.ndarray) -> np.ndarray:
        sorter = np.argsort(a, kind="mergesort")
        inv = np.empty_like(sorter)
        inv[sorter] = np.arange(len(a))
        a_sorted = a[sorter]
        ranks_sorted = np.zeros(len(a), dtype=float)

        i = 0
        while i < len(a):
            j = i + 1
            while j < len(a) and a_sorted[j] == a_sorted[i]:
                j += 1
            avg_rank = 0.5 * ((i + 1) + j)
            ranks_sorted[i:j] = avg_rank
            i = j

        return ranks_sorted[inv]

    rx = rankdata_average(pred)
    ry = rankdata_average(target)

    sxr = np.std(rx)
    syr = np.std(ry)
    if sxr < 1e-14 or syr < 1e-14:
        spearman = 0.0
    else:
        spearman = float(np.corrcoef(rx, ry)[0, 1])

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
    }


def affine_fit(source: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """
    Ajuste target ≈ a * source + b au sens des moindres carrés.
    """
    source = flatten_scalar(np.asarray(source, dtype=float))
    target = flatten_scalar(np.asarray(target, dtype=float))

    A = np.column_stack([source, np.ones_like(source)])
    coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])

    fitted = a * source + b
    metrics = regression_metrics(fitted, target)

    return {
        "a": a,
        "b": b,
        "fitted": fitted,
        "metrics": metrics,
    }


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize01(v: np.ndarray) -> np.ndarray:
    """
    Normalisation [0,1] pour comparaison visuelle.
    """
    v = np.asarray(v, dtype=float)
    vmin, vmax = np.min(v), np.max(v)
    if np.isclose(vmin, vmax):
        return np.zeros_like(v)
    return (v - vmin) / (vmax - vmin)


# =============================================================================
# 4. Fonctions de tracé
# =============================================================================

def plot_scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, s=8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_trajectory_comparison(
    t: np.ndarray,
    z: np.ndarray,
    d: np.ndarray,
    title: str,
    output_path: str | Path,
) -> None:
    """
    Compare visuellement z(t) et dissipation(t) après normalisation [0,1].
    """
    z01 = normalize01(z)
    d01 = normalize01(d)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, z01, label="z(t) normalisé")
    ax.plot(t, d01, label="dissipation(t) normalisée")
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("Valeur normalisée")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# =============================================================================
# 5. Script principal
# =============================================================================

def main() -> Dict[str, Any]:
    config = load_config_object()

    data_dir = Path(_get_value(config, "paths.data_dir", "outputs/data"))
    models_dir = Path(_get_value(config, "paths.models_dir", "outputs/models"))
    logs_dir = ensure_dir(Path(_get_value(config, "paths.logs_dir", "outputs/logs")))
    figures_dir = ensure_dir(Path(_get_value(config, "paths.figures_dir", "outputs/figures")))
    dataset_path = data_dir / str(_get_value(config, "dataset.test_filename", "test_data.npz"))
    device = _get_value(config, "ml.device", "cpu")
    nu = float(_get_value(config, "pde.nu", 0.05))

    checkpoint_path = models_dir / "latent_dynamics" / "best_latent_dynamics_model.pt"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint introuvable : {checkpoint_path}")

    print_section("1. Chargement du dataset")
    dataset = plot_module.load_npz_dataset(dataset_path)
    states = np.asarray(dataset["states"], dtype=float)
    x = np.asarray(dataset["x"], dtype=float)
    dx = float(np.asarray(dataset["dx"]).item())
    times = np.asarray(dataset["times"], dtype=float)

    print(f"- dataset_path : {dataset_path}")
    print(f"- states shape : {states.shape}")
    print(f"- nu           : {nu}")

    print_section("2. Reconstruction de z via le checkpoint latent_dynamics")
    results = eval_module.evaluate_saved_latent_dynamics_model(
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        batch_size=128,
        stride=None,
        device=device,
        output_json_path=None,
    )

    z = flatten_scalar(results["extracted"]["z_t"])
    trajectory_index = np.asarray(results["extracted"]["trajectory_index"], dtype=int)
    time_index = np.asarray(results["extracted"]["time_index_t"], dtype=int)

    print(f"- nombre de points z : {z.shape[0]}")

    print_section("3. Calcul de la dissipation candidate")
    candidate_tensor = candidate_module.compute_candidate_tensor_from_dataset(
        states=states,
        x=x,
        dx=dx,
        nu=nu,
    )

    dissipation = np.asarray(
        candidate_tensor["dissipation_nu_grad2"][trajectory_index, time_index],
        dtype=float,
    )
    gradient_l2 = np.asarray(
        candidate_tensor["gradient_l2"][trajectory_index, time_index],
        dtype=float,
    )
    max_abs_grad = np.asarray(
        candidate_tensor["max_abs_gradient"][trajectory_index, time_index],
        dtype=float,
    )
    energy = np.asarray(
        candidate_tensor["energy"][trajectory_index, time_index],
        dtype=float,
    )

    # Temps associé à chaque échantillon plat
    time_values = np.asarray(times[time_index], dtype=float)


    eps = 1.0e-12
    log_diss = np.log(dissipation + eps)
    sqrt_diss = np.sqrt(np.maximum(dissipation, 0.0))

    print(f"- dissipation min/max : {float(np.min(dissipation)):.6e} / {float(np.max(dissipation)):.6e}")

    print_section("4. Comparaisons quantitatives")
    comparisons = {
        "z_vs_dissipation": regression_metrics(z, dissipation),
        "-z_vs_dissipation": regression_metrics(-z, dissipation),
        "z_vs_log_dissipation": regression_metrics(z, log_diss),
        "-z_vs_log_dissipation": regression_metrics(-z, log_diss),
        "z_vs_sqrt_dissipation": regression_metrics(z, sqrt_diss),
        "-z_vs_sqrt_dissipation": regression_metrics(-z, sqrt_diss),
        "z_vs_gradient_l2": regression_metrics(z, gradient_l2),
        "-z_vs_gradient_l2": regression_metrics(-z, gradient_l2),
        "z_vs_max_abs_gradient": regression_metrics(z, max_abs_grad),
        "-z_vs_max_abs_gradient": regression_metrics(-z, max_abs_grad),
    }

    affine_comparisons = {
        "dissipation_from_z": affine_fit(z, dissipation),
        "dissipation_from_minus_z": affine_fit(-z, dissipation),
        "log_dissipation_from_z": affine_fit(z, log_diss),
        "sqrt_dissipation_from_z": affine_fit(z, sqrt_diss),
        "gradient_l2_from_z": affine_fit(z, gradient_l2),
        "max_abs_gradient_from_z": affine_fit(z, max_abs_grad),
    }

    print_section("4bis. Analyse des résidus pour la dissipation")

    # On choisit automatiquement la meilleure convention de signe
    # pour approximer la dissipation à partir de z :
    # dissipation_from_z ou dissipation_from_minus_z
    if affine_comparisons["dissipation_from_z"]["metrics"]["rmse"] <= affine_comparisons["dissipation_from_minus_z"]["metrics"]["rmse"]:
        best_dissipation_fit_name = "dissipation_from_z"
        z_used = z
        z_used_label = "z"
    else:
        best_dissipation_fit_name = "dissipation_from_minus_z"
        z_used = -z
        z_used_label = "-z"

    best_dissipation_fit = affine_comparisons[best_dissipation_fit_name]
    fitted_dissipation = best_dissipation_fit["fitted"]
    residuals_dissipation = dissipation - fitted_dissipation

    residual_metrics = {
        "residual_vs_energy": regression_metrics(residuals_dissipation, energy),
        "residual_vs_time": regression_metrics(residuals_dissipation, time_values),
        "residual_vs_max_abs_gradient": regression_metrics(residuals_dissipation, max_abs_grad),
    }

    print(f"- meilleure convention de signe : {z_used_label}")
    print(f"- fit utilisé                   : {best_dissipation_fit_name}")

    for name, metrics in residual_metrics.items():
        print(
            f"{name:28s} | "
            f"pearson={metrics['pearson']:+.4f} | "
            f"spearman={metrics['spearman']:+.4f} | "
            f"rmse={metrics['rmse']:.4e}"
        )


    for name, metrics in comparisons.items():
        print(f"{name:28s} | pearson={metrics['pearson']:+.4f} | spearman={metrics['spearman']:+.4f} | rmse={metrics['rmse']:.4e}")

    print_section("5. Génération des figures globales")

    plot_scatter(
        z, dissipation,
        title="z vs dissipation brute",
        xlabel="z",
        ylabel="dissipation = nu * ∫|u_x|² dx",
        output_path=figures_dir / "compare_z_dissipation_scatter.png",
    )

    plot_scatter(
        -z, dissipation,
        title="-z vs dissipation brute",
        xlabel="-z",
        ylabel="dissipation = nu * ∫|u_x|² dx",
        output_path=figures_dir / "compare_minus_z_dissipation_scatter.png",
    )

    plot_scatter(
        z, log_diss,
        title="z vs log(dissipation)",
        xlabel="z",
        ylabel="log(dissipation + eps)",
        output_path=figures_dir / "compare_z_log_dissipation_scatter.png",
    )

    plot_scatter(
        z, sqrt_diss,
        title="z vs sqrt(dissipation)",
        xlabel="z",
        ylabel="sqrt(dissipation)",
        output_path=figures_dir / "compare_z_sqrt_dissipation_scatter.png",
    )

    # -------------------------------------------------------------
    # Plots de résidus pour le meilleur fit dissipation <- z
    # -------------------------------------------------------------
    plot_scatter(
        energy,
        residuals_dissipation,
        title=f"Résidu (dissipation - fit({z_used_label})) vs énergie",
        xlabel="Énergie",
        ylabel="Résidu dissipation",
        output_path=figures_dir / "compare_residuals_vs_energy.png",
    )

    plot_scatter(
        time_values,
        residuals_dissipation,
        title=f"Résidu (dissipation - fit({z_used_label})) vs temps",
        xlabel="Temps",
        ylabel="Résidu dissipation",
        output_path=figures_dir / "compare_residuals_vs_time.png",
    )

    plot_scatter(
        max_abs_grad,
        residuals_dissipation,
        title=f"Résidu (dissipation - fit({z_used_label})) vs max|u_x|",
        xlabel="max|u_x|",
        ylabel="Résidu dissipation",
        output_path=figures_dir / "compare_residuals_vs_max_abs_gradient.png",
    )


    print_section("6. Génération de courbes temporelles sur quelques trajectoires")

    # Reconstruction trajectoire par trajectoire
    n_traj, n_times = states.shape[0], states.shape[1]
    z_traj = np.full((n_traj, n_times), np.nan, dtype=float)
    diss_traj = candidate_tensor["dissipation_nu_grad2"].copy()

    z_traj[trajectory_index, time_index] = z

    # On choisit quelques trajectoires simples
    selected_trajectories = [0, 1, 2]
    for k in selected_trajectories:
        if np.any(np.isnan(z_traj[k])):
            continue

        plot_trajectory_comparison(
            t=times,
            z=z_traj[k],
            d=diss_traj[k],
            title=f"Comparaison z(t) / dissipation(t) — trajectoire {k}",
            output_path=figures_dir / f"compare_z_dissipation_traj{k}.png",
        )

    print_section("7. Sauvegarde du résumé")
    
    summary = {
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint_path),
        "comparisons": comparisons,
        "affine_comparisons": {
            key: {
                "a": value["a"],
                "b": value["b"],
                "metrics": value["metrics"],
            }
            for key, value in affine_comparisons.items()
        },
        "best_dissipation_fit": {
            "fit_name": best_dissipation_fit_name,
            "z_used_label": z_used_label,
            "a": best_dissipation_fit["a"],
            "b": best_dissipation_fit["b"],
            "metrics": best_dissipation_fit["metrics"],
        },
        "residual_analysis": {
            "residual_vs_energy": residual_metrics["residual_vs_energy"],
            "residual_vs_time": residual_metrics["residual_vs_time"],
            "residual_vs_max_abs_gradient": residual_metrics["residual_vs_max_abs_gradient"],
        },
        "generated_figures": [
            "compare_z_dissipation_scatter.png",
            "compare_minus_z_dissipation_scatter.png",
            "compare_z_log_dissipation_scatter.png",
            "compare_z_sqrt_dissipation_scatter.png",
            "compare_z_dissipation_traj0.png",
            "compare_z_dissipation_traj1.png",
            "compare_z_dissipation_traj2.png",
            "compare_residuals_vs_energy.png",
            "compare_residuals_vs_time.png",
            "compare_residuals_vs_max_abs_gradient.png",
        ],
    }

    save_json(logs_dir / "compare_z_and_dissipation_summary.json", summary)

    print("Résumé sauvegardé dans :")
    print(logs_dir / "compare_z_and_dissipation_summary.json")

    print_section("Terminé")
    print("Tu peux maintenant regarder :")
    print("- les scatter plots dans outputs/figures/")
    print("- les courbes temporelles z(t) / dissipation(t)")
    print("- le fichier JSON de résumé")

    return summary


if __name__ == "__main__":
    main()