"""
physics_candidates.py
=====================

Bibliothèque d'observables candidates pour identifier, a posteriori,
quelle variable physique (ou pseudo-physique) pourrait être approchée
par la variable scalaire latente apprise par le modèle.

Idée
----
À partir d'un champ 1D u(x), on construit un dictionnaire d'observables :
- masse,
- normes L1 / L2 / Linf,
- énergie,
- dissipation / gradients,
- variation totale,
- localisation de choc,
- contenu spectral / modes dominants,
- statistiques simples.

Ces observables peuvent ensuite être comparées à la variable latente z
par corrélation, calibration affine, et cohérence de monotonie.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
import pandas as pd


# =============================================================================
# 1. Utilitaires statistiques simples
# =============================================================================

def flatten_scalar_array(a: np.ndarray) -> np.ndarray:
    """Aplati une observable scalaire (n,) ou (n,1) vers (n,)."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a
    if a.ndim == 2 and a.shape[1] == 1:
        return a[:, 0]
    raise ValueError(f"Observable scalaire attendue de forme (n,) ou (n,1), obtenu {a.shape}")



def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = flatten_scalar_array(np.asarray(x, dtype=float))
    y = flatten_scalar_array(np.asarray(y, dtype=float))
    if x.shape != y.shape:
        raise ValueError(f"Formes incompatibles : {x.shape} vs {y.shape}")
    sx = np.std(x)
    sy = np.std(y)
    if sx < 1.0e-14 or sy < 1.0e-14:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])



def rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    sorter = np.argsort(a, kind='mergesort')
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



def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_correlation(rankdata_average(x), rankdata_average(y))



def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = flatten_scalar_array(np.asarray(pred, dtype=float))
    target = flatten_scalar_array(np.asarray(target, dtype=float))
    err = pred - target
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    max_abs = float(np.max(np.abs(err)))
    var_target = float(np.var(target))
    r2 = float(1.0 - mse / var_target) if var_target > 1.0e-14 else 0.0
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'max_abs_error': max_abs,
        'r2': r2,
    }



def fit_affine_calibration(source: np.ndarray, target: np.ndarray) -> Dict[str, Any]:
    """Ajuste target ≈ a * source + b."""
    source = flatten_scalar_array(np.asarray(source, dtype=float))
    target = flatten_scalar_array(np.asarray(target, dtype=float))
    A = np.column_stack([source, np.ones_like(source)])
    coeffs, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    a, b = float(coeffs[0]), float(coeffs[1])
    calibrated = a * source + b
    return {
        'a': a,
        'b': b,
        'calibrated': calibrated,
        'metrics': regression_metrics(calibrated, target),
    }



def compute_monotonicity_metrics(values: np.ndarray, atol: float = 0.0) -> Dict[str, float | bool]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError('`values` doit être 1D et contenir au moins deux valeurs.')
    diffs = values[1:] - values[:-1]
    violations = np.maximum(0.0, diffs - float(atol))
    return {
        'monotone_fraction': float(np.mean(diffs <= atol)),
        'mean_positive_violation': float(np.mean(violations)),
        'max_positive_violation': float(np.max(violations)),
        'is_nonincreasing': bool(np.all(diffs <= atol)),
    }


# =============================================================================
# 2. Observables à partir d'un snapshot u(x)
# =============================================================================

def periodic_gradient(u: np.ndarray, dx: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)



def periodic_laplacian(u: np.ndarray, dx: float) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx**2)



def compute_fourier_amplitudes(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    coeffs = np.fft.rfft(u)
    amps = np.abs(coeffs)
    return amps



def compute_candidate_observables_for_snapshot(u: np.ndarray, x: np.ndarray, dx: float, nu: float | None = None) -> Dict[str, float]:
    """
    Construit un dictionnaire d'observables candidates à partir d'un seul snapshot.
    """
    u = np.asarray(u, dtype=float)
    x = np.asarray(x, dtype=float)
    ux = periodic_gradient(u, dx)

    mass = float(dx * np.sum(u))
    mean_u = float(np.mean(u))
    l1 = float(dx * np.sum(np.abs(u)))
    l2 = float(np.sqrt(dx * np.sum(u**2)))
    linf = float(np.max(np.abs(u)))
    energy = float(0.5 * dx * np.sum(u**2))

    total_variation = float(dx * np.sum(np.abs(ux)))
    gradient_l2_sq = float(dx * np.sum(ux**2))
    gradient_l2 = float(np.sqrt(max(gradient_l2_sq, 0.0)))
    max_abs_gradient = float(np.max(np.abs(ux)))

    idx_shock = int(np.argmax(np.abs(ux)))
    shock_location = float(x[idx_shock])
    shock_strength_signed = float(ux[idx_shock])

    positive_mass = float(dx * np.sum(np.maximum(u, 0.0)))
    negative_mass = float(dx * np.sum(np.maximum(-u, 0.0)))

    amps = compute_fourier_amplitudes(u)
    # Ignore k=0 for dominant oscillatory mode if possible
    if len(amps) > 1:
        dom_k = int(1 + np.argmax(amps[1:]))
        dom_amp = float(amps[dom_k])
    else:
        dom_k = 0
        dom_amp = float(amps[0])

    amp_sum = float(np.sum(amps[1:])) if len(amps) > 1 else 0.0
    spectral_centroid = float(np.sum(np.arange(len(amps)) * amps) / np.sum(amps)) if np.sum(amps) > 1e-14 else 0.0
    low_freq_sum = float(np.sum(amps[1:4])) if len(amps) > 4 else float(np.sum(amps[1:]))
    high_freq_sum = float(np.sum(amps[4:])) if len(amps) > 4 else 0.0
    low_high_ratio = float(low_freq_sum / max(high_freq_sum, 1.0e-12))

    centered = u - mean_u
    std_u = float(np.std(centered))
    if std_u > 1.0e-14:
        skewness = float(np.mean((centered / std_u) ** 3))
        kurtosis = float(np.mean((centered / std_u) ** 4))
    else:
        skewness = 0.0
        kurtosis = 0.0

    out = {
        'mass': mass,
        'mean_u': mean_u,
        'l1_norm': l1,
        'l2_norm': l2,
        'linf_norm': linf,
        'energy': energy,
        'positive_mass': positive_mass,
        'negative_mass': negative_mass,
        'total_variation': total_variation,
        'gradient_l2': gradient_l2,
        'gradient_l2_sq': gradient_l2_sq,
        'max_abs_gradient': max_abs_gradient,
        'shock_location': shock_location,
        'shock_strength_signed': shock_strength_signed,
        'dominant_mode_index': float(dom_k),
        'dominant_mode_amplitude': dom_amp,
        'spectral_centroid': spectral_centroid,
        'low_high_ratio': low_high_ratio,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }

    # Amplitudes des premiers modes de Fourier
    for k in range(1, min(6, len(amps))):
        out[f'fourier_amp_k{k}'] = float(amps[k])

    # Forme de type entropie / moments absolus
    for p in [0.5, 1.5, 3.0, 4.0]:
        out[f'abs_moment_p{str(p).replace('.', '_')}'] = float(dx * np.sum(np.abs(u) ** p))

   # Dissipation si nu connu
    eps = 1.0e-12

    if nu is not None:
        dissipation = float(nu * gradient_l2_sq)
        out['dissipation_nu_grad2'] = dissipation
        out['log_dissipation_nu_grad2'] = float(np.log(dissipation + eps))
        out['sqrt_dissipation_nu_grad2'] = float(np.sqrt(max(dissipation, 0.0)))

    # Transformations utiles des observables de gradient
    out['log_gradient_l2'] = float(np.log(gradient_l2 + eps))
    out['sqrt_gradient_l2'] = float(np.sqrt(max(gradient_l2, 0.0)))
    out['log_max_abs_gradient'] = float(np.log(max_abs_gradient + eps))

    return out


# =============================================================================
# 3. Observables pour tout un dataset
# =============================================================================

def compute_candidate_tensor_from_dataset(states: np.ndarray, x: np.ndarray, dx: float, nu: float | None = None) -> Dict[str, np.ndarray]:
    """
    Calcule toutes les observables candidates pour un tableau d'états.

    Paramètres
    ----------
    states : np.ndarray
        Tableau de forme (n_traj, n_times, Nx).

    Retour
    ------
    Dict[str, np.ndarray]
        Dictionnaire : nom_observable -> tableau (n_traj, n_times)
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 3:
        raise ValueError('`states` doit être de forme (n_traj, n_times, Nx).')

    n_traj, n_times, _ = states.shape
    first = compute_candidate_observables_for_snapshot(states[0, 0], x, dx, nu=nu)
    candidates = {name: np.zeros((n_traj, n_times), dtype=float) for name in first.keys()}

    for i in range(n_traj):
        for t in range(n_times):
            vals = compute_candidate_observables_for_snapshot(states[i, t], x, dx, nu=nu)
            for name, value in vals.items():
                candidates[name][i, t] = value
    return candidates



def extract_candidate_values_from_indices(candidate_tensor: Dict[str, np.ndarray], trajectory_index: np.ndarray, time_index: np.ndarray) -> Dict[str, np.ndarray]:
    """Extrait les valeurs d'observables selon des indices (traj, time) plats."""
    trajectory_index = np.asarray(trajectory_index, dtype=int)
    time_index = np.asarray(time_index, dtype=int)
    out = {}
    for name, values in candidate_tensor.items():
        out[name] = values[trajectory_index, time_index]
    return out


# =============================================================================
# 4. Comparaison z <-> observables candidates
# =============================================================================

def evaluate_candidate_against_latent(z: np.ndarray, candidate: np.ndarray, candidate_name: str) -> Dict[str, Any]:
    z = flatten_scalar_array(np.asarray(z, dtype=float))
    c = flatten_scalar_array(np.asarray(candidate, dtype=float))

    direct = regression_metrics(z, c)
    direct['pearson_correlation'] = pearson_correlation(z, c)
    direct['spearman_correlation'] = spearman_correlation(z, c)
    direct['abs_pearson'] = abs(direct['pearson_correlation'])
    direct['abs_spearman'] = abs(direct['spearman_correlation'])

    affine = fit_affine_calibration(z, c)

    increasing_mono = compute_monotonicity_metrics(z)  # tests nonincreasing by default
    # no trajectory info here, just placeholder for compatibility
    return {
        'candidate_name': candidate_name,
        'direct_metrics': direct,
        'affine_calibration': {
            'a': affine['a'],
            'b': affine['b'],
            'metrics': affine['metrics'],
        },
    }



def rank_candidates_against_latent(z: np.ndarray, candidate_values: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Retourne un DataFrame trié des candidats les plus proches de z.

    Tri principal : abs_spearman décroissant, puis RMSE affine croissant.
    """
    rows: List[Dict[str, Any]] = []
    for name, values in candidate_values.items():
        res = evaluate_candidate_against_latent(z, values, name)
        rows.append({
            'candidate': name,
            'pearson': res['direct_metrics']['pearson_correlation'],
            'spearman': res['direct_metrics']['spearman_correlation'],
            'abs_pearson': res['direct_metrics']['abs_pearson'],
            'abs_spearman': res['direct_metrics']['abs_spearman'],
            'direct_rmse': res['direct_metrics']['rmse'],
            'direct_mae': res['direct_metrics']['mae'],
            'affine_a': res['affine_calibration']['a'],
            'affine_b': res['affine_calibration']['b'],
            'affine_rmse': res['affine_calibration']['metrics']['rmse'],
            'affine_mae': res['affine_calibration']['metrics']['mae'],
            'affine_r2': res['affine_calibration']['metrics']['r2'],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['abs_spearman', 'affine_rmse'], ascending=[False, True]).reset_index(drop=True)
    return df


# =============================================================================
# 5. Sauvegarde pratique
# =============================================================================

def save_rankings(df: pd.DataFrame, output_dir: str | Path, prefix: str) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f'{prefix}_candidate_rankings.csv'
    json_path = output_dir / f'{prefix}_candidate_rankings.json'

    df.to_csv(csv_path, index=False)
    with json_path.open('w', encoding='utf-8') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2, ensure_ascii=False)

    return {
        'csv_path': str(csv_path),
        'json_path': str(json_path),
    }
