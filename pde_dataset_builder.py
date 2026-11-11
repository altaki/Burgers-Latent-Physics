"""
pde/dataset_builder.py
======================

Module autonome de construction de dataset pour l'étude :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module permet de générer directement des datasets de trajectoires sans
supposer que les autres modules du projet sont déjà installés comme package.
Il contient donc une version autoportée des briques nécessaires :

- génération de conditions initiales,
- résolution numérique simple de Burgers,
- calcul des observables physiques de référence,
- assemblage et sauvegarde des données.

Pourquoi une version autoportée ?
---------------------------------
Dans le cadre d'un prototypage rapide ou d'un échange de fichiers, il est
pratique d'avoir un fichier unique capable de fonctionner tout seul. Cela ne
remplace pas l'architecture modulaire du projet, mais cela permet de démarrer
sans friction.

Contenu principal
-----------------
Ce module fournit :

1. une fonction pour générer une trajectoire complète avec métadonnées,
2. une fonction pour générer un split entier (train / val / test),
3. des fonctions simples de sauvegarde / chargement,
4. un résumé statistique du dataset,
5. des sanity checks de base.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Sequence, Tuple
from pathlib import Path
import numpy as np


# =============================================================================
# A. UTILITAIRES GENERAUX
# =============================================================================

def _get_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Retourne un générateur numpy, en crée un si nécessaire."""
    return rng if rng is not None else np.random.default_rng()


def _validate_grid(x: np.ndarray) -> np.ndarray:
    """Vérifie que `x` est une grille 1D non vide."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")
    if x.size < 2:
        raise ValueError("`x` doit contenir au moins deux points.")
    return x


def _domain_length_from_grid(x: np.ndarray) -> float:
    """Retourne la longueur du domaine pour une grille uniforme endpoint=False."""
    x = _validate_grid(x)
    dx = x[1] - x[0]
    return float(x.size * dx)


def normalize_l2(u: np.ndarray, x: np.ndarray, target_norm: float = 1.0) -> np.ndarray:
    """Renormalise un champ pour lui imposer une norme L2 discrète cible."""
    x = _validate_grid(x)
    u = np.asarray(u, dtype=float)
    if u.shape != x.shape:
        raise ValueError("`u` et `x` doivent avoir la même forme.")

    dx = x[1] - x[0]
    current_norm = np.sqrt(np.sum(u ** 2) * dx)
    if current_norm < 1.0e-14:
        return u.copy()
    return (target_norm / current_norm) * u


# =============================================================================
# B. CONDITIONS INITIALES (VERSION AUTO-PORTEE)
# =============================================================================

def random_fourier_ic(
    x: np.ndarray,
    max_mode: int = 5,
    amplitude_scale: float = 1.0,
    zero_mean: bool = True,
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Condition initiale = combinaison aléatoire de modes de Fourier."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    if max_mode < 1:
        raise ValueError("`max_mode` doit être >= 1.")

    L = _domain_length_from_grid(x)
    u0 = np.zeros_like(x, dtype=float)
    for k in range(1, max_mode + 1):
        a_k = amplitude_scale * rng.uniform(-1.0, 1.0)
        b_k = amplitude_scale * rng.uniform(-1.0, 1.0)
        phase = 2.0 * np.pi * k * x / L
        u0 += a_k * np.sin(phase) + b_k * np.cos(phase)

    if not zero_mean:
        u0 += amplitude_scale * rng.uniform(-1.0, 1.0)

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)
    return u0


def _periodic_distance(x: np.ndarray, center: float, L: float) -> np.ndarray:
    """Distance périodique minimale entre x et un centre."""
    raw = np.abs(x - center)
    return np.minimum(raw, L - raw)


def gaussian_bumps_ic(
    x: np.ndarray,
    n_bumps: int = 2,
    amplitude_range: Sequence[float] = (-1.0, 1.0),
    width_range: Sequence[float] = (0.1, 0.4),
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Condition initiale = somme de bosses gaussiennes périodisées."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    if n_bumps < 1:
        raise ValueError("`n_bumps` doit être >= 1.")

    amin, amax = float(amplitude_range[0]), float(amplitude_range[1])
    wmin, wmax = float(width_range[0]), float(width_range[1])
    if wmin <= 0.0 or wmax <= 0.0:
        raise ValueError("Les largeurs doivent être strictement positives.")

    L = _domain_length_from_grid(x)
    u0 = np.zeros_like(x, dtype=float)
    for _ in range(n_bumps):
        center = rng.uniform(x[0], x[0] + L)
        amplitude = rng.uniform(amin, amax)
        sigma = rng.uniform(wmin, wmax) * L
        d = _periodic_distance(x, center, L)
        u0 += amplitude * np.exp(-(d ** 2) / (sigma ** 2))

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)
    return u0


def _enforce_hermitian_symmetry(coeffs: np.ndarray) -> np.ndarray:
    """Impose une symétrie hermitienne pour obtenir une ifft réelle."""
    coeffs = np.asarray(coeffs, dtype=complex).copy()
    N = coeffs.size
    coeffs[0] = coeffs[0].real + 0j
    if N % 2 == 0:
        coeffs[N // 2] = coeffs[N // 2].real + 0j
    for k in range(1, N // 2 + (0 if N % 2 == 0 else 1)):
        coeffs[-k] = np.conjugate(coeffs[k])
    return coeffs


def random_smooth_ic(
    x: np.ndarray,
    spectral_decay: float = 2.0,
    amplitude_scale: float = 1.0,
    zero_mean: bool = True,
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Condition initiale = champ aléatoire lisse construit spectralement."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    Nx = x.size

    modes = np.fft.fftfreq(Nx, d=1.0 / Nx)
    coeffs = rng.normal(size=Nx) + 1j * rng.normal(size=Nx)
    coeffs *= amplitude_scale * (1.0 + np.abs(modes)) ** (-spectral_decay)

    if zero_mean:
        coeffs[0] = 0.0
    coeffs = _enforce_hermitian_symmetry(coeffs)
    u0 = np.fft.ifft(coeffs).real

    if normalize:
        u0 = normalize_l2(u0, x, target_norm=target_norm)
    return u0


def sample_initial_condition(
    x: np.ndarray,
    ic_type: str = "fourier",
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> np.ndarray:
    """Interface unique pour générer une condition initiale."""
    x = _validate_grid(x)
    rng = _get_rng(rng)
    ic_type = ic_type.lower()

    # On filtre les kwargs selon la famille choisie, pour éviter les erreurs
    # quand on mutualise un même dictionnaire de paramètres.
    if ic_type == "fourier":
        allowed = {"max_mode", "amplitude_scale", "zero_mean"}
        family_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return random_fourier_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **family_kwargs,
        )

    if ic_type == "gaussian_bumps":
        allowed = {"n_bumps", "amplitude_range", "width_range"}
        family_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return gaussian_bumps_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **family_kwargs,
        )

    if ic_type == "random_smooth":
        allowed = {"spectral_decay", "amplitude_scale", "zero_mean"}
        family_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return random_smooth_ic(
            x,
            normalize=normalize,
            target_norm=target_norm,
            rng=rng,
            **family_kwargs,
        )

    raise ValueError(
        f"Type de condition initiale inconnu : {ic_type!r}. "
        "Choisir parmi {'fourier', 'gaussian_bumps', 'random_smooth'}."
    )


def sample_initial_condition_from_list(
    x: np.ndarray,
    ic_types: Sequence[str],
    normalize: bool = False,
    target_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    **kwargs,
) -> tuple[np.ndarray, str]:
    """Tire aléatoirement un type de CI parmi une liste puis génère le champ."""
    if len(ic_types) == 0:
        raise ValueError("La séquence `ic_types` ne doit pas être vide.")
    rng = _get_rng(rng)
    chosen_type = str(rng.choice(list(ic_types)))
    u0 = sample_initial_condition(
        x,
        ic_type=chosen_type,
        normalize=normalize,
        target_norm=target_norm,
        rng=rng,
        **kwargs,
    )
    return u0, chosen_type


# =============================================================================
# C. SOLVEUR BURGERS (VERSION AUTO-PORTEE)
# =============================================================================

def build_spatial_grid(L: float, Nx: int) -> tuple[np.ndarray, float]:
    """Construit une grille uniforme périodique sur [0, L)."""
    if L <= 0.0:
        raise ValueError("`L` doit être strictement positif.")
    if Nx < 4:
        raise ValueError("`Nx` doit être >= 4.")
    x = np.linspace(0.0, L, Nx, endpoint=False)
    dx = x[1] - x[0]
    return x, float(dx)


def compute_first_derivative_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """Approxime u_x par différences finies centrées périodiques."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    return (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)


def compute_second_derivative_periodic(u: np.ndarray, dx: float) -> np.ndarray:
    """Approxime u_xx par différences finies centrées périodiques."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx ** 2)


def burgers_rhs(u: np.ndarray, dx: float, nu: float) -> np.ndarray:
    """Second membre de Burgers : u_t = -u u_x + nu u_xx."""
    ux = compute_first_derivative_periodic(u, dx)
    uxx = compute_second_derivative_periodic(u, dx)
    return -u * ux + nu * uxx


def step_euler_explicit(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Un pas en temps par Euler explicite."""
    return u + dt * burgers_rhs(u, dx, nu)


def step_rk4(u: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Un pas en temps par Runge-Kutta d'ordre 4."""
    k1 = burgers_rhs(u, dx, nu)
    k2 = burgers_rhs(u + 0.5 * dt * k1, dx, nu)
    k3 = burgers_rhs(u + 0.5 * dt * k2, dx, nu)
    k4 = burgers_rhs(u + dt * k3, dx, nu)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def estimate_stability_indicators(u: np.ndarray, dt: float, dx: float, nu: float) -> Dict[str, float]:
    """Calcule quelques indicateurs simples de stabilité / raideur."""
    u = np.asarray(u, dtype=float)
    max_abs_u = float(np.max(np.abs(u)))
    advective_cfl = float(dt * max_abs_u / dx) if dx > 0.0 else np.inf
    diffusive_number = float(nu * dt / (dx ** 2)) if dx > 0.0 else np.inf
    return {
        "max_abs_u": max_abs_u,
        "advective_cfl": advective_cfl,
        "diffusive_number": diffusive_number,
    }


def solve_burgers(
    u0: np.ndarray,
    L: float,
    T: float,
    dt: float,
    nu: float,
    save_every: int = 1,
    time_scheme: str = "rk4",
    check_nan: bool = True,
    return_full_grid: bool = True,
) -> Dict[str, Any]:
    """Résout numériquement Burgers sur [0, L] périodique."""
    u0 = np.asarray(u0, dtype=float)
    if u0.ndim != 1:
        raise ValueError("`u0` doit être 1D.")
    if T <= 0.0 or dt <= 0.0:
        raise ValueError("`T` et `dt` doivent être strictement positifs.")
    if save_every < 1:
        raise ValueError("`save_every` doit être >= 1.")
    if nu < 0.0:
        raise ValueError("`nu` doit être positive ou nulle.")

    Nx = u0.size
    x, dx = build_spatial_grid(L, Nx)

    n_steps = int(np.round(T / dt))
    if n_steps < 1:
        raise ValueError("Le nombre de pas de temps doit être >= 1.")

    scheme = time_scheme.lower()
    if scheme == "euler":
        step_function = step_euler_explicit
    elif scheme == "rk4":
        step_function = step_rk4
    else:
        raise ValueError("`time_scheme` doit être 'euler' ou 'rk4'.")

    n_saved = n_steps // save_every + 1
    states = np.empty((n_saved, Nx), dtype=float)
    times = np.empty(n_saved, dtype=float)

    u = u0.copy()
    states[0] = u
    times[0] = 0.0
    save_id = 1
    stability_indicators_initial = estimate_stability_indicators(u, dt, dx, nu)

    for n in range(1, n_steps + 1):
        u = step_function(u, dt, dx, nu)

        if check_nan and (not np.all(np.isfinite(u))):
            raise FloatingPointError(
                f"Des valeurs non finies ont été détectées au pas de temps {n}."
            )

        if n % save_every == 0:
            states[save_id] = u
            times[save_id] = n * dt
            save_id += 1

    result: Dict[str, Any] = {
        "times": times,
        "states": states,
        "stability_indicators_initial": stability_indicators_initial,
    }
    if return_full_grid:
        result["x"] = x
        result["dx"] = dx
    return result


# =============================================================================
# D. OBSERVABLES (VERSION AUTO-PORTEE)
# =============================================================================

def compute_energy(u: np.ndarray, dx: float) -> float:
    """Énergie L2 discrète : 1/2 * somme(u^2) * dx."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    return 0.5 * float(np.sum(u ** 2) * dx)


def compute_energy_trajectory(states: np.ndarray, dx: float) -> np.ndarray:
    """Énergie pour tous les snapshots d'une trajectoire."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("`states` doit être 2D de forme (n_times, Nx).")
    return 0.5 * np.sum(states ** 2, axis=1) * dx


def compute_spatial_gradient(u: np.ndarray, dx: float) -> np.ndarray:
    """Gradient spatial périodique par différences finies centrées."""
    return compute_first_derivative_periodic(u, dx)


def compute_dissipation(u: np.ndarray, dx: float) -> float:
    """Dissipation : somme(|u_x|^2) * dx."""
    ux = compute_spatial_gradient(u, dx)
    return float(np.sum(ux ** 2) * dx)


def compute_dissipation_trajectory(states: np.ndarray, dx: float) -> np.ndarray:
    """Dissipation pour tous les snapshots d'une trajectoire."""
    states = np.asarray(states, dtype=float)
    if states.ndim != 2:
        raise ValueError("`states` doit être 2D de forme (n_times, Nx).")
    out = np.empty(states.shape[0], dtype=float)
    for n in range(states.shape[0]):
        out[n] = compute_dissipation(states[n], dx)
    return out


def compute_monotonicity_violations(values: np.ndarray) -> np.ndarray:
    """Violations positives d'une monotonie décroissante."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("`values` doit être 1D.")
    diffs = values[1:] - values[:-1]
    return np.maximum(0.0, diffs)


def compute_monotonicity_metrics(values: np.ndarray, atol: float = 0.0) -> Dict[str, Any]:
    """Métriques simples de monotonie décroissante."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("`values` doit être 1D et contenir au moins deux valeurs.")

    diffs = values[1:] - values[:-1]
    positive_violations = np.maximum(0.0, diffs - atol)
    return {
        "n_steps": int(values.size - 1),
        "monotone_fraction": float(np.mean(diffs <= atol)),
        "mean_positive_violation": float(np.mean(positive_violations)),
        "max_positive_violation": float(np.max(positive_violations)),
        "is_nonincreasing": bool(np.all(diffs <= atol)),
    }


def compute_reference_observables(states: np.ndarray, dx: float) -> Dict[str, np.ndarray]:
    """Calcule les observables physiques de référence pour une trajectoire."""
    return {
        "energy": compute_energy_trajectory(states, dx),
        "dissipation": compute_dissipation_trajectory(states, dx),
    }


# =============================================================================
# E. CONSTRUCTION DU DATASET
# =============================================================================

def save_dataset_npz(path: str | Path, dataset: Dict[str, Any]) -> None:
    """Sauvegarde un dataset dans un fichier .npz compressé."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for key, value in dataset.items():
        serializable[key] = np.asarray(value) if isinstance(value, list) else value
    np.savez_compressed(path, **serializable)


def load_dataset_npz(path: str | Path) -> Dict[str, Any]:
    """Recharge un dataset sauvegardé au format .npz."""
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_ic_type_mapping(ic_types: Sequence[str]) -> Dict[str, int]:
    """Construit un dictionnaire nom -> identifiant entier pour les familles de CI."""
    if len(ic_types) == 0:
        raise ValueError("La séquence `ic_types` ne doit pas être vide.")
    return {name: idx for idx, name in enumerate(ic_types)}


def generate_single_trajectory(
    x: np.ndarray,
    L: float,
    T: float,
    dt: float,
    nu: float,
    save_every: int,
    time_scheme: str,
    ic_type: Optional[str] = None,
    allowed_ic_types: Optional[Sequence[str]] = None,
    normalize_ic: bool = False,
    target_ic_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    ic_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Génère une trajectoire unique complète avec observables et métadonnées."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")
    if ic_kwargs is None:
        ic_kwargs = {}
    rng = _get_rng(rng)

    # 1) Condition initiale
    if ic_type is not None:
        chosen_ic_type = str(ic_type)
        u0 = sample_initial_condition(
            x,
            ic_type=chosen_ic_type,
            normalize=normalize_ic,
            target_norm=target_ic_norm,
            rng=rng,
            **ic_kwargs,
        )
    else:
        if allowed_ic_types is None or len(allowed_ic_types) == 0:
            raise ValueError("Si `ic_type` est None, fournir `allowed_ic_types` non vide.")
        u0, chosen_ic_type = sample_initial_condition_from_list(
            x,
            ic_types=allowed_ic_types,
            normalize=normalize_ic,
            target_norm=target_ic_norm,
            rng=rng,
            **ic_kwargs,
        )

    # 2) Résolution de Burgers
    solution = solve_burgers(
        u0=u0,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme=time_scheme,
        check_nan=True,
        return_full_grid=True,
    )

    states = solution["states"]
    times = solution["times"]
    dx = float(solution["dx"])

    # 3) Observables de référence
    observables = compute_reference_observables(states, dx)
    energy_monotonicity = compute_monotonicity_metrics(observables["energy"], atol=1.0e-12)

    return {
        "u0": u0,
        "states": states,
        "times": times,
        "x": solution["x"],
        "dx": dx,
        "nu": float(nu),
        "ic_type": chosen_ic_type,
        "energy": observables["energy"],
        "dissipation": observables["dissipation"],
        "stability_indicators_initial": solution["stability_indicators_initial"],
        "energy_monotonicity": energy_monotonicity,
    }


def generate_dataset_split(
    n_trajectories: int,
    x: np.ndarray,
    L: float,
    T: float,
    dt: float,
    nu: float,
    save_every: int,
    time_scheme: str,
    ic_types: Sequence[str],
    normalize_ic: bool = False,
    target_ic_norm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    ic_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Génère un split complet du dataset (train / val / test)."""
    if n_trajectories < 1:
        raise ValueError("`n_trajectories` doit être >= 1.")

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")

    rng = _get_rng(rng)
    if ic_kwargs is None:
        ic_kwargs = {}

    ic_type_to_id = build_ic_type_mapping(ic_types)

    # Première trajectoire pour inférer les dimensions de sortie
    first = generate_single_trajectory(
        x=x,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme=time_scheme,
        ic_type=None,
        allowed_ic_types=ic_types,
        normalize_ic=normalize_ic,
        target_ic_norm=target_ic_norm,
        rng=rng,
        ic_kwargs=ic_kwargs,
    )

    states0 = first["states"]
    times = first["times"]
    x_out = first["x"]
    dx = float(first["dx"])
    n_times, Nx = states0.shape

    states = np.empty((n_trajectories, n_times, Nx), dtype=float)
    energy = np.empty((n_trajectories, n_times), dtype=float)
    dissipation = np.empty((n_trajectories, n_times), dtype=float)
    u0_all = np.empty((n_trajectories, Nx), dtype=float)
    nu_values = np.empty(n_trajectories, dtype=float)
    ic_type_ids = np.empty(n_trajectories, dtype=int)

    # Diagnostics simples par trajectoire
    initial_max_abs_u = np.empty(n_trajectories, dtype=float)
    initial_advective_cfl = np.empty(n_trajectories, dtype=float)
    initial_diffusive_number = np.empty(n_trajectories, dtype=float)
    energy_monotone_fraction = np.empty(n_trajectories, dtype=float)
    energy_mean_positive_violation = np.empty(n_trajectories, dtype=float)

    # Remplissage avec la première trajectoire
    states[0] = first["states"]
    energy[0] = first["energy"]
    dissipation[0] = first["dissipation"]
    u0_all[0] = first["u0"]
    nu_values[0] = first["nu"]
    ic_type_ids[0] = ic_type_to_id[first["ic_type"]]

    initial_max_abs_u[0] = first["stability_indicators_initial"]["max_abs_u"]
    initial_advective_cfl[0] = first["stability_indicators_initial"]["advective_cfl"]
    initial_diffusive_number[0] = first["stability_indicators_initial"]["diffusive_number"]
    energy_monotone_fraction[0] = first["energy_monotonicity"]["monotone_fraction"]
    energy_mean_positive_violation[0] = first["energy_monotonicity"]["mean_positive_violation"]

    if verbose:
        print(f"[dataset_builder] Trajectoire 1 / {n_trajectories} générée.")

    for i in range(1, n_trajectories):
        traj = generate_single_trajectory(
            x=x,
            L=L,
            T=T,
            dt=dt,
            nu=nu,
            save_every=save_every,
            time_scheme=time_scheme,
            ic_type=None,
            allowed_ic_types=ic_types,
            normalize_ic=normalize_ic,
            target_ic_norm=target_ic_norm,
            rng=rng,
            ic_kwargs=ic_kwargs,
        )

        states[i] = traj["states"]
        energy[i] = traj["energy"]
        dissipation[i] = traj["dissipation"]
        u0_all[i] = traj["u0"]
        nu_values[i] = traj["nu"]
        ic_type_ids[i] = ic_type_to_id[traj["ic_type"]]

        initial_max_abs_u[i] = traj["stability_indicators_initial"]["max_abs_u"]
        initial_advective_cfl[i] = traj["stability_indicators_initial"]["advective_cfl"]
        initial_diffusive_number[i] = traj["stability_indicators_initial"]["diffusive_number"]
        energy_monotone_fraction[i] = traj["energy_monotonicity"]["monotone_fraction"]
        energy_mean_positive_violation[i] = traj["energy_monotonicity"]["mean_positive_violation"]

        if verbose and ((i + 1) % max(1, n_trajectories // 10) == 0 or (i + 1) == n_trajectories):
            print(f"[dataset_builder] Trajectoire {i + 1} / {n_trajectories} générée.")

    dataset = {
        "states": states,
        "times": times,
        "energy": energy,
        "dissipation": dissipation,
        "u0": u0_all,
        "x": x_out,
        "dx": np.array(dx, dtype=float),
        "nu_values": nu_values,
        "ic_type_ids": ic_type_ids,
        "ic_type_names": np.asarray(list(ic_type_to_id.keys())),
        "initial_max_abs_u": initial_max_abs_u,
        "initial_advective_cfl": initial_advective_cfl,
        "initial_diffusive_number": initial_diffusive_number,
        "energy_monotone_fraction": energy_monotone_fraction,
        "energy_mean_positive_violation": energy_mean_positive_violation,
    }
    return dataset


def build_dataset_from_config(
    config: Any,
    split: str,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """Génère un split du dataset à partir de l'objet de configuration global."""
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("`split` doit être 'train', 'val' ou 'test'.")

    if split == "train":
        n_trajectories = int(config.dataset.n_train)
        filename = config.dataset.train_filename
    elif split == "val":
        n_trajectories = int(config.dataset.n_val)
        filename = config.dataset.val_filename
    else:
        n_trajectories = int(config.dataset.n_test)
        filename = config.dataset.test_filename

    L = float(config.pde.L)
    Nx = int(config.pde.Nx)
    x = np.linspace(0.0, L, Nx, endpoint=False)

    dataset = generate_dataset_split(
        n_trajectories=n_trajectories,
        x=x,
        L=L,
        T=float(config.pde.T),
        dt=float(config.pde.dt),
        nu=float(config.pde.nu),
        save_every=int(config.pde.save_every),
        time_scheme=str(config.pde.time_scheme),
        ic_types=list(config.dataset.ic_types),
        normalize_ic=True,
        target_ic_norm=1.0,
        rng=rng,
        ic_kwargs={
            "max_mode": int(config.dataset.max_fourier_mode),
            "amplitude_scale": float(config.dataset.fourier_amplitude_scale),
            "n_bumps": int(config.dataset.n_gaussian_bumps),
            "spectral_decay": float(config.dataset.spectral_decay),
        },
        verbose=verbose,
    )

    output_path = Path(config.paths.data_dir) / filename
    return dataset, output_path


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Retourne un résumé simple et lisible du dataset généré."""
    states = np.asarray(dataset["states"], dtype=float)
    energy = np.asarray(dataset["energy"], dtype=float)
    dissipation = np.asarray(dataset["dissipation"], dtype=float)
    ic_type_ids = np.asarray(dataset["ic_type_ids"], dtype=int)
    ic_type_names = np.asarray(dataset["ic_type_names"])

    if states.ndim != 3:
        raise ValueError("`dataset['states']` doit être de dimension 3 : (n_traj, n_times, Nx).")

    n_traj, n_times, Nx = states.shape
    unique_ids, counts = np.unique(ic_type_ids, return_counts=True)
    ic_distribution = {
        str(ic_type_names[idx]): int(count)
        for idx, count in zip(unique_ids, counts)
    }

    return {
        "n_trajectories": int(n_traj),
        "n_times": int(n_times),
        "Nx": int(Nx),
        "global_min_state": float(np.min(states)),
        "global_max_state": float(np.max(states)),
        "mean_initial_energy": float(np.mean(energy[:, 0])),
        "mean_final_energy": float(np.mean(energy[:, -1])),
        "mean_dissipation": float(np.mean(dissipation)),
        "ic_distribution": ic_distribution,
        "mean_energy_monotone_fraction": float(np.mean(dataset["energy_monotone_fraction"])),
        "mean_energy_positive_violation": float(np.mean(dataset["energy_mean_positive_violation"])),
    }


def basic_dataset_sanity_checks(dataset: Dict[str, Any]) -> Dict[str, bool]:
    """Effectue quelques tests très simples de cohérence structurelle."""
    states = np.asarray(dataset["states"])
    energy = np.asarray(dataset["energy"])
    dissipation = np.asarray(dataset["dissipation"])
    times = np.asarray(dataset["times"])
    x = np.asarray(dataset["x"])

    return {
        "states_is_finite": bool(np.all(np.isfinite(states))),
        "energy_is_finite": bool(np.all(np.isfinite(energy))),
        "dissipation_is_finite": bool(np.all(np.isfinite(dissipation))),
        "times_is_sorted": bool(np.all(np.diff(times) > 0.0)),
        "x_is_sorted": bool(np.all(np.diff(x) > 0.0)),
        "energy_nonnegative": bool(np.all(energy >= -1.0e-14)),
        "dissipation_nonnegative": bool(np.all(dissipation >= -1.0e-14)),
    }


def build_and_save_split_from_config(
    config: Any,
    split: str,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pipeline pratique : génère un split, le sauvegarde et retourne le dataset."""
    dataset, output_path = build_dataset_from_config(
        config=config,
        split=split,
        rng=rng,
        verbose=verbose,
    )
    save_dataset_npz(output_path, dataset)
    if verbose:
        print(f"[dataset_builder] Split '{split}' sauvegardé dans : {output_path}")
    return dataset


# =============================================================================
# E2. DATASET nu_sweep_v1 (BALAYAGE DE VISCOSITE)
# =============================================================================
# Protocole nu_sweep_v1 (voir section 8bis de config.py) :
# - viscosités stratifées par split (blocs contigus par nu, ex. 200/50/50),
# - familles de CI en round-robin dans chaque bloc de nu,
# - norme L2 cible log-uniforme dans target_norm_range (E(0) = 0.5 * norme^2),
# - zero_mean=True forcée pour TOUTES les familles de CI,
# - une graine par trajectoire, enregistrée dans `trajectory_seeds`,
# - nouvelles clés npz : target_l2_norm, energy0, mass0, dissipation_physical
#   (= nu_values * dissipation), trajectory_seeds, initial_cell_peclet,
#   dataset_version, protocol ; les clés legacy sont conservées telles quelles.

def compute_mass(u: np.ndarray, dx: float) -> float:
    """Masse discrète : somme(u) * dx (conservée au cours du temps)."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    return float(np.sum(u) * dx)


def compute_cell_peclet(u: np.ndarray, dx: float, nu: float) -> float:
    """Nombre de Péclet de maille : max|u| * dx / nu."""
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("`u` doit être 1D.")
    if nu <= 0.0:
        raise ValueError("`nu` doit être strictement positif.")
    return float(np.max(np.abs(u)) * dx / nu)


def _sample_log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
    """Tire un scalaire log-uniforme dans [low, high]."""
    if not (0.0 < low <= high):
        raise ValueError("`low` et `high` doivent vérifier 0 < low <= high.")
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def generate_nu_sweep_single_trajectory(
    x: np.ndarray,
    L: float,
    T: float,
    dt: float,
    nu: float,
    save_every: int,
    time_scheme: str,
    ic_type: str,
    target_norm_range: Sequence[float] = (0.5, 2.0),
    zero_mean: bool = True,
    trajectory_seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    ic_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Génère une trajectoire nu_sweep_v1 unique avec métadonnées enrichies.

    Spécificités du protocole nu_sweep_v1 :
    - une graine propre à la trajectoire (`trajectory_seed`) pilote toute la
      génération de la CI, ce qui rend chaque trajectoire individuellement
      reproductible ;
    - la norme L2 cible est tirée log-uniformément dans `target_norm_range`
      (d'où E(0) = 0.5 * norme^2) ;
    - la moyenne nulle est forcée pour TOUTES les familles de CI (y compris
      gaussian_bumps, qui n'expose pas de paramètre zero_mean) ;
    - la CI est ensuite renormalisée exactement vers la norme cible.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")
    if ic_kwargs is None:
        ic_kwargs = {}
    rng = _get_rng(rng)

    # Graine propre à la trajectoire : reproductibilité individuelle.
    if trajectory_seed is None:
        trajectory_seed = int(rng.integers(0, 2 ** 31 - 1))
    traj_rng = np.random.default_rng(trajectory_seed)

    # Norme L2 cible tirée log-uniformément.
    target_l2_norm = _sample_log_uniform(
        traj_rng, float(target_norm_range[0]), float(target_norm_range[1])
    )

    # 1) Condition initiale de la famille choisie (non normalisée ici).
    u0 = sample_initial_condition(
        x,
        ic_type=ic_type,
        normalize=False,
        target_norm=target_l2_norm,
        rng=traj_rng,
        **ic_kwargs,
    )

    # 2) Moyenne nulle forcée pour toutes les familles de CI.
    if zero_mean:
        u0 = u0 - np.mean(u0)

    # 3) Normalisation L2 exacte vers la cible.
    u0 = normalize_l2(u0, x, target_norm=target_l2_norm)

    # 4) Résolution de Burgers.
    solution = solve_burgers(
        u0=u0,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme=time_scheme,
        check_nan=True,
        return_full_grid=True,
    )

    states = solution["states"]
    times = solution["times"]
    dx = float(solution["dx"])

    # 5) Observables de référence.
    observables = compute_reference_observables(states, dx)
    energy_monotonicity = compute_monotonicity_metrics(observables["energy"], atol=1.0e-12)

    return {
        "u0": u0,
        "states": states,
        "times": times,
        "x": solution["x"],
        "dx": dx,
        "nu": float(nu),
        "ic_type": str(ic_type),
        "energy": observables["energy"],
        "dissipation": observables["dissipation"],
        "stability_indicators_initial": solution["stability_indicators_initial"],
        "energy_monotonicity": energy_monotonicity,
        # Métadonnées nu_sweep_v1.
        "target_l2_norm": target_l2_norm,
        "energy0": compute_energy(u0, dx),
        "mass0": compute_mass(u0, dx),
        "trajectory_seed": int(trajectory_seed),
        "initial_cell_peclet": compute_cell_peclet(u0, dx, nu),
    }


def generate_nu_sweep_split(
    n_trajectories: int,
    nu_values: Sequence[float],
    x: np.ndarray,
    L: float,
    T: float,
    dt: float,
    save_every: int,
    time_scheme: str,
    ic_types: Sequence[str],
    target_norm_range: Sequence[float] = (0.5, 2.0),
    zero_mean: bool = True,
    rng: Optional[np.random.Generator] = None,
    ic_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Génère un split nu_sweep_v1 complet, stratifié par viscosité.

    Protocole :
    - les trajectoires sont réparties en blocs contigus par valeur de nu
      (`n_trajectories` doit être divisible par `len(nu_values)`) ;
    - dans chaque bloc de nu, les familles de CI sont distribuées en
      round-robin (`ic_types[j % len(ic_types)]`) ;
    - chaque trajectoire a sa propre graine, enregistrée dans `trajectory_seeds`.

    Le dictionnaire retourné contient toutes les clés legacy (compatibilité
    avec `ml_datasets.py`) plus les nouvelles clés nu_sweep_v1 :
    `target_l2_norm`, `energy0`, `mass0`, `dissipation_physical`
    (= nu_values * dissipation, par trajectoire et par snapshot),
    `trajectory_seeds`, `initial_cell_peclet`.
    """
    # Validation + plan stratifié : blocs contigus par nu, familles de CI en
    # round-robin à l'intérieur de chaque bloc (voir build_stratified_assignments).
    plan = build_stratified_assignments(n_trajectories, nu_values, ic_types)

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("`x` doit être un tableau 1D.")

    rng = _get_rng(rng)
    if ic_kwargs is None:
        ic_kwargs = {}

    ic_type_to_id = build_ic_type_mapping(ic_types)

    # Génération trajectoire par trajectoire ; les tableaux sont alloués
    # après la première trajectoire, une fois les dimensions connues.
    states = None
    for i, (nu, ic_type) in enumerate(plan):
        trajectory_seed = int(rng.integers(0, 2 ** 31 - 1))
        traj = generate_nu_sweep_single_trajectory(
            x=x,
            L=L,
            T=T,
            dt=dt,
            nu=nu,
            save_every=save_every,
            time_scheme=time_scheme,
            ic_type=ic_type,
            target_norm_range=target_norm_range,
            zero_mean=zero_mean,
            trajectory_seed=trajectory_seed,
            rng=rng,
            ic_kwargs=ic_kwargs,
        )

        if i == 0:
            n_times, Nx = traj["states"].shape
            times = traj["times"]
            x_out = traj["x"]
            dx = float(traj["dx"])
            states = np.empty((n_trajectories, n_times, Nx), dtype=float)
            energy = np.empty((n_trajectories, n_times), dtype=float)
            dissipation = np.empty((n_trajectories, n_times), dtype=float)
            u0_all = np.empty((n_trajectories, Nx), dtype=float)
            nu_values_arr = np.empty(n_trajectories, dtype=float)
            ic_type_ids = np.empty(n_trajectories, dtype=int)
            initial_max_abs_u = np.empty(n_trajectories, dtype=float)
            initial_advective_cfl = np.empty(n_trajectories, dtype=float)
            initial_diffusive_number = np.empty(n_trajectories, dtype=float)
            energy_monotone_fraction = np.empty(n_trajectories, dtype=float)
            energy_mean_positive_violation = np.empty(n_trajectories, dtype=float)
            target_l2_norm_all = np.empty(n_trajectories, dtype=float)
            energy0_all = np.empty(n_trajectories, dtype=float)
            mass0_all = np.empty(n_trajectories, dtype=float)
            trajectory_seeds = np.empty(n_trajectories, dtype=np.int64)
            initial_cell_peclet = np.empty(n_trajectories, dtype=float)

        states[i] = traj["states"]
        energy[i] = traj["energy"]
        dissipation[i] = traj["dissipation"]
        u0_all[i] = traj["u0"]
        nu_values_arr[i] = traj["nu"]
        ic_type_ids[i] = ic_type_to_id[traj["ic_type"]]
        initial_max_abs_u[i] = traj["stability_indicators_initial"]["max_abs_u"]
        initial_advective_cfl[i] = traj["stability_indicators_initial"]["advective_cfl"]
        initial_diffusive_number[i] = traj["stability_indicators_initial"]["diffusive_number"]
        energy_monotone_fraction[i] = traj["energy_monotonicity"]["monotone_fraction"]
        energy_mean_positive_violation[i] = traj["energy_monotonicity"]["mean_positive_violation"]
        target_l2_norm_all[i] = traj["target_l2_norm"]
        energy0_all[i] = traj["energy0"]
        mass0_all[i] = traj["mass0"]
        trajectory_seeds[i] = traj["trajectory_seed"]
        initial_cell_peclet[i] = traj["initial_cell_peclet"]

        if verbose and ((i + 1) % max(1, n_trajectories // 10) == 0 or (i + 1) == n_trajectories):
            print(
                f"[dataset_builder][nu_sweep] Trajectoire {i + 1} / {n_trajectories} "
                f"générée (nu={nu}, CI={ic_type})."
            )

    dataset = {
        # Clés legacy (compatibilité avec ml_datasets.py et le code existant).
        "states": states,
        "times": times,
        "energy": energy,
        "dissipation": dissipation,
        "u0": u0_all,
        "x": x_out,
        "dx": np.array(dx, dtype=float),
        "nu_values": nu_values_arr,
        "ic_type_ids": ic_type_ids,
        "ic_type_names": np.asarray(list(ic_type_to_id.keys())),
        "initial_max_abs_u": initial_max_abs_u,
        "initial_advective_cfl": initial_advective_cfl,
        "initial_diffusive_number": initial_diffusive_number,
        "energy_monotone_fraction": energy_monotone_fraction,
        "energy_mean_positive_violation": energy_mean_positive_violation,
        # Nouvelles clés nu_sweep_v1.
        "target_l2_norm": target_l2_norm_all,
        "energy0": energy0_all,
        "mass0": mass0_all,
        "dissipation_physical": nu_values_arr[:, None] * dissipation,
        "trajectory_seeds": trajectory_seeds,
        "initial_cell_peclet": initial_cell_peclet,
    }
    return dataset


def build_nu_sweep_dataset_from_config(
    config: Any,
    split: str,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Path]:
    """
    Génère un split nu_sweep_v1 à partir de l'objet de configuration global.

    Le bloc `config.nu_sweep` (section 8bis de config.py) fournit la
    configuration numérique (Nx, dt, save_every, L, T, schéma temporel),
    les viscosités, les tailles de split et les noms de fichiers ; les
    familles de CI et leurs paramètres restent ceux du bloc
    `config.dataset` (familles identiques au dataset legacy).

    Retourne le dataset et le chemin de sortie prévu
    (`outputs/data/{split}_data_nu_sweep.npz`) SANS rien sauvegarder.
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("`split` doit être 'train', 'val' ou 'test'.")

    ns = config.nu_sweep
    if split == "train":
        n_trajectories = int(ns.n_train)
        filename = ns.train_filename
    elif split == "val":
        n_trajectories = int(ns.n_val)
        filename = ns.val_filename
    else:
        n_trajectories = int(ns.n_test)
        filename = ns.test_filename

    L = float(ns.L)
    Nx = int(ns.Nx)
    x = np.linspace(0.0, L, Nx, endpoint=False)

    dataset = generate_nu_sweep_split(
        n_trajectories=n_trajectories,
        nu_values=list(ns.nu_values),
        x=x,
        L=L,
        T=float(ns.T),
        dt=float(ns.dt),
        save_every=int(ns.save_every),
        time_scheme=str(ns.time_scheme),
        ic_types=list(config.dataset.ic_types),
        target_norm_range=list(ns.target_norm_range),
        zero_mean=bool(ns.zero_mean),
        rng=rng,
        ic_kwargs={
            "max_mode": int(config.dataset.max_fourier_mode),
            "amplitude_scale": float(config.dataset.fourier_amplitude_scale),
            "n_bumps": int(config.dataset.n_gaussian_bumps),
            "spectral_decay": float(config.dataset.spectral_decay),
            "zero_mean": bool(ns.zero_mean),
        },
        verbose=verbose,
    )

    # Métadonnées de version / protocole (scalaires, sérialisés dans le npz).
    dataset["dataset_version"] = str(ns.dataset_version)
    dataset["protocol"] = str(ns.protocol)

    output_path = Path(config.paths.data_dir) / filename
    return dataset, output_path


def build_and_save_nu_sweep_split_from_config(
    config: Any,
    split: str,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline pratique nu_sweep_v1 : génère un split, le sauvegarde dans
    `outputs/data/{split}_data_nu_sweep.npz`, et retourne le dataset.
    """
    dataset, output_path = build_nu_sweep_dataset_from_config(
        config=config,
        split=split,
        rng=rng,
        verbose=verbose,
    )
    save_dataset_npz(output_path, dataset)
    if verbose:
        print(f"[dataset_builder][nu_sweep] Split '{split}' sauvegardé dans : {output_path}")
    return dataset


def summarize_nu_sweep_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Résumé d'un dataset nu_sweep_v1 : résumé legacy plus statistiques
    spécifiques au protocole (distribution des nu, normes cibles, E(0),
    masse initiale, Péclet de maille initial).
    """
    summary = summarize_dataset(dataset)

    nu_arr = np.asarray(dataset["nu_values"], dtype=float)
    unique_nu, counts = np.unique(nu_arr, return_counts=True)
    summary["nu_distribution"] = {
        f"nu={v:g}": int(c) for v, c in zip(unique_nu, counts)
    }

    # Table d'assignation (nu, famille de CI) : effectifs par couple.
    # L'équilibrage des familles est vérifié À L'INTÉRIEUR de chaque strate
    # de nu (round-robin), pas seulement globalement.
    ic_names = np.asarray(dataset["ic_type_names"])
    ic_ids = np.asarray(dataset["ic_type_ids"], dtype=int)
    pair_counts: Dict[str, int] = {}
    for nu_val, ic_id in zip(nu_arr, ic_ids):
        key = f"nu={nu_val:g}|{ic_names[ic_id]}"
        pair_counts[key] = pair_counts.get(key, 0) + 1
    summary["count_nu_ic_family"] = dict(sorted(pair_counts.items()))

    summary["target_l2_norm_min"] = float(np.min(dataset["target_l2_norm"]))
    summary["target_l2_norm_max"] = float(np.max(dataset["target_l2_norm"]))
    summary["mean_energy0"] = float(np.mean(dataset["energy0"]))
    summary["max_abs_mass0"] = float(np.max(np.abs(dataset["mass0"])))
    summary["max_initial_cell_peclet"] = float(np.max(dataset["initial_cell_peclet"]))
    if "dataset_version" in dataset:
        summary["dataset_version"] = str(dataset["dataset_version"])
    return summary


def nu_sweep_dataset_sanity_checks(
    dataset: Dict[str, Any],
    nu_values: Optional[Sequence[float]] = None,
    target_norm_range: Optional[Sequence[float]] = None,
) -> Dict[str, bool]:
    """
    Sanity checks structurels d'un dataset nu_sweep_v1.

    Étend `basic_dataset_sanity_checks` avec les spécificités du protocole :
    présence des nouvelles clés, stratification par nu, round-robin des
    familles de CI dans chaque bloc, moyenne nulle, cohérence
    E(0) = 0.5 * target_l2_norm^2, et forme de `dissipation_physical`.
    """
    checks = dict(basic_dataset_sanity_checks(dataset))

    required_new_keys = [
        "target_l2_norm", "energy0", "mass0", "dissipation_physical",
        "trajectory_seeds", "initial_cell_peclet", "dataset_version", "protocol",
    ]
    checks["nu_sweep_keys_present"] = all(key in dataset for key in required_new_keys)

    states = np.asarray(dataset["states"], dtype=float)
    n_traj = states.shape[0]
    nu_arr = np.asarray(dataset["nu_values"], dtype=float)
    ic_ids = np.asarray(dataset["ic_type_ids"], dtype=int)

    if nu_values is not None:
        expected_nu = [float(v) for v in nu_values]
        per_nu = n_traj // len(expected_nu)
        strat_ok = all(
            bool(np.sum(nu_arr == nu) == per_nu) for nu in expected_nu
        )
        checks["nu_stratification_ok"] = strat_ok

        n_fam = int(np.asarray(dataset["ic_type_names"]).size)
        rr_ok = True
        for s, nu in enumerate(expected_nu):
            block = ic_ids[s * per_nu:(s + 1) * per_nu]
            rr_ok = rr_ok and bool(np.array_equal(block, np.arange(per_nu) % n_fam))
        checks["ic_round_robin_ok"] = rr_ok

        # Équilibrage des familles de CI à l'intérieur de chaque strate de nu :
        # les effectifs par couple (nu, famille) ne doivent différer que d'au
        # plus 1 (ex. 200 trajectoires / 3 familles -> 67, 67, 66 par strate).
        balance_ok = True
        for s, nu in enumerate(expected_nu):
            block_ids = ic_ids[s * per_nu:(s + 1) * per_nu]
            fam_counts = np.bincount(block_ids, minlength=n_fam)
            balance_ok = balance_ok and bool(np.max(fam_counts) - np.min(fam_counts) <= 1)
        checks["ic_balanced_within_nu_stratum"] = balance_ok

    target = np.asarray(dataset["target_l2_norm"], dtype=float)
    energy0 = np.asarray(dataset["energy0"], dtype=float)
    mass0 = np.asarray(dataset["mass0"], dtype=float)
    diss_phys = np.asarray(dataset["dissipation_physical"], dtype=float)
    peclet = np.asarray(dataset["initial_cell_peclet"], dtype=float)

    checks["target_l2_norm_finite"] = bool(np.all(np.isfinite(target)))
    checks["energy0_finite"] = bool(np.all(np.isfinite(energy0)))
    checks["mass0_near_zero"] = bool(np.max(np.abs(mass0)) < 1.0e-10)
    checks["dissipation_physical_shape_ok"] = bool(
        diss_phys.shape == np.asarray(dataset["dissipation"]).shape
    )
    checks["dissipation_physical_nonnegative"] = bool(np.all(diss_phys >= -1.0e-14))
    checks["energy0_matches_target"] = bool(
        np.allclose(energy0, 0.5 * target ** 2, rtol=1e-6, atol=1e-12)
    )
    checks["initial_cell_peclet_finite"] = bool(np.all(np.isfinite(peclet)))
    if target_norm_range is not None:
        lo, hi = float(target_norm_range[0]), float(target_norm_range[1])
        checks["target_l2_norm_in_range"] = bool(np.all((target >= lo) & (target <= hi)))

    return checks


# Clés du dataset nu_sweep_v1 dont la première dimension est l'indice de
# trajectoire : ce sont les seules permutées par `shuffle_nu_sweep_dataset`.
NU_SWEEP_PER_TRAJECTORY_KEYS = [
    "states", "energy", "dissipation", "u0", "nu_values", "ic_type_ids",
    "initial_max_abs_u", "initial_advective_cfl", "initial_diffusive_number",
    "energy_monotone_fraction", "energy_mean_positive_violation",
    "target_l2_norm", "energy0", "mass0", "dissipation_physical",
    "trajectory_seeds", "initial_cell_peclet",
]


def shuffle_nu_sweep_dataset(
    dataset: Dict[str, Any],
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Mélange les trajectoires d'un dataset nu_sweep_v1 de manière cohérente.

    Les splits nu_sweep_v1 sont stockés par blocs contigus de nu : dans le
    fichier brut, l'indice de ligne est donc un proxy du bloc de nu. Cette
    fonction applique UNE MÊME permutation à toutes les clés par trajectoire
    (voir NU_SWEEP_PER_TRAJECTORY_KEYS), ce qui détruit cette corrélation
    d'ordre tout en préservant la cohérence ligne par ligne (nu, CI, seeds,
    trajectoires). Les clés scalaires ou temporelles (times, x, dx,
    ic_type_names, dataset_version, protocol) sont conservées telles quelles.

    Retourne un nouveau dictionnaire ; le dataset d'entrée n'est pas modifié.
    """
    rng = _get_rng(rng)
    n_traj = int(np.asarray(dataset["states"]).shape[0])
    permutation = rng.permutation(n_traj)

    shuffled: Dict[str, Any] = {}
    for key, value in dataset.items():
        if key in NU_SWEEP_PER_TRAJECTORY_KEYS:
            shuffled[key] = np.asarray(value)[permutation]
        else:
            shuffled[key] = value
    return shuffled


def sample_target_norm(
    rng: np.random.Generator,
    target_norm_range: Sequence[float] = (0.5, 2.0),
) -> float:
    """
    Tire une norme L2 cible log-uniforme dans `target_norm_range`.

    Échantillonneur de `target_l2_norm` pour le protocole nu_sweep_v1 :
    avec [0.5, 2.0], E(0) = 0.5 * norme^2 tombe dans [0.125, 2.0].
    """
    return _sample_log_uniform(rng, float(target_norm_range[0]), float(target_norm_range[1]))


def build_stratified_assignments(
    n_trajectories: int,
    nu_values: Sequence[float],
    ic_types: Sequence[str],
) -> list[tuple[float, str]]:
    """
    Construit le plan d'assignation stratifié du protocole nu_sweep_v1.

    Retourne une liste de `n_trajectories` couples (nu, famille de CI) :
    - blocs contigus par valeur de nu (`n_trajectories` doit être divisible
      par `len(nu_values)` ; ex. 600 trajectoires et 3 nu -> 200 par nu) ;
    - familles de CI distribuées en round-robin à l'intérieur de chaque bloc
      de nu : `ic_types[j % len(ic_types)]` avec j l'index dans le bloc.
    """
    if n_trajectories < 1:
        raise ValueError("`n_trajectories` doit être >= 1.")
    nu_values = [float(v) for v in nu_values]
    if len(nu_values) == 0:
        raise ValueError("`nu_values` ne doit pas être vide.")
    if any(v <= 0.0 for v in nu_values):
        raise ValueError("Toutes les viscosités doivent être strictement positives.")
    if len(ic_types) == 0:
        raise ValueError("`ic_types` ne doit pas être vide.")
    if n_trajectories % len(nu_values) != 0:
        raise ValueError(
            f"Stratification impossible : {n_trajectories} trajectoires "
            f"non divisible par {len(nu_values)} viscosités."
        )

    per_nu = n_trajectories // len(nu_values)
    plan: list[tuple[float, str]] = []
    for nu in nu_values:
        for j in range(per_nu):
            plan.append((nu, str(ic_types[j % len(ic_types)])))
    return plan


def generate_dataset_split_nu_sweep(
    n_trajectories: int,
    nu_values: Sequence[float],
    x: np.ndarray,
    L: float,
    T: float,
    dt: float,
    save_every: int,
    time_scheme: str,
    ic_types: Sequence[str],
    target_norm_range: Sequence[float] = (0.5, 2.0),
    zero_mean: bool = True,
    rng: Optional[np.random.Generator] = None,
    ic_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Nom canonique du générateur de split nu_sweep_v1.

    Délègue à `generate_nu_sweep_split` (implémentation validée) : split
    stratifié par nu, familles de CI en round-robin par strate, norme L2
    cible log-uniforme, zero_mean pour toutes les familles, clés legacy +
    métadonnées nu_sweep_v1.
    """
    return generate_nu_sweep_split(
        n_trajectories=n_trajectories,
        nu_values=nu_values,
        x=x,
        L=L,
        T=T,
        dt=dt,
        save_every=save_every,
        time_scheme=time_scheme,
        ic_types=ic_types,
        target_norm_range=target_norm_range,
        zero_mean=zero_mean,
        rng=rng,
        ic_kwargs=ic_kwargs,
        verbose=verbose,
    )


def evaluate_nu_sweep_gates(
    dataset: Dict[str, Any],
    gate_monotone_frac: float = 0.99,
    gate_monotone_atol_rel: float = 1e-8,
    gate_pass_fraction: float = 0.95,
    gate_max_abs_u: float = 5.0,
    gate_dedt_rel_err: float = 1e-2,
    gate_max_cell_peclet: float = 2.0,
) -> Dict[str, Any]:
    """
    Évalue les portes de qualité numériques du protocole nu_sweep_v1.

    Portes bloquantes :
    - toutes les valeurs finies (pas de NaN/inf) ;
    - fraction de trajectoires dont l'énergie est non croissante (avec
      atol = `gate_monotone_atol_rel` * E(0) par trajectoire) sur au moins
      `gate_monotone_frac` des intervalles sauvegardés, pour au moins
      `gate_pass_fraction` des trajectoires ;
    - max |u| sur tous les snapshots <= `gate_max_abs_u` ;
    - médiane du résidu relatif du bilan d'énergie
      |dE/dt + nu*D| / (nu*D) <= `gate_dedt_rel_err`.

    Diagnostic (rapporté, non bloquant) : max du Péclet de maille initial
    `initial_cell_peclet` <= `gate_max_cell_peclet`.
    """
    states = np.asarray(dataset["states"], dtype=float)
    energy = np.asarray(dataset["energy"], dtype=float)
    times = np.asarray(dataset["times"], dtype=float)
    diss_phys = np.asarray(dataset["dissipation_physical"], dtype=float)
    energy0 = np.asarray(dataset["energy0"], dtype=float)
    peclet = np.asarray(dataset["initial_cell_peclet"], dtype=float)

    n_traj = int(states.shape[0])
    dt_saved = float(times[1] - times[0])

    # Monotonie de l'énergie, par trajectoire, avec atol relatif à E(0).
    diffs = np.diff(energy, axis=1)                       # (n_traj, n_times-1)
    atol = gate_monotone_atol_rel * energy0[:, None]       # (n_traj, 1)
    monotone_frac_per_traj = np.mean(diffs <= atol, axis=1)
    pass_fraction = float(np.mean(monotone_frac_per_traj >= gate_monotone_frac))

    # Bilan d'énergie : dE/dt + nu*D ~ 0 (nu*D = dissipation_physical).
    dedt = np.diff(energy, axis=1) / dt_saved
    nuD = diss_phys[:, :-1]
    mask = nuD > 1.0e-14
    if np.any(mask):
        rel_err = np.abs(dedt[mask] + nuD[mask]) / nuD[mask]
        median_dedt_rel_err = float(np.median(rel_err))
    else:
        median_dedt_rel_err = float("inf")

    max_abs_u = float(np.max(np.abs(states)))
    max_cell_peclet = float(np.max(peclet))

    checks = {
        "all_finite": bool(np.all(np.isfinite(states))),
        "monotone_pass_fraction_ok": bool(pass_fraction >= gate_pass_fraction),
        "max_abs_u_ok": bool(max_abs_u <= gate_max_abs_u),
        "dedt_rel_err_ok": bool(median_dedt_rel_err <= gate_dedt_rel_err),
    }
    return {
        "n_trajectories": n_traj,
        "dt_saved": dt_saved,
        "pass_fraction": pass_fraction,
        "gate_pass_fraction": gate_pass_fraction,
        "mean_monotone_fraction": float(np.mean(monotone_frac_per_traj)),
        "min_monotone_fraction": float(np.min(monotone_frac_per_traj)),
        "gate_monotone_frac": gate_monotone_frac,
        "gate_monotone_atol_rel": gate_monotone_atol_rel,
        "max_abs_u": max_abs_u,
        "gate_max_abs_u": gate_max_abs_u,
        "median_dedt_rel_err": median_dedt_rel_err,
        "gate_dedt_rel_err": gate_dedt_rel_err,
        "checks": checks,
        "gates_passed": bool(all(checks.values())),
        # Diagnostic non bloquant.
        "diagnostics": {
            "max_cell_peclet": max_cell_peclet,
            "gate_max_cell_peclet": gate_max_cell_peclet,
            "cell_peclet_ok": bool(max_cell_peclet <= gate_max_cell_peclet),
        },
    }


def run_nu_sweep_pilot(config: Any, verbose: bool = True) -> Dict[str, Any]:
    """
    Pilote de qualité / convergence du protocole nu_sweep_v1.

    Pour chaque niveau du ladder `config.nu_sweep.pilot_ladder`
    ([Nx, dt, save_every]) :
    - génère `pilot_n_per_nu` trajectoires par nu (split stratifié,
      round-robin des CI), EN MÉMOIRE UNIQUEMENT (rien n'est sauvegardé) ;
    - évalue les portes via `evaluate_nu_sweep_gates`.

    S'arrête au premier niveau passant toutes les portes ; sinon essaie le
    niveau suivant. La graine du niveau k est `pilot_seed + k` (reproductible).

    Retourne un rapport complet : métriques par niveau, niveau retenu,
    `overall_pass` (True si au moins un niveau passe les portes).
    """
    ns = config.nu_sweep
    nu_values = [float(v) for v in ns.nu_values]
    ic_types = list(config.dataset.ic_types)
    ic_kwargs = {
        "max_mode": int(config.dataset.max_fourier_mode),
        "amplitude_scale": float(config.dataset.fourier_amplitude_scale),
        "n_bumps": int(config.dataset.n_gaussian_bumps),
        "spectral_decay": float(config.dataset.spectral_decay),
        "zero_mean": bool(ns.zero_mean),
    }
    n_per_nu = int(ns.pilot_n_per_nu)
    n_traj = n_per_nu * len(nu_values)
    L = float(ns.L)
    T = float(ns.T)
    time_scheme = str(ns.time_scheme)
    target_norm_range = list(ns.target_norm_range)
    zero_mean = bool(ns.zero_mean)

    gate_kwargs = {
        "gate_monotone_frac": float(ns.gate_monotone_frac),
        "gate_monotone_atol_rel": float(ns.gate_monotone_atol_rel),
        "gate_pass_fraction": float(ns.gate_pass_fraction),
        "gate_max_abs_u": float(ns.gate_max_abs_u),
        "gate_dedt_rel_err": float(ns.gate_dedt_rel_err),
        "gate_max_cell_peclet": float(ns.gate_max_cell_peclet),
    }

    levels_report = []
    selected_level = None
    for level_idx, level in enumerate(ns.pilot_ladder):
        Nx_l = int(level[0])
        dt_l = float(level[1])
        save_every_l = int(level[2])
        x_l = np.linspace(0.0, L, Nx_l, endpoint=False)
        pilot_rng = np.random.default_rng(int(ns.pilot_seed) + level_idx)
        if verbose:
            print(
                f"[dataset_builder][nu_sweep][pilot] Niveau {level_idx + 1} : "
                f"Nx={Nx_l}, dt={dt_l}, save_every={save_every_l}, "
                f"{n_traj} trajectoires ({n_per_nu} par nu)."
            )
        ds = generate_dataset_split_nu_sweep(
            n_trajectories=n_traj,
            nu_values=nu_values,
            x=x_l,
            L=L,
            T=T,
            dt=dt_l,
            save_every=save_every_l,
            time_scheme=time_scheme,
            ic_types=ic_types,
            target_norm_range=target_norm_range,
            zero_mean=zero_mean,
            rng=pilot_rng,
            ic_kwargs=ic_kwargs,
            verbose=verbose,
        )
        gates = evaluate_nu_sweep_gates(ds, **gate_kwargs)

        # Diagnostic : énergie finale moyenne par nu (ordre attendu
        # E_final(nu=0.20) < E_final(nu=0.05) < E_final(nu=0.01)).
        nu_arr = np.asarray(ds["nu_values"], dtype=float)
        mean_final_energy_by_nu = {
            f"nu={nu:g}": float(np.mean(ds["energy"][nu_arr == nu, -1]))
            for nu in nu_values
        }

        levels_report.append({
            "level": level_idx + 1,
            "Nx": Nx_l,
            "dt": dt_l,
            "save_every": save_every_l,
            "n_trajectories": n_traj,
            "gates": gates,
            "mean_final_energy_by_nu": mean_final_energy_by_nu,
        })
        if verbose:
            print(
                f"[dataset_builder][nu_sweep][pilot] Niveau {level_idx + 1} : "
                f"gates_passed={gates['gates_passed']} "
                f"(pass_fraction={gates['pass_fraction']:.3f}, "
                f"max|u|={gates['max_abs_u']:.3f}, "
                f"dedt_rel_err={gates['median_dedt_rel_err']:.2e}, "
                f"Pe_cell={gates['diagnostics']['max_cell_peclet']:.2f})"
            )
        if gates["gates_passed"]:
            selected_level = level_idx + 1
            break

    return {
        "pilot_seed": int(ns.pilot_seed),
        "pilot_n_per_nu": n_per_nu,
        "nu_values": nu_values,
        "levels": levels_report,
        "selected_level": selected_level,
        "overall_pass": selected_level is not None,
    }


def summarize_dataset_nu_sweep(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nom canonique du résumé nu_sweep_v1.

    Délègue à `summarize_nu_sweep_dataset` : résumé legacy + distribution des
    nu, table `count_nu_ic_family`, normes cibles, E(0), masse, Péclet.
    """
    return summarize_nu_sweep_dataset(dataset)


def build_and_save_split_from_config_nu_sweep(
    config: Any,
    split: str,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Nom canonique du pipeline nu_sweep_v1 : génère un split depuis
    `config.nu_sweep` et le sauvegarde dans
    `outputs/data/{split}_data_nu_sweep.npz`.

    Délègue à `build_and_save_nu_sweep_split_from_config`.
    """
    return build_and_save_nu_sweep_split_from_config(
        config=config,
        split=split,
        rng=rng,
        verbose=verbose,
    )


# =============================================================================
# F. BLOC DE TEST MINIMAL
# =============================================================================

if __name__ == "__main__":
    L = 2.0 * np.pi
    Nx = 64
    T = 0.5
    dt = 1.0e-3
    nu = 0.05
    save_every = 20
    x = np.linspace(0.0, L, Nx, endpoint=False)
    rng = np.random.default_rng(123)

    dataset = generate_dataset_split(
        n_trajectories=5,
        x=x,
        L=L,
        T=T,
        dt=dt,
        nu=nu,
        save_every=save_every,
        time_scheme="rk4",
        ic_types=["fourier", "gaussian_bumps", "random_smooth"],
        normalize_ic=False,
        target_ic_norm=1.0,
        rng=rng,
        ic_kwargs={
            "max_mode": 4,
            "amplitude_scale": 1.0,
            "n_bumps": 2,
            "spectral_decay": 2.0,
        },
        verbose=True,
    )

    print("\nRésumé du dataset de test :")
    summary = summarize_dataset(dataset)
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print("\nSanity checks :")
    checks = basic_dataset_sanity_checks(dataset)
    for key, value in checks.items():
        print(f"- {key}: {value}")

    # --- Smoke test nu_sweep_v1 (configuration réduite, en mémoire uniquement) ---
    print("\nSmoke test nu_sweep_v1 (configuration réduite, rien n'est sauvegardé) :")
    ns_dataset = generate_nu_sweep_split(
        n_trajectories=6,
        nu_values=[0.01, 0.05, 0.20],
        x=np.linspace(0.0, L, 64, endpoint=False),
        L=L,
        T=0.05,
        dt=1.0e-3,
        save_every=10,
        time_scheme="rk4",
        ic_types=["fourier", "gaussian_bumps", "random_smooth"],
        target_norm_range=[0.5, 2.0],
        zero_mean=True,
        rng=np.random.default_rng(2024),
        ic_kwargs={
            "max_mode": 4,
            "amplitude_scale": 1.0,
            "n_bumps": 2,
            "spectral_decay": 2.0,
            "zero_mean": True,
        },
        verbose=False,
    )
    ns_dataset["dataset_version"] = "nu_sweep_v1"
    ns_dataset["protocol"] = "smoke-test"
    ns_summary = summarize_nu_sweep_dataset(ns_dataset)
    for key, value in ns_summary.items():
        print(f"- {key}: {value}")
    ns_checks = nu_sweep_dataset_sanity_checks(
        ns_dataset,
        nu_values=[0.01, 0.05, 0.20],
        target_norm_range=[0.5, 2.0],
    )
    for key, value in ns_checks.items():
        print(f"- {key}: {value}")

    # --- Smoke test du wrapper config (objet duck-typé, rien n'est sauvegardé) ---
    from types import SimpleNamespace
    ns_cfg = SimpleNamespace(
        nu_sweep=SimpleNamespace(
            n_train=6,
            n_val=6,
            n_test=6,
            train_filename="train_data_nu_sweep.npz",
            val_filename="val_data_nu_sweep.npz",
            test_filename="test_data_nu_sweep.npz",
            nu_values=[0.01, 0.05, 0.20],
            target_norm_range=[0.5, 2.0],
            zero_mean=True,
            Nx=64,
            L=L,
            T=0.05,
            dt=1.0e-3,
            save_every=10,
            time_scheme="rk4",
            dataset_version="nu_sweep_v1",
            protocol="smoke-test",
            pilot_n_per_nu=2,
            pilot_seed=1234,
            pilot_ladder=[[64, 1.0e-3, 10]],
            gate_monotone_frac=0.99,
            gate_monotone_atol_rel=1e-8,
            gate_pass_fraction=0.95,
            gate_max_abs_u=5.0,
            gate_dedt_rel_err=1e-2,
            gate_max_cell_peclet=2.0,
        ),
        dataset=SimpleNamespace(
            ic_types=["fourier", "gaussian_bumps", "random_smooth"],
            max_fourier_mode=4,
            fourier_amplitude_scale=1.0,
            n_gaussian_bumps=2,
            spectral_decay=2.0,
        ),
        paths=SimpleNamespace(data_dir=Path("outputs") / "data"),
    )
    ns_ds_cfg, ns_out = build_nu_sweep_dataset_from_config(
        ns_cfg, split="train", rng=np.random.default_rng(7), verbose=False
    )
    print(f"- wrapper config -> sortie prévue : {ns_out}")
    print(f"- states : {ns_ds_cfg['states'].shape}, version : {ns_ds_cfg['dataset_version']}")
    ns_shuffled = shuffle_nu_sweep_dataset(ns_ds_cfg, rng=np.random.default_rng(0))
    print("- après shuffle, nu_values :", ns_shuffled["nu_values"])
    print("- après shuffle, ic_type_ids :", ns_shuffled["ic_type_ids"])
    ns_gates = evaluate_nu_sweep_gates(ns_dataset)
    print("- gates (smoke) :", ns_gates["gates_passed"], ns_gates["checks"])
    ns_pilot = run_nu_sweep_pilot(ns_cfg, verbose=False)
    print("- pilot (smoke) : overall_pass =", ns_pilot["overall_pass"],
          "| niveaux testés =", len(ns_pilot["levels"]),
          "| niveau retenu =", ns_pilot["selected_level"])
