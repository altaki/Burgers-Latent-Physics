"""
ml/datasets_experiment_c.py
===========================

Jeux de données PyTorch pour l'expérience C : baseline supervisée de
séparabilité des observables physiques sur le dataset `nu_sweep_v1`.

Objectif
--------
L'expérience C mesure dans quelle mesure cinq observables physiques sont
prédictibles à partir des états de Burgers :

1. énergie              E(u)  = 1/2 ∫ u^2 dx
2. dissipation physique nu*D = nu ∫ u_x^2 dx
3. enstrophie           D(u)  = ∫ u_x^2 dx
4. variation totale     TV(u) = ∫ |u_x| dx
5. max |u_x|

Deux variantes d'entrée sont prévues :

- snapshot unique   (modèles C1 / C2) : u_t                       -> cibles à t
- fenêtre temporelle (modèle C3)       : (u_{t-w+1}, ..., u_t)     -> cibles à t

Le module est additif et autonome : il ne modifie aucun script existant et
réutilise au maximum les utilitaires du dépôt :

- `ml_datasets.load_dataset_npz` / `validate_dataset_dict` /
  `GlobalStandardizer` / `fit_global_standardizer` / `to_float_tensor`
- `pde_observables.compute_experiment_c_targets` (TV, max|u_x|, nu*D,
  calcul vectorisé par différences finies centrées périodiques)

Protocole prévu
---------------
- split standard   : train sur les 3 strates de nu, val / test complets ;
- protocole held-out-nu : `nu_filter={0.05, 0.20}` sur train et val,
  évaluation sur la strate nu=0.01 du test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib.util

import numpy as np
import torch
from torch.utils.data import Dataset


# =============================================================================
# 1. Chargement flexible des modules du projet (même pattern que ml_train_supervised)
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


def _load_datasets_module():
    """Charge `ml_datasets` (racine ou package `ml/`)."""
    return _load_module_from_candidates(
        module_name='ml_datasets',
        candidates=['ml_datasets.py', 'ml/datasets.py'],
    )


def _load_observables_module():
    """Charge `pde_observables` (racine ou package `pde/`)."""
    return _load_module_from_candidates(
        module_name='pde_observables',
        candidates=['pde_observables.py', 'pde/observables.py'],
    )


# =============================================================================
# 2. Constantes et standardisation des cibles
# =============================================================================

#: Noms des cinq cibles supervisées de l'expérience C, dans l'ordre du
#: vecteur `targets` renvoyé par les datasets.
EXPERIMENT_C_TARGET_NAMES: Tuple[str, ...] = (
    "energy",
    "nu_dissipation",
    "dissipation",
    "tv",
    "max_abs_ux",
)


@dataclass
class TargetStandardizer:
    """
    Standardisation par cible :

        target_normalisé = (target - mean) / std

    où `means` et `stds` sont des vecteurs de longueur n_targets, ajustés
    sur les cibles du split train uniquement.
    """

    means: np.ndarray
    stds: np.ndarray
    eps: float = 1.0e-8
    target_names: Tuple[str, ...] = EXPERIMENT_C_TARGET_NAMES

    def transform(self, array: np.ndarray) -> np.ndarray:
        """Applique la normalisation à un tableau (..., n_targets)."""
        return (array - self.means) / np.maximum(self.stds, self.eps)

    def inverse_transform(self, array: np.ndarray) -> np.ndarray:
        """Revient à l'échelle physique initiale."""
        return array * np.maximum(self.stds, self.eps) + self.means

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire sérialisable."""
        return {
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "eps": float(self.eps),
            "target_names": list(self.target_names),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetStandardizer":
        """Reconstruit un standardizer depuis un dictionnaire."""
        return cls(
            means=np.asarray(data["means"], dtype=float),
            stds=np.asarray(data["stds"], dtype=float),
            eps=float(data.get("eps", 1.0e-8)),
            target_names=tuple(data.get("target_names", EXPERIMENT_C_TARGET_NAMES)),
        )


def fit_target_standardizer(targets: np.ndarray, eps: float = 1.0e-8) -> TargetStandardizer:
    """
    Ajuste un `TargetStandardizer` sur un tableau de cibles de forme
    (n_samples, n_targets).
    """
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != len(EXPERIMENT_C_TARGET_NAMES):
        raise ValueError(
            f"`targets` doit être de forme (n_samples, {len(EXPERIMENT_C_TARGET_NAMES)})."
        )
    return TargetStandardizer(
        means=np.mean(targets, axis=0),
        stds=np.std(targets, axis=0),
        eps=eps,
    )


# =============================================================================
# 3. Calcul des cibles depuis un dictionnaire de dataset nu_sweep_v1
# =============================================================================

def compute_experiment_c_targets_from_dataset(
    dataset: Dict[str, Any],
    check_consistency: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Calcule les cinq cibles de l'expérience C pour un dataset nu_sweep_v1.

    Les cibles E, D et nu*D sont lues dans les clés stockées du npz
    (`energy`, `dissipation`, `dissipation_physical`) lorsqu'elles existent ;
    TV et max|u_x| sont calculées depuis `states` via `pde_observables`.

    Paramètres
    ----------
    dataset : Dict[str, Any]
        Dictionnaire du dataset (déjà chargé).
    check_consistency : bool
        Si True, compare les cibles recalculées depuis `states` aux valeurs
        stockées et retourne les écarts maximaux dans le second élément.

    Retour
    ------
    tuple
        (targets, consistency) où `targets` est un dict de tableaux
        (n_traj, n_times) indexés par `EXPERIMENT_C_TARGET_NAMES`, et
        `consistency` un dict de flottants (écarts max vs stocké).
    """
    datasets_module = _load_datasets_module()
    observables_module = _load_observables_module()

    dataset = datasets_module.validate_dataset_dict(dataset)

    states = np.asarray(dataset["states"], dtype=float)
    dx = float(np.asarray(dataset["dx"]).item())
    nu_values = (
        np.asarray(dataset["nu_values"], dtype=float)
        if "nu_values" in dataset
        else None
    )

    computed = observables_module.compute_experiment_c_targets(states, dx, nu_values)

    targets: Dict[str, np.ndarray] = {}
    consistency: Dict[str, float] = {}

    stored_keys = {
        "energy": "energy",
        "dissipation": "dissipation",
        "nu_dissipation": "dissipation_physical",
    }
    for name in EXPERIMENT_C_TARGET_NAMES:
        if name in stored_keys and stored_keys[name] in dataset:
            stored = np.asarray(dataset[stored_keys[name]], dtype=float)
            if check_consistency:
                consistency[f"{name}_max_abs_err_vs_stored"] = float(
                    np.max(np.abs(computed[name] - stored))
                )
            targets[name] = stored
        else:
            targets[name] = computed[name]

    return targets, consistency


def _build_nu_mask(nu_values: np.ndarray, nu_filter: Optional[Sequence[float]]) -> np.ndarray:
    """
    Construit un masque booléen sur les trajectoires : True si la viscosité
    de la trajectoire appartient à `nu_filter` (None = toutes les strates).
    """
    if nu_filter is None:
        return np.ones(nu_values.shape[0], dtype=bool)
    allowed = np.asarray(list(nu_filter), dtype=float)
    return np.any(np.isclose(nu_values[:, None], allowed[None, :]), axis=1)


# =============================================================================
# 4. Dataset de snapshots uniques (modèles C1 / C2)
# =============================================================================

class NuSweepSnapshotDataset(Dataset):
    """
    Dataset où chaque item est un snapshot individuel (u_t) avec le vecteur
    des cinq cibles à l'instant t.

    Chaque item renvoie :
    - u                : tenseur (Nx,)
    - targets          : tenseur (5,) dans l'ordre `EXPERIMENT_C_TARGET_NAMES`
    - nu, time, trajectory_index, time_index, energy0  (métadonnées)
    """

    def __init__(
        self,
        dataset: Dict[str, Any] | str | Path,
        normalize: bool = False,
        standardizer: Optional[Any] = None,
        normalize_targets: bool = True,
        target_standardizer: Optional[TargetStandardizer] = None,
        nu_filter: Optional[Sequence[float]] = None,
        return_metadata: bool = True,
    ) -> None:
        """
        Paramètres
        ----------
        dataset : dict ou chemin `.npz`
            Dataset nu_sweep_v1 déjà chargé ou chemin vers le fichier.
        normalize : bool
            Si True, standardise les snapshots (standardizer du train requis
            pour val/test).
        standardizer : GlobalStandardizer ou None
            Standardizer des états, ajusté sur le train.
        normalize_targets : bool
            Si True, standardise les cibles (target_standardizer du train
            requis pour val/test).
        target_standardizer : TargetStandardizer ou None
            Standardizer des cibles, ajusté sur le train.
        nu_filter : séquence de flottants ou None
            Si fourni, seules les trajectoires dont nu appartient à cette
            liste sont conservées (protocole held-out-nu).
        return_metadata : bool
            Si True, inclut nu / time / indices / energy0 dans chaque item.
        """
        self._datasets_module = _load_datasets_module()
        datasets_module = self._datasets_module

        if isinstance(dataset, (str, Path)):
            dataset = datasets_module.load_dataset_npz(dataset)
        dataset = datasets_module.validate_dataset_dict(dataset)

        self.dataset = dataset
        self.states = np.asarray(dataset["states"], dtype=np.float32)
        self.times = np.asarray(dataset["times"], dtype=np.float32)
        self.n_traj, self.n_times, self.nx = self.states.shape
        self.dx = float(np.asarray(dataset["dx"]).item())

        targets, self.target_consistency = compute_experiment_c_targets_from_dataset(dataset)
        self.targets = np.stack(
            [targets[name] for name in EXPERIMENT_C_TARGET_NAMES], axis=-1
        ).astype(np.float32)  # (n_traj, n_times, 5)

        self.nu_values = (
            np.asarray(dataset["nu_values"], dtype=np.float32)
            if "nu_values" in dataset
            else np.zeros(self.n_traj, dtype=np.float32)
        )
        self.energy0 = (
            np.asarray(dataset["energy0"], dtype=np.float32)
            if "energy0" in dataset
            else np.asarray(dataset["energy"][:, 0], dtype=np.float32)
        )

        self.normalize = normalize
        if normalize:
            self.standardizer = standardizer or datasets_module.fit_global_standardizer(self.states)
            self.states = self.standardizer.transform(self.states).astype(np.float32)
        else:
            self.standardizer = standardizer

        self.normalize_targets = normalize_targets
        if normalize_targets:
            self.target_standardizer = target_standardizer or fit_target_standardizer(
                self.targets.reshape(-1, len(EXPERIMENT_C_TARGET_NAMES))
            )
            self.targets = self.target_standardizer.transform(self.targets).astype(np.float32)
        else:
            self.target_standardizer = target_standardizer

        self.nu_filter = None if nu_filter is None else tuple(float(v) for v in nu_filter)
        self.nu_mask = _build_nu_mask(self.nu_values, self.nu_filter)
        self.index_map = [
            (i, t)
            for i in range(self.n_traj)
            for t in range(self.n_times)
            if self.nu_mask[i]
        ]
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, time_idx = self.index_map[idx]

        item: Dict[str, Any] = {
            "u": self._datasets_module.to_float_tensor(self.states[traj_idx, time_idx]),
            "targets": self._datasets_module.to_float_tensor(self.targets[traj_idx, time_idx]),
        }

        if self.return_metadata:
            item["nu"] = float(self.nu_values[traj_idx])
            item["time"] = float(self.times[time_idx])
            item["trajectory_index"] = int(traj_idx)
            item["time_index"] = int(time_idx)
            item["energy0"] = float(self.energy0[traj_idx])

        return item


# =============================================================================
# 5. Dataset de fenêtres temporelles (modèle C3)
# =============================================================================

class NuSweepWindowDataset(Dataset):
    """
    Dataset où chaque item est une fenêtre de `window_size` snapshots
    consécutifs (u_{t-w+1}, ..., u_t), empilés en canaux, avec le vecteur
    des cinq cibles à l'instant final t.

    Chaque item renvoie :
    - window           : tenseur (window_size, Nx)
    - targets          : tenseur (5,)
    - nu, time, trajectory_index, time_index, energy0  (métadonnées)
    """

    def __init__(
        self,
        dataset: Dict[str, Any] | str | Path,
        window_size: int = 5,
        normalize: bool = False,
        standardizer: Optional[Any] = None,
        normalize_targets: bool = True,
        target_standardizer: Optional[TargetStandardizer] = None,
        nu_filter: Optional[Sequence[float]] = None,
        return_metadata: bool = True,
    ) -> None:
        self._datasets_module = _load_datasets_module()
        datasets_module = self._datasets_module

        if isinstance(dataset, (str, Path)):
            dataset = datasets_module.load_dataset_npz(dataset)
        dataset = datasets_module.validate_dataset_dict(dataset)

        self.dataset = dataset
        self.states = np.asarray(dataset["states"], dtype=np.float32)
        self.times = np.asarray(dataset["times"], dtype=np.float32)
        self.n_traj, self.n_times, self.nx = self.states.shape
        self.dx = float(np.asarray(dataset["dx"]).item())

        if window_size < 1:
            raise ValueError("`window_size` doit être >= 1.")
        if window_size > self.n_times:
            raise ValueError("`window_size` doit être <= n_times.")
        self.window_size = int(window_size)

        targets, self.target_consistency = compute_experiment_c_targets_from_dataset(dataset)
        self.targets = np.stack(
            [targets[name] for name in EXPERIMENT_C_TARGET_NAMES], axis=-1
        ).astype(np.float32)

        self.nu_values = (
            np.asarray(dataset["nu_values"], dtype=np.float32)
            if "nu_values" in dataset
            else np.zeros(self.n_traj, dtype=np.float32)
        )
        self.energy0 = (
            np.asarray(dataset["energy0"], dtype=np.float32)
            if "energy0" in dataset
            else np.asarray(dataset["energy"][:, 0], dtype=np.float32)
        )

        self.normalize = normalize
        if normalize:
            self.standardizer = standardizer or datasets_module.fit_global_standardizer(self.states)
            self.states = self.standardizer.transform(self.states).astype(np.float32)
        else:
            self.standardizer = standardizer

        self.normalize_targets = normalize_targets
        if normalize_targets:
            self.target_standardizer = target_standardizer or fit_target_standardizer(
                self.targets.reshape(-1, len(EXPERIMENT_C_TARGET_NAMES))
            )
            self.targets = self.target_standardizer.transform(self.targets).astype(np.float32)
        else:
            self.target_standardizer = target_standardizer

        self.nu_filter = None if nu_filter is None else tuple(float(v) for v in nu_filter)
        self.nu_mask = _build_nu_mask(self.nu_values, self.nu_filter)
        self.index_map = [
            (i, t)
            for i in range(self.n_traj)
            for t in range(self.window_size - 1, self.n_times)
            if self.nu_mask[i]
        ]
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, time_idx = self.index_map[idx]
        t0 = time_idx - self.window_size + 1

        item: Dict[str, Any] = {
            "window": self._datasets_module.to_float_tensor(self.states[traj_idx, t0:time_idx + 1]),
            "targets": self._datasets_module.to_float_tensor(self.targets[traj_idx, time_idx]),
        }

        if self.return_metadata:
            item["nu"] = float(self.nu_values[traj_idx])
            item["time"] = float(self.times[time_idx])
            item["trajectory_index"] = int(traj_idx)
            item["time_index"] = int(time_idx)
            item["energy0"] = float(self.energy0[traj_idx])

        return item


# =============================================================================
# 5b. Dataset de paires de fenêtres temporelles (window-A / A-Sequence)
# =============================================================================

class NuSweepWindowPairDataset(Dataset):
    """
    Paires de fenêtres temporelles consécutives (stride configurable, 1 par
    défaut) pour window-A (Architecture A-Sequence, Step 5).

    Chaque item renvoie :
    - window_t  : tenseur (window_size, Nx) — fenêtre finissant à t
    - window_tp : tenseur (window_size, Nx) — fenêtre finissant à t+stride
    - u_t       : tenseur (Nx,) — snapshot FINAL de window_t (cible de
                  reconstruction du décodeur)
    - u_tp      : tenseur (Nx,) — snapshot final de window_tp
    - métadonnées : nu, time_t, time_tp, trajectory_index, time_index_t,
      time_index_tp, energy_t, dissipation_t, nu_dissipation_t, energy0,
      nuD_est_t (νD estimé par différence centrée de l'énergie stockée :
      (E_{t-1} - E_{t+1}) / (2*dt) = -dE/dt, cible auto-supervisée de
      l'Experiment A-Physics)
    """

    def __init__(
        self,
        dataset: Dict[str, Any] | str | Path,
        window_size: int = 25,
        stride: int = 1,
        normalize: bool = True,
        standardizer: Optional[Any] = None,
        nu_filter: Optional[Sequence[float]] = None,
        return_metadata: bool = True,
    ) -> None:
        self._datasets_module = _load_datasets_module()
        datasets_module = self._datasets_module

        if isinstance(dataset, (str, Path)):
            dataset = datasets_module.load_dataset_npz(dataset)
        dataset = datasets_module.validate_dataset_dict(dataset)

        self.dataset = dataset
        self.states = np.asarray(dataset["states"], dtype=np.float32)
        self.times = np.asarray(dataset["times"], dtype=np.float32)
        self.n_traj, self.n_times, self.nx = self.states.shape
        self.dx = float(np.asarray(dataset["dx"]).item())
        self.dt = float(self.times[1] - self.times[0]) if self.n_times > 1 else 1.0

        if window_size < 1:
            raise ValueError("`window_size` doit être >= 1.")
        if window_size > self.n_times:
            raise ValueError("`window_size` doit être <= n_times.")
        if stride < 1:
            raise ValueError("`stride` doit être >= 1.")
        if window_size + stride > self.n_times:
            raise ValueError("`window_size + stride` doit être <= n_times.")
        self.window_size = int(window_size)
        self.stride = int(stride)

        self.energy = np.asarray(dataset["energy"], dtype=np.float32)
        self.dissipation = np.asarray(dataset["dissipation"], dtype=np.float32)
        if "dissipation_physical" in dataset:
            self.nu_dissipation = np.asarray(dataset["dissipation_physical"], dtype=np.float32)
        else:
            nu = (
                np.asarray(dataset["nu_values"], dtype=np.float32)
                if "nu_values" in dataset
                else np.zeros(self.n_traj, dtype=np.float32)
            )
            self.nu_dissipation = (nu[:, None] * self.dissipation).astype(np.float32)

        self.nu_values = (
            np.asarray(dataset["nu_values"], dtype=np.float32)
            if "nu_values" in dataset
            else np.zeros(self.n_traj, dtype=np.float32)
        )
        self.energy0 = (
            np.asarray(dataset["energy0"], dtype=np.float32)
            if "energy0" in dataset
            else np.asarray(dataset["energy"][:, 0], dtype=np.float32)
        )

        self.normalize = normalize
        if normalize:
            self.standardizer = standardizer or datasets_module.fit_global_standardizer(self.states)
            self.states = self.standardizer.transform(self.states).astype(np.float32)
        else:
            self.standardizer = standardizer

        self.ic_type_ids = (
            np.asarray(dataset["ic_type_ids"], dtype=np.int64) if "ic_type_ids" in dataset else None
        )
        self.nu_filter = None if nu_filter is None else tuple(float(v) for v in nu_filter)
        self.nu_mask = _build_nu_mask(self.nu_values, self.nu_filter)
        # paires (i, t) : fenêtre [t-w+1, t] et fenêtre suivante [t-w+1+stride, t+stride]
        self.index_map = [
            (i, t)
            for i in range(self.n_traj)
            for t in range(self.window_size - 1, self.n_times - self.stride)
            if self.nu_mask[i]
        ]
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, time_idx = self.index_map[idx]
        next_idx = time_idx + self.stride
        t0 = time_idx - self.window_size + 1
        t0p = next_idx - self.window_size + 1

        item: Dict[str, Any] = {
            "window_t": self._datasets_module.to_float_tensor(self.states[traj_idx, t0:time_idx + 1]),
            "window_tp": self._datasets_module.to_float_tensor(self.states[traj_idx, t0p:next_idx + 1]),
            "u_t": self._datasets_module.to_float_tensor(self.states[traj_idx, time_idx]),
            "u_tp": self._datasets_module.to_float_tensor(self.states[traj_idx, next_idx]),
        }

        if self.return_metadata:
            item["nu"] = float(self.nu_values[traj_idx])
            item["time_t"] = float(self.times[time_idx])
            item["time_tp"] = float(self.times[next_idx])
            item["trajectory_index"] = int(traj_idx)
            item["time_index_t"] = int(time_idx)
            item["time_index_tp"] = int(next_idx)
            item["energy_t"] = float(self.energy[traj_idx, time_idx])
            item["dissipation_t"] = float(self.dissipation[traj_idx, time_idx])
            item["nu_dissipation_t"] = float(self.nu_dissipation[traj_idx, time_idx])
            item["energy0"] = float(self.energy0[traj_idx])
            item["ic_type_id"] = int(self.ic_type_ids[traj_idx]) if self.ic_type_ids is not None else -1
            # νD estimé par différence centrée de l'énergie (identité dE/dt = -νD) :
            # (E_{t-1} - E_{t+1}) / (2*dt) ; repli décentré au bord.
            if time_idx >= 1:
                nuD_est = (self.energy[traj_idx, time_idx - 1] - self.energy[traj_idx, next_idx]) / (2.0 * self.dt)
            else:
                nuD_est = (self.energy[traj_idx, time_idx] - self.energy[traj_idx, next_idx]) / self.dt
            item["nuD_est_t"] = float(nuD_est)

        return item


# =============================================================================
# 6. Constructeur des splits train / val / test
# =============================================================================

def build_experiment_c_datasets(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    normalize: bool = True,
    normalize_targets: bool = True,
    window_size: int = 5,
    nu_filter_train: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Charge les trois splits nu_sweep_v1 et construit les datasets snapshot
    et fenêtre.

    Les standardizers (états et cibles) sont ajustés sur le split train
    uniquement, puis réutilisés tels quels pour val et test.

    Paramètres
    ----------
    train_path, val_path, test_path : str ou Path
        Chemins des fichiers `*_data_nu_sweep.npz`.
    normalize : bool
        Standardisation globale des états (fit sur train).
    normalize_targets : bool
        Standardisation par cible (fit sur train).
    window_size : int
        Taille de la fenêtre temporelle pour le dataset C3.
    nu_filter_train : séquence de flottants ou None
        Si fourni (ex. {0.05, 0.20}), restreint train ET val aux trajectoires
        de ces strates (protocole held-out-nu). Le test reste complet pour
        permettre l'évaluation par strate de nu.

    Retour
    ------
    Dict[str, Any]
        - "train" / "val" / "test" : dict {"snapshot": ..., "window": ...}
        - "input_standardizer", "target_standardizer", "window_size"
    """
    datasets_module = _load_datasets_module()

    train_raw = datasets_module.load_dataset_npz(train_path)
    val_raw = datasets_module.load_dataset_npz(val_path)
    test_raw = datasets_module.load_dataset_npz(test_path)

    input_standardizer = None
    if normalize:
        input_standardizer = datasets_module.fit_global_standardizer(
            np.asarray(train_raw["states"], dtype=np.float32)
        )

    target_standardizer = None
    if normalize_targets:
        train_targets, _ = compute_experiment_c_targets_from_dataset(train_raw)
        train_targets_arr = np.stack(
            [train_targets[name] for name in EXPERIMENT_C_TARGET_NAMES], axis=-1
        )
        target_standardizer = fit_target_standardizer(train_targets_arr.reshape(-1, len(EXPERIMENT_C_TARGET_NAMES)))

    def _make(split_raw: Dict[str, Any], nu_filter: Optional[Sequence[float]]) -> Dict[str, Any]:
        snapshot = NuSweepSnapshotDataset(
            split_raw,
            normalize=normalize,
            standardizer=input_standardizer,
            normalize_targets=normalize_targets,
            target_standardizer=target_standardizer,
            nu_filter=nu_filter,
        )
        window = NuSweepWindowDataset(
            split_raw,
            window_size=window_size,
            normalize=normalize,
            standardizer=input_standardizer,
            normalize_targets=normalize_targets,
            target_standardizer=target_standardizer,
            nu_filter=nu_filter,
        )
        return {"snapshot": snapshot, "window": window}

    return {
        "train": _make(train_raw, nu_filter_train),
        "val": _make(val_raw, nu_filter_train),
        "test": _make(test_raw, None),
        "input_standardizer": input_standardizer,
        "target_standardizer": target_standardizer,
        "window_size": int(window_size),
    }


# =============================================================================
# 7. Bloc de test minimal (dataset synthétique en mémoire)
# =============================================================================

if __name__ == "__main__":
    rng = np.random.default_rng(123)
    n_traj, n_times, nx = 6, 12, 32
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, nx, endpoint=False)
    dx = float(x[1] - x[0])
    times = np.linspace(0.0, 1.0, n_times)

    states = rng.normal(size=(n_traj, n_times, nx)).astype(np.float32)
    energy = (0.5 * np.sum(states ** 2, axis=2) * dx).astype(np.float32)
    nu_values = np.array([0.01, 0.05, 0.2, 0.01, 0.05, 0.2], dtype=np.float32)
    ux = (np.roll(states, -1, axis=2) - np.roll(states, 1, axis=2)) / (2.0 * dx)
    dissipation = (np.sum(ux ** 2, axis=2) * dx).astype(np.float32)

    fake = {
        "states": states,
        "times": times,
        "energy": energy,
        "dissipation": dissipation,
        "dissipation_physical": (nu_values[:, None] * dissipation).astype(np.float32),
        "x": x,
        "dx": np.array(dx, dtype=float),
        "nu_values": nu_values,
        "energy0": energy[:, 0].copy(),
        "ic_type_ids": np.arange(n_traj) % 3,
        "ic_type_names": np.array(["fourier", "gaussian_bumps", "random_smooth"]),
    }

    snap = NuSweepSnapshotDataset(fake, normalize=True, normalize_targets=True)
    win = NuSweepWindowDataset(fake, window_size=5, normalize=True, normalize_targets=True,
                               standardizer=snap.standardizer, target_standardizer=snap.target_standardizer)
    snap_ho = NuSweepSnapshotDataset(fake, nu_filter=(0.05, 0.2))

    print("Tailles :")
    print("- snapshot complet      :", len(snap))
    print("- fenêtre (w=5)         :", len(win))
    print("- snapshot nu in {05,2} :", len(snap_ho))

    item_s = snap[0]
    item_w = win[0]
    print("\nClés item snapshot :", list(item_s.keys()))
    print("Formes : u", tuple(item_s["u"].shape), "| targets", tuple(item_s["targets"].shape))
    print("Clés item fenêtre  :", list(item_w.keys()))
    print("Formes : window", tuple(item_w["window"].shape), "| targets", tuple(item_w["targets"].shape))
    print("\nConsistance cibles vs stocké :", snap.target_consistency)
    print("Standardizer cibles :", snap.target_standardizer.to_dict()["target_names"])
