"""
ml/train_experiment_c.py
========================

Moteur d'entraînement et d'évaluation pour l'expérience C : baseline
supervisée multi-tâches de séparabilité des observables physiques sur le
dataset `nu_sweep_v1`.

Schéma général
--------------
1. construire les datasets / DataLoaders via `ml_datasets_experiment_c`,
2. construire un modèle C1 / C2 / C3 via `ml_models_experiment_c`,
3. entraîner en MSE multi-tâches standardisée (Adam, early stopping,
   sauvegarde du meilleur checkpoint, format identique à ml_train_supervised),
4. prédire sur le split test et calculer les métriques par sous-ensemble :
   global, par strate de nu, par temps, par quartile de E(0), agrégé par
   trajectoire,
5. comparer aux baselines triviales (moyenne globale / moyenne par nu),
6. calculer les diagnostics de séparabilité (têtes D vs nu*D),
7. sauvegarder métriques JSON + CSV long format.

Protocoles
----------
- "standard"    : entraînement sur les 3 strates de nu ;
- "heldout_nu"  : entraînement restreint aux strates `heldout_nu_values`
                  (ex. {0.05, 0.20}), évaluation sur la strate nu=0.01 du test.

Métriques (par sous-ensemble x cible, en unités physiques) :
    rmse, rel_rmse_mean, rel_rmse_std, mae, max_abs_error,
    pearson, spearman, r2, calibration_slope, calibration_intercept

Réutilise : `physics_candidates.regression_metrics` / `pearson_correlation` /
`spearman_correlation` / `fit_affine_calibration`, et les helpers de
`ml_train_supervised` (set_global_seed, save_json, print_section, ensure_dir).

Ce module est additif : aucun fichier existant n'est modifié.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib.util
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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


def _load_datasets_module():
    return _load_module_from_candidates('ml_datasets', ['ml_datasets.py', 'ml/datasets.py'])


def _load_experiment_c_datasets_module():
    return _load_module_from_candidates(
        'ml_datasets_experiment_c', ['ml_datasets_experiment_c.py'])


def _load_models_module():
    return _load_module_from_candidates('ml_models', ['ml_models.py', 'ml/models.py'])


def _load_experiment_c_models_module():
    return _load_module_from_candidates(
        'ml_models_experiment_c', ['ml_models_experiment_c.py'])


def _load_physics_module():
    return _load_module_from_candidates(
        'physics_candidates', ['physics_candidates.py'])


# =============================================================================
# 2. Utilitaires (mêmes patterns que ml_train_supervised)
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Fixe les graines numpy / python / torch pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def input_key_for_variant(variant: str) -> str:
    """Clé d'entrée du batch selon la variante : 'window' pour C3, 'u' sinon."""
    return "window" if variant == "c3_window" else "u"


# =============================================================================
# 3. DataLoaders
# =============================================================================

def build_experiment_c_dataloaders(
    bundle: Dict[str, Any],
    variant: str,
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Construit les DataLoaders train / val / test pour une variante donnée.

    Retour
    ------
    Tuple
        (train_loader, val_loader, test_loader, datasets) où `datasets` est
        le dict {"train": ..., "val": ..., "test": ...} des datasets
        snapshot/window du bundle.
    """
    key = "window" if variant == "c3_window" else "snapshot"
    datasets = {split: bundle[split][key] for split in ("train", "val", "test")}

    def _loader(ds: Any, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader(datasets["train"], shuffle=True),
        _loader(datasets["val"], shuffle=False),
        _loader(datasets["test"], shuffle=False),
        datasets,
    )


# =============================================================================
# 4. Boucles d'entraînement (multi-tâches, MSE standardisée)
# =============================================================================

def train_one_epoch_experiment_c(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    input_key: str,
) -> Dict[str, float]:
    """
    Une époque d'entraînement : loss = MSE(preds (B,5), targets (B,5)),
    les cibles étant déjà standardisées par le dataset.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x = batch[input_key].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = loss_fn(outputs["preds"], targets)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_samples += bs

    return {"loss": total_loss / max(total_samples, 1)}


@torch.no_grad()
def evaluate_loss_experiment_c(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    input_key: str,
) -> Dict[str, float]:
    """Loss de validation (MSE multi-tâches standardisée)."""
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        x = batch[input_key].to(device)
        targets = batch["targets"].to(device)
        outputs = model(x)
        loss = loss_fn(outputs["preds"], targets)
        bs = x.shape[0]
        total_loss += float(loss.item()) * bs
        total_samples += bs

    return {"loss": total_loss / max(total_samples, 1)}


def train_experiment_c_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    input_key: str,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 200,
    patience: int = 10,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    model_config: Optional[Dict[str, Any]] = None,
    input_standardizer: Optional[Any] = None,
    target_standardizer: Optional[Any] = None,
    variant: str = "c2_cnn",
    print_every: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Entraîne un modèle de l'expérience C avec early stopping sur la loss de
    validation et sauvegarde du meilleur checkpoint (format proche de
    `ml_train_supervised.train_energy_regressor`).

    Retour
    ------
    Dict[str, Any]
        history, best_val_loss, best_epoch, best_model_path, stopped_early.
    """
    device_obj = device or torch.device("cpu")
    output_dir = ensure_dir(output_dir)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch_experiment_c(model, train_loader, optimizer, loss_fn, device_obj, input_key)
        val_metrics = evaluate_loss_experiment_c(model, val_loader, loss_fn, device_obj, input_key)

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])

        if verbose and (epoch % print_every == 0 or epoch == 1):
            print(
                f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"train_loss={train_metrics['loss']:.6e} | val_loss={val_metrics['loss']:.6e}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "variant": variant,
                    "model_config": model_config or {},
                    "input_standardizer": input_standardizer.to_dict() if input_standardizer is not None else None,
                    "target_standardizer": target_standardizer.to_dict() if target_standardizer is not None else None,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"Early stopping à l'époque {epoch} (patience={patience}).")
                break

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_path": str(best_model_path),
        "stopped_early": epochs_without_improvement >= patience,
    }


def load_best_model_experiment_c(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Recharge le meilleur checkpoint dans le modèle et retourne le dict."""
    device_obj = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return checkpoint


# =============================================================================
# 5. Inférence sur un dataset complet
# =============================================================================

@torch.no_grad()
def predict_dataset_experiment_c(
    model: nn.Module,
    dataset: Any,
    target_standardizer: Optional[Any],
    input_key: str,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
    num_workers: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Prédit toutes les cibles d'un dataset et renvoie les prédictions en
    unités PHYSIQUES (inverse du target_standardizer) ainsi que les cibles
    physiques et les métadonnées.

    Retour
    ------
    Dict[str, np.ndarray]
        - "preds"   : (n, 5) prédictions physiques
        - "targets" : (n, 5) cibles physiques
        - "meta"    : dict de tableaux 1D (nu, time, trajectory_index,
                      time_index, energy0)
    """
    device_obj = device or torch.device("cpu")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    preds_std, targets_std = [], []
    metas: Dict[str, List[np.ndarray]] = {
        "nu": [], "time": [], "trajectory_index": [], "time_index": [], "energy0": [],
    }

    for batch in loader:
        x = batch[input_key].to(device_obj)
        outputs = model(x)
        preds_std.append(outputs["preds"].cpu().numpy())
        targets_std.append(batch["targets"].cpu().numpy())
        for key in metas:
            metas[key].append(np.asarray(batch[key].cpu().numpy()))

    preds_std = np.concatenate(preds_std, axis=0)
    targets_std = np.concatenate(targets_std, axis=0)
    meta = {key: np.concatenate(vals, axis=0) for key, vals in metas.items()}

    if target_standardizer is not None:
        preds = target_standardizer.inverse_transform(preds_std)
        targets = target_standardizer.inverse_transform(targets_std)
    else:
        preds = preds_std
        targets = targets_std

    return {"preds": preds, "targets": targets, "meta": meta}


# =============================================================================
# 6. Métriques (numpy pur, réutilise physics_candidates)
# =============================================================================

def compute_target_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Métriques complètes pour une cible scalaire (unités physiques) :
    rmse, rel_rmse_mean, rel_rmse_std, mae, max_abs_error, pearson, spearman,
    r2, calibration_slope, calibration_intercept.
    """
    physics = _load_physics_module()
    pred = np.asarray(pred, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)

    m = physics.regression_metrics(pred, target)
    mean_abs_target = float(np.mean(np.abs(target)))
    std_target = float(np.std(target))
    m["rel_rmse_mean"] = m["rmse"] / max(mean_abs_target, 1.0e-30)
    m["rel_rmse_std"] = m["rmse"] / max(std_target, 1.0e-30)
    m["pearson"] = physics.pearson_correlation(pred, target)
    m["spearman"] = physics.spearman_correlation(pred, target)
    cal = physics.fit_affine_calibration(pred, target)  # target ≈ a * pred + b
    m["calibration_slope"] = cal["a"]
    m["calibration_intercept"] = cal["b"]
    return m


def build_eval_subsets(
    meta: Dict[str, np.ndarray],
    e0_quartile_edges: Optional[np.ndarray] = None,
    eval_times: Sequence[float] = (0.0, 1.0, 2.0),
) -> Dict[str, np.ndarray]:
    """
    Construit les masques de sous-ensembles d'évaluation :
    global, par strate de nu, par temps, par quartile de E(0).
    """
    n = meta["nu"].shape[0]
    subsets: Dict[str, np.ndarray] = {"global": np.ones(n, dtype=bool)}

    for nu in np.sort(np.unique(meta["nu"])):
        subsets[f"nu={nu:g}"] = np.isclose(meta["nu"], nu)

    for t in eval_times:
        subsets[f"t={t:g}"] = np.isclose(meta["time"], float(t))

    if e0_quartile_edges is not None:
        edges = np.asarray(e0_quartile_edges, dtype=float)
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            if i == len(edges) - 2:
                mask = (meta["energy0"] >= lo) & (meta["energy0"] <= hi)
            else:
                mask = (meta["energy0"] >= lo) & (meta["energy0"] < hi)
            subsets[f"e0_quartile={i}"] = mask

    return subsets


def _aggregate_per_trajectory(
    preds: np.ndarray,
    targets: np.ndarray,
    trajectory_index: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Moyenne temporelle par trajectoire de preds et targets."""
    uniq, inv = np.unique(trajectory_index, return_inverse=True)
    n = len(uniq)
    agg_p = np.zeros((n, preds.shape[1]), dtype=float)
    agg_t = np.zeros((n, targets.shape[1]), dtype=float)
    np.add.at(agg_p, inv, preds)
    np.add.at(agg_t, inv, targets)
    counts = np.bincount(inv)[:, None]
    return agg_p / counts, agg_t / counts


def evaluate_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    meta: Dict[str, np.ndarray],
    target_names: Sequence[str],
    e0_quartile_edges: Optional[np.ndarray] = None,
    eval_times: Sequence[float] = (0.0, 1.0, 2.0),
    protocol: str = "standard",
    model_label: str = "model",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Évalue des prédictions (n, 5) sur tous les sous-ensembles, plus le
    sous-ensemble agrégé par trajectoire.

    Retour
    ------
    Tuple
        (nested, rows) :
        - nested : {subset: {target: {metric: valeur}}}
        - rows   : liste de dicts longs {protocol, model, subset, target,
                   metric, value} (pour CSV)
    """
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    target_names = tuple(target_names)

    subsets = build_eval_subsets(meta, e0_quartile_edges, eval_times)

    nested: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []

    def _add(subset_name: str, p: np.ndarray, t: np.ndarray) -> None:
        nested[subset_name] = {}
        for j, name in enumerate(target_names):
            m = compute_target_metrics(p[:, j], t[:, j])
            nested[subset_name][name] = m
            for metric_name, value in m.items():
                rows.append({
                    "protocol": protocol,
                    "model": model_label,
                    "subset": subset_name,
                    "target": name,
                    "metric": metric_name,
                    "value": float(value),
                })

    for subset_name, mask in subsets.items():
        if mask.sum() == 0:
            # sous-ensemble vide (ex. temps demandé absent de la grille) : ignoré
            continue
        _add(subset_name, preds[mask], targets[mask])

    agg_p, agg_t = _aggregate_per_trajectory(preds, targets, meta["trajectory_index"])
    _add("per_trajectory", agg_p, agg_t)

    return nested, rows


# =============================================================================
# 7. Baselines triviales et diagnostics de séparabilité
# =============================================================================

def baseline_predictions(
    train_targets: np.ndarray,
    train_nu: np.ndarray,
    test_nu: np.ndarray,
    kind: str = "global_mean",
) -> np.ndarray:
    """
    Prédictions de baselines triviales (en unités physiques) :

    - "global_mean" : moyenne globale (par cible) calculée sur le train ;
    - "nu_oracle"   : moyenne par strate de nu (oracle de nu), calculée sur
                      le train.
    """
    train_targets = np.asarray(train_targets, dtype=float)
    train_nu = np.asarray(train_nu, dtype=float)
    test_nu = np.asarray(test_nu, dtype=float)

    if kind == "global_mean":
        means = train_targets.mean(axis=0)
        return np.tile(means, (test_nu.shape[0], 1))
    if kind == "nu_oracle":
        out = np.zeros((test_nu.shape[0], train_targets.shape[1]), dtype=float)
        global_means = train_targets.mean(axis=0)
        for i, nu in enumerate(test_nu):
            mask = np.isclose(train_nu, nu)
            out[i] = train_targets[mask].mean(axis=0) if mask.any() else global_means
        return out
    raise ValueError(f"`kind` inconnu : {kind!r} ('global_mean' ou 'nu_oracle').")


def separability_diagnostics(
    preds: np.ndarray,
    meta: Dict[str, np.ndarray],
    target_names: Sequence[str],
) -> Dict[str, Any]:
    """
    Diagnostics de séparabilité entre les têtes D et nu*D :

    - r2_pred_D_vs_pred_nuD_global : R² de la régression pred_D ~ pred_nuD
      (1.0 = têtes parfaitement équivalentes à une constante près) ;
    - par strate de nu : médiane du ratio pred_nuD / pred_D (doit suivre nu).
    """
    physics = _load_physics_module()
    target_names = tuple(target_names)
    j_nuD = target_names.index("nu_dissipation")
    j_D = target_names.index("dissipation")

    pred_nuD = preds[:, j_nuD]
    pred_D = preds[:, j_D]

    r2 = physics.regression_metrics(pred_D, pred_nuD)["r2"]

    per_nu_ratio: Dict[str, float] = {}
    for nu in np.sort(np.unique(meta["nu"])):
        mask = np.isclose(meta["nu"], nu)
        ratio = pred_nuD[mask] / np.maximum(np.abs(pred_D[mask]), 1.0e-30)
        per_nu_ratio[f"nu={nu:g}"] = float(np.median(ratio))

    return {
        "r2_pred_D_vs_pred_nuD_global": float(r2),
        "per_nu_median_ratio_pred_nuD_over_pred_D": per_nu_ratio,
    }


# =============================================================================
# 8. Orchestration complète : entraînement + évaluation + sauvegarde
# =============================================================================

def train_and_evaluate_experiment_c(
    variant: str,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    protocol: str = "standard",
    heldout_nu_values: Sequence[float] = (0.05, 0.2),
    window_size: int = 5,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 200,
    patience: int = 10,
    weight_decay: float = 0.0,
    latent_dim: int = 128,
    encoder_hidden_dims: Sequence[int] = (512, 256),
    conv_channels: Sequence[int] = (16, 32, 32, 64, 64),
    kernel_size: int = 5,
    conv_stride: int = 2,
    trunk_dims: Sequence[int] = (256,),
    head_hidden_dims: Sequence[int] = (64,),
    activation: str = "relu",
    dropout: float = 0.0,
    eval_times: Sequence[float] = (0.0, 1.0, 2.0),
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: int = 0,
    print_every: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Pipeline complet de l'expérience C pour une variante et un protocole :

    1. construit les datasets (avec filtre de nu si protocol="heldout_nu"),
    2. entraîne le modèle (early stopping, meilleur checkpoint),
    3. prédit sur le test complet et calcule les métriques par sous-ensemble,
    4. évalue les baselines triviales,
    5. calcule les diagnostics de séparabilité,
    6. sauvegarde metrics JSON + CSV long format + standardizers.

    Retour
    ------
    Dict[str, Any]
        Résultats complets (history, métriques, baselines, séparabilité).
    """
    if protocol not in ("standard", "heldout_nu"):
        raise ValueError("`protocol` doit être 'standard' ou 'heldout_nu'.")

    set_global_seed(seed)
    device_obj = torch.device(device or "cpu")
    output_dir = ensure_dir(output_dir)

    ec_datasets = _load_experiment_c_datasets_module()
    ec_models = _load_experiment_c_models_module()
    datasets_module = _load_datasets_module()
    models_module = _load_models_module()

    target_names = ec_datasets.EXPERIMENT_C_TARGET_NAMES

    if verbose:
        print_section(f"Expérience C : variante={variant}, protocole={protocol}")

    # 1) datasets
    nu_filter_train = tuple(float(v) for v in heldout_nu_values) if protocol == "heldout_nu" else None
    bundle = ec_datasets.build_experiment_c_datasets(
        train_path, val_path, test_path,
        normalize=True,
        normalize_targets=True,
        window_size=window_size,
        nu_filter_train=nu_filter_train,
    )

    # cibles physiques du train (pour baselines) et bords des quartiles de E(0)
    train_raw = datasets_module.load_dataset_npz(train_path)
    train_targets_dict, _ = ec_datasets.compute_experiment_c_targets_from_dataset(train_raw)
    n_times = train_raw["states"].shape[1]
    train_targets_phys = np.stack(
        [train_targets_dict[name] for name in target_names], axis=-1
    ).reshape(-1, len(target_names))
    train_nu_rep = np.repeat(np.asarray(train_raw["nu_values"], dtype=float), n_times)
    e0_quartile_edges = np.quantile(np.asarray(train_raw["energy0"], dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])

    # 2) modèle + entraînement
    nx = bundle["train"]["snapshot"].nx
    model = ec_models.build_experiment_c_model(
        variant,
        nx=nx,
        window_size=window_size,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        trunk_dims=trunk_dims,
        head_hidden_dims=head_hidden_dims,
        activation=activation,
        dropout=dropout,
    )
    model_config = {
        "variant": variant,
        "nx": nx,
        "window_size": window_size,
        "latent_dim": latent_dim,
        "encoder_hidden_dims": list(encoder_hidden_dims),
        "conv_channels": list(conv_channels),
        "kernel_size": kernel_size,
        "conv_stride": conv_stride,
        "trunk_dims": list(trunk_dims),
        "head_hidden_dims": list(head_hidden_dims),
        "activation": activation,
        "dropout": dropout,
    }
    if verbose:
        print(f"Modèle : {models_module.summarize_model(model, variant).to_dict()}")

    input_key = input_key_for_variant(variant)
    train_loader, val_loader, test_loader, datasets = build_experiment_c_dataloaders(
        bundle, variant, batch_size=batch_size, num_workers=num_workers
    )

    train_result = train_experiment_c_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        input_key=input_key,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience,
        weight_decay=weight_decay,
        device=device_obj,
        model_config=model_config,
        input_standardizer=bundle["input_standardizer"],
        target_standardizer=bundle["target_standardizer"],
        variant=variant,
        print_every=print_every,
        verbose=verbose,
    )

    # recharge le meilleur checkpoint
    load_best_model_experiment_c(model, train_result["best_model_path"], device=device_obj)

    # 3) prédictions sur le test complet
    test_dataset = datasets["test"]
    pred_out = predict_dataset_experiment_c(
        model, test_dataset, bundle["target_standardizer"], input_key,
        device=device_obj, batch_size=batch_size, num_workers=num_workers,
    )
    preds, targets_phys, meta = pred_out["preds"], pred_out["targets"], pred_out["meta"]

    nested, rows = evaluate_predictions(
        preds, targets_phys, meta, target_names,
        e0_quartile_edges=e0_quartile_edges,
        eval_times=eval_times,
        protocol=protocol,
        model_label=variant,
    )

    # 4) baselines triviales
    baselines: Dict[str, Any] = {}
    baseline_rows: List[Dict[str, Any]] = []
    for kind in ("global_mean", "nu_oracle"):
        bp = baseline_predictions(train_targets_phys, train_nu_rep, meta["nu"], kind=kind)
        bn, br = evaluate_predictions(
            bp, targets_phys, meta, target_names,
            e0_quartile_edges=e0_quartile_edges,
            eval_times=eval_times,
            protocol=protocol,
            model_label=f"baseline_{kind}",
        )
        baselines[kind] = bn
        baseline_rows.extend(br)

    # 5) diagnostics de séparabilité
    sep = separability_diagnostics(preds, meta, target_names)

    # 6) sauvegarde
    results: Dict[str, Any] = {
        "variant": variant,
        "protocol": protocol,
        "heldout_nu_values": list(heldout_nu_values) if protocol == "heldout_nu" else None,
        "target_names": list(target_names),
        "model_config": model_config,
        "model_summary": models_module.summarize_model(model, variant).to_dict(),
        "best_epoch": train_result["best_epoch"],
        "best_val_loss": train_result["best_val_loss"],
        "stopped_early": train_result["stopped_early"],
        "history": train_result["history"],
        "checkpoint_path": train_result["best_model_path"],
        "e0_quartile_edges": e0_quartile_edges.tolist(),
        "metrics": nested,
        "baselines": baselines,
        "separability": sep,
    }

    metrics_json_path = output_dir / f"metrics_{variant}_{protocol}.json"
    save_json(metrics_json_path, results)

    all_rows = rows + baseline_rows
    metrics_csv_path = output_dir / f"metrics_long_{variant}_{protocol}.csv"
    pd.DataFrame(all_rows).to_csv(metrics_csv_path, index=False)

    save_json(
        output_dir / "standardizers.json",
        {
            "input_standardizer": bundle["input_standardizer"].to_dict() if bundle["input_standardizer"] is not None else None,
            "target_standardizer": bundle["target_standardizer"].to_dict() if bundle["target_standardizer"] is not None else None,
        },
    )

    if verbose:
        print(f"Métriques sauvegardées : {metrics_json_path}")
        print(f"CSV long format       : {metrics_csv_path}")
        # aperçu des métriques globales
        for name in target_names:
            m = nested["global"][name]
            print(
                f"  global {name:15s} rmse={m['rmse']:.4e} rel_rmse={m['rel_rmse_mean']:.4e} "
                f"r2={m['r2']:.4f} spearman={m['spearman']:.4f}"
            )

    return results


# =============================================================================
# 9. Bloc de test minimal (nécessite torch ; petit dataset synthétique)
# =============================================================================

if __name__ == "__main__":
    # Petit dataset synthétique en mémoire, sauvegardé en npz temporaires.
    import tempfile

    rng = np.random.default_rng(0)
    n_traj, n_times, nx = 12, 21, 64
    L = 2.0 * np.pi
    x = np.linspace(0.0, L, nx, endpoint=False)
    dx = float(x[1] - x[0])
    times = np.linspace(0.0, 2.0, n_times)

    states = rng.normal(size=(n_traj, n_times, nx)).astype(np.float32)
    energy = (0.5 * np.sum(states ** 2, axis=2) * dx).astype(np.float32)
    nu_values = np.tile(np.array([0.01, 0.05, 0.2], dtype=np.float32), n_traj // 3)
    ux = (np.roll(states, -1, axis=2) - np.roll(states, 1, axis=2)) / (2.0 * dx)
    dissipation = (np.sum(ux ** 2, axis=2) * dx).astype(np.float32)

    def _fake(split_n):
        idx = np.arange(split_n)
        return {
            "states": states[idx % n_traj],
            "times": times,
            "energy": energy[idx % n_traj],
            "dissipation": dissipation[idx % n_traj],
            "dissipation_physical": (nu_values[idx % n_traj][:, None] * dissipation[idx % n_traj]).astype(np.float32),
            "x": x,
            "dx": np.array(dx, dtype=float),
            "nu_values": nu_values[idx % n_traj],
            "energy0": energy[idx % n_traj, 0].copy(),
            "ic_type_ids": (idx % 3).astype(np.int64),
            "ic_type_names": np.array(["fourier", "gaussian_bumps", "random_smooth"]),
        }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        paths = {}
        for split, n in (("train", 9), ("val", 3), ("test", 3)):
            p = tmp / f"{split}.npz"
            np.savez(p, **_fake(n))
            paths[split] = p

        results = train_and_evaluate_experiment_c(
            variant="c2_cnn",
            train_path=paths["train"],
            val_path=paths["val"],
            test_path=paths["test"],
            output_dir=tmp / "out",
            protocol="standard",
            window_size=5,
            batch_size=16,
            num_epochs=2,
            patience=2,
            latent_dim=16,
            conv_channels=(8, 16),
            trunk_dims=(32,),
            head_hidden_dims=(16,),
            encoder_hidden_dims=(32,),
            print_every=1,
            verbose=True,
        )
        print("\nClés du résultat :", sorted(results.keys()))
        print("Sous-ensembles évalués :", sorted(results["metrics"].keys()))
        print("Séparabilité :", results["separability"])
