"""
ml_train_experiment_a_seq.py
============================

Moteur d'entraînement / évaluation de window-A (Architecture A-Sequence, Step 5) :
`LatentDynamicsAutoencoder` (Architecture A) avec un encodeur Conv1D sur fenêtre
de `window_size` snapshots (le temps entre comme canaux, comme c3_window), sur le
dataset nu_sweep_v1.

Pipeline par protocole ("standard" / "heldout_nu") :

1. datasets de paires de fenêtres (`NuSweepWindowPairDataset`, stride 1) avec
   standardizer des états ajusté sur le train uniquement ;
2. modèle window-A (`build_window_dynamics_autoencoder`) + loss Architecture A
   inchangée (rec + prédiction latente + hinge de monotonie + plancher de variance
   par dimension), early stopping sur la val loss, meilleur checkpoint ;
3. prédiction de z (et h) sur le test complet ;
4. évaluation de z vs nu*D avec la machinerie de sous-ensembles d'Experiment C
   (global, par nu, par t, par quartile de E(0), par trajectoire) + baselines
   (global_mean, nu_oracle) + rang PCA de h + RMSE de reconstruction + fraction
   monotone de z ;
5. verdict d'identifiabilité de z vs nu*D à nu=0.01 held-out (Step 5) :
   r_holdout, R² held-out, pente de calibration.

Additif : ne modifie aucun module existant (réutilise ml_datasets_experiment_c,
ml_models_experiment_c, ml_models, ml_losses, ml_train_experiment_c).
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# 1. Chargement flexible des modules du projet (même pattern que ml_train_experiment_c)
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: List[str]):
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
        f"Impossible de charger le module {module_name!r}. Candidats testés : {candidates}"
    )


def _load_datasets_module():
    return _load_module_from_candidates('ml_datasets', ['ml_datasets.py'])


def _load_experiment_c_datasets_module():
    return _load_module_from_candidates('ml_datasets_experiment_c', ['ml_datasets_experiment_c.py'])


def _load_models_module():
    return _load_module_from_candidates('ml_models', ['ml_models.py'])


def _load_experiment_c_models_module():
    return _load_module_from_candidates('ml_models_experiment_c', ['ml_models_experiment_c.py'])


def _load_losses_module():
    return _load_module_from_candidates('ml_losses', ['ml_losses.py'])


def _load_train_experiment_c_module():
    return _load_module_from_candidates('ml_train_experiment_c', ['ml_train_experiment_c.py'])


# =============================================================================
# 2. Utilitaires
# =============================================================================

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


LOSS_KEYS: Tuple[str, ...] = (
    "total_loss", "reconstruction_loss", "prediction_loss",
    "monotonicity_loss", "variance_loss",
)


def _loss_kwargs(
    reconstruction_weight: float,
    prediction_weight: float,
    monotonicity_weight: float,
    variance_weight: float,
    min_latent_variance: float,
    monotonicity_mode: str,
    prediction_mode: str,
) -> Dict[str, Any]:
    return {
        "reconstruction_weight": reconstruction_weight,
        "prediction_weight": prediction_weight,
        "monotonicity_weight": monotonicity_weight,
        "variance_weight": variance_weight,
        "min_latent_variance": min_latent_variance,
        "monotonicity_mode": monotonicity_mode,
        "prediction_mode": prediction_mode,
    }


# =============================================================================
# 3. Datasets et DataLoaders (paires de fenêtres)
# =============================================================================

def build_a_seq_datasets(
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    window_size: int = 25,
    nu_filter_train: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Construit les datasets de paires de fenêtres train / val / test.

    Le standardizer des états est ajusté sur le split train uniquement.
    `nu_filter_train` restreint train ET val aux strates de nu données
    (protocole heldout_nu) ; le test reste complet.
    """
    datasets_module = _load_datasets_module()
    ec_datasets = _load_experiment_c_datasets_module()

    train_raw = datasets_module.load_dataset_npz(train_path)
    val_raw = datasets_module.load_dataset_npz(val_path)
    test_raw = datasets_module.load_dataset_npz(test_path)

    input_standardizer = datasets_module.fit_global_standardizer(
        np.asarray(train_raw["states"], dtype=np.float32)
    )

    def _make(split_raw: Dict[str, Any], nu_filter: Optional[Sequence[float]]) -> Any:
        return ec_datasets.NuSweepWindowPairDataset(
            split_raw,
            window_size=window_size,
            stride=1,
            normalize=True,
            standardizer=input_standardizer,
            nu_filter=nu_filter,
        )

    return {
        "train": _make(train_raw, nu_filter_train),
        "val": _make(val_raw, nu_filter_train),
        "test": _make(test_raw, None),
        "input_standardizer": input_standardizer,
    }


def build_a_seq_dataloaders(
    bundle: Dict[str, Any],
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    def _loader(ds: Any, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return (
        _loader(bundle["train"], shuffle=True),
        _loader(bundle["val"], shuffle=False),
        _loader(bundle["test"], shuffle=False),
    )


# =============================================================================
# 4. Boucles d'entraînement (loss Architecture A, 4 termes)
# =============================================================================

def train_one_epoch_a_seq(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_kwargs: Dict[str, Any],
) -> Dict[str, float]:
    """Une époque : paires de fenêtres (window_t, window_tp), cible de
    reconstruction = snapshot final u_t. Loss = latent_dynamics_autoencoder_total_loss."""
    losses_module = _load_losses_module()
    model.train()
    total = {key: 0.0 for key in LOSS_KEYS}
    total_samples = 0

    for batch in dataloader:
        window_t = batch["window_t"].to(device)
        window_tp = batch["window_tp"].to(device)
        u_t = batch["u_t"].to(device)

        optimizer.zero_grad()
        outputs = model(window_t, window_tp)
        losses = losses_module.latent_dynamics_autoencoder_total_loss(
            reconstruction=outputs["reconstruction"],
            target_u=u_t,
            predicted_latent=outputs["h_tp_pred"],
            target_latent=outputs["h_tp"],
            z_t=outputs["z_t"],
            z_tp=outputs["z_tp"],
            h_t=outputs["h_t"],
            **loss_kwargs,
        )
        loss = losses["total_loss"]
        loss.backward()
        optimizer.step()

        bs = window_t.shape[0]
        total_samples += bs
        for key in LOSS_KEYS:
            total[key] += float(losses[key].item()) * bs

    return {key: value / max(total_samples, 1) for key, value in total.items()}


@torch.no_grad()
def evaluate_loss_a_seq(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Loss de validation + fraction monotone (z_{t+1} <= z_t sur les paires)."""
    losses_module = _load_losses_module()
    model.eval()
    total = {key: 0.0 for key in LOSS_KEYS}
    total_samples = 0
    monotone_ok = 0
    monotone_count = 0

    for batch in dataloader:
        window_t = batch["window_t"].to(device)
        window_tp = batch["window_tp"].to(device)
        u_t = batch["u_t"].to(device)

        outputs = model(window_t, window_tp)
        losses = losses_module.latent_dynamics_autoencoder_total_loss(
            reconstruction=outputs["reconstruction"],
            target_u=u_t,
            predicted_latent=outputs["h_tp_pred"],
            target_latent=outputs["h_tp"],
            z_t=outputs["z_t"],
            z_tp=outputs["z_tp"],
            h_t=outputs["h_t"],
            **loss_kwargs,
        )

        bs = window_t.shape[0]
        total_samples += bs
        for key in LOSS_KEYS:
            total[key] += float(losses[key].item()) * bs

        z_t = outputs["z_t"].detach().reshape(-1).cpu().numpy()
        z_tp = outputs["z_tp"].detach().reshape(-1).cpu().numpy()
        monotone_ok += int(np.sum(z_tp <= z_t))
        monotone_count += int(z_t.shape[0])

    metrics: Dict[str, Any] = {key: value / max(total_samples, 1) for key, value in total.items()}
    metrics["monotone_fraction"] = float(monotone_ok / max(monotone_count, 1))
    return metrics


def train_window_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 200,
    patience: int = 10,
    weight_decay: float = 0.0,
    device: Optional[torch.device] = None,
    loss_kwargs: Optional[Dict[str, Any]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    input_standardizer: Optional[Any] = None,
    print_every: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Early stopping sur la val loss (total), sauvegarde du meilleur checkpoint."""
    device_obj = device or torch.device("cpu")
    loss_kwargs = loss_kwargs or _loss_kwargs(1.0, 1.0, 1.0, 0.01, 0.01, "squared_hinge", "mse")
    output_dir = ensure_dir(output_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch_a_seq(model, train_loader, optimizer, device_obj, loss_kwargs)
        val_metrics = evaluate_loss_a_seq(model, val_loader, device_obj, loss_kwargs)

        history["train_loss"].append(train_metrics["total_loss"])
        history["val_loss"].append(val_metrics["total_loss"])

        if verbose and (epoch % print_every == 0 or epoch == 1):
            print(
                f"[Epoch {epoch:03d}/{num_epochs:03d}] "
                f"train_loss={train_metrics['total_loss']:.6e} | val_loss={val_metrics['total_loss']:.6e}"
            )

        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "model_config": model_config or {},
                    "input_standardizer": input_standardizer.to_dict() if input_standardizer is not None else None,
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


def load_best_model_a_seq(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    device_obj = device or torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device_obj)
    model.eval()
    return checkpoint


# =============================================================================
# 5. Inférence sur le test complet (z, h, reconstruction)
# =============================================================================

@torch.no_grad()
def predict_a_seq(
    model: nn.Module,
    dataset: Any,
    device: Optional[torch.device] = None,
    batch_size: int = 512,
    num_workers: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Prédit z_t, h_t et la reconstruction pour chaque paire du dataset
    (fenêtre window_t seule ; la fenêtre suivante sert uniquement à la loss).

    Retour : dict avec "z" (n,), "h" (n, latent_dim), "rec" (n, Nx), "u" (n, Nx)
    et "meta" (nu, time_t, trajectory_index, time_index_t, energy0,
    nu_dissipation_t, energy_t, dissipation_t).
    """
    device_obj = device or torch.device("cpu")
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    z_parts, h_parts, rec_parts, u_parts = [], [], [], []
    meta_keys = ("nu", "time_t", "trajectory_index", "time_index_t", "energy0",
                 "nu_dissipation_t", "energy_t", "dissipation_t")
    metas: Dict[str, List[np.ndarray]] = {k: [] for k in meta_keys}

    for batch in loader:
        window_t = batch["window_t"].to(device_obj)
        outputs = model(window_t)
        z_parts.append(outputs["z_t"].detach().reshape(-1).cpu().numpy().astype(np.float64))
        h_parts.append(outputs["h_t"].detach().cpu().numpy().astype(np.float64))
        rec_parts.append(outputs["reconstruction"].detach().cpu().numpy().astype(np.float64))
        u_parts.append(batch["u_t"].detach().cpu().numpy().astype(np.float64))
        for key in meta_keys:
            metas[key].append(np.asarray(batch[key].cpu().numpy()))

    return {
        "z": np.concatenate(z_parts, axis=0),
        "h": np.concatenate(h_parts, axis=0),
        "rec": np.concatenate(rec_parts, axis=0),
        "u": np.concatenate(u_parts, axis=0),
        "meta": {key: np.concatenate(vals, axis=0) for key, vals in metas.items()},
    }


# =============================================================================
# 6. Évaluation de z vs nu*D (réutilise la machinerie d'Experiment C)
# =============================================================================

def _meta_for_subsets(meta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "nu": meta["nu"],
        "time": meta["time_t"],
        "energy0": meta["energy0"],
        "trajectory_index": meta["trajectory_index"],
    }


def evaluate_z_vs_nuD(
    z: np.ndarray,
    meta: Dict[str, np.ndarray],
    e0_quartile_edges: Optional[np.ndarray],
    eval_times: Sequence[float],
    protocol: str,
    model_label: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Métriques de z vs nu*D sur tous les sous-ensembles (global, par nu, par t,
    par quartile de E(0), par trajectoire). Réutilise evaluate_predictions de
    ml_train_experiment_c avec une seule cible ("nu_dissipation")."""
    mt = _load_train_experiment_c_module()
    preds = np.asarray(z, dtype=float).reshape(-1, 1)
    targets = np.asarray(meta["nu_dissipation_t"], dtype=float).reshape(-1, 1)
    return mt.evaluate_predictions(
        preds, targets, _meta_for_subsets(meta), ("nu_dissipation",),
        e0_quartile_edges=e0_quartile_edges,
        eval_times=eval_times,
        protocol=protocol,
        model_label=model_label,
    )


def baseline_z_vs_nuD(
    train_dataset: Any,
    meta: Dict[str, np.ndarray],
    e0_quartile_edges: Optional[np.ndarray],
    eval_times: Sequence[float],
    protocol: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Baselines triviales pour z vs nu*D : moyenne globale et moyenne par nu
    (nu_oracle, oracle de nu) calculées sur le train."""
    mt = _load_train_experiment_c_module()
    train_targets = np.asarray(
        [[train_dataset.nu_dissipation[i, t]] for (i, t) in train_dataset.index_map],
        dtype=float,
    )
    train_nu = np.asarray([train_dataset.nu_values[i] for (i, _t) in train_dataset.index_map], dtype=float)
    baselines: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for kind in ("global_mean", "nu_oracle"):
        bp = mt.baseline_predictions(train_targets, train_nu, meta["nu"], kind=kind)
        bn, br = mt.evaluate_predictions(
            bp, np.asarray(meta["nu_dissipation_t"], dtype=float).reshape(-1, 1),
            _meta_for_subsets(meta), ("nu_dissipation",),
            e0_quartile_edges=e0_quartile_edges,
            eval_times=eval_times,
            protocol=protocol,
            model_label=f"baseline_{kind}",
        )
        baselines[kind] = bn
        rows.extend(br)
    return baselines, rows


# =============================================================================
# 7. Verdict d'identifiabilité de z vs nu*D (critères Step 5)
# =============================================================================

def verdict_a_seq(
    standard_results: Dict[str, Any],
    heldout_results: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """
    Verdict sur le protocole heldout_nu, strate nu=0.01, z vs nu*D :

    - r_holdout      : Spearman(z, nuD) @ nu=0.01 held-out ;
    - r2_holdout     : R² affine(nuD ~ z) @ nu=0.01 held-out ;
    - calibration_slope : pente a de nuD ≈ a·z + b @ nu=0.01 held-out ;
    - beats_nu_oracle   : rel_rmse(z) < rel_rmse(nu_oracle) @ nu=0.01 held-out ;
    - structurel (protocole standard) : rang effectif de h, RMSE de
      reconstruction (test), fraction monotone de z (test).

    IDENTIFIABLE si les 3 critères held-out + beats_nu_oracle + structurels
    passent ; NOT IDENTIFIABLE si r_holdout < r_holdout_fail ; PARTIAL sinon.
    """
    h = heldout_results["metrics"]["nu=0.01"]["nu_dissipation"]
    r_holdout = float(h["spearman"])
    r2_holdout = float(h["r2"])
    slope = float(h["calibration_slope"])
    rel_rmse_holdout = float(h["rel_rmse_mean"])
    oracle = float(
        heldout_results["baselines"]["nu_oracle"]["nu=0.01"]["nu_dissipation"]["rel_rmse_mean"]
    )
    beats_nu_oracle = bool(rel_rmse_holdout < oracle)

    s = standard_results
    effective_rank = float(s["latent_rank"]["effective_rank"])
    rec_rmse = float(s["reconstruction_rmse_test"])
    monotone = float(s["monotone_fraction_test"])
    struct_ok = bool(
        effective_rank >= thresholds["rank_min"]
        and rec_rmse <= thresholds["rec_rmse_max"]
        and monotone >= thresholds["monotone_min"]
    )

    held_ok = bool(
        r_holdout >= thresholds["r_holdout_min"]
        and r2_holdout >= thresholds["r2_holdout_min"]
        and thresholds["slope_min"] <= slope <= thresholds["slope_max"]
        and beats_nu_oracle
    )

    if held_ok and struct_ok:
        verdict = "IDENTIFIABLE"
    elif r_holdout < thresholds["r_holdout_fail"]:
        verdict = "NOT IDENTIFIABLE"
    else:
        verdict = "PARTIAL"

    return {
        "target": "nu_dissipation",
        "r_holdout": r_holdout,
        "r2_holdout_nu001": r2_holdout,
        "calibration_slope_nu001": slope,
        "rel_rmse_holdout_nu001": rel_rmse_holdout,
        "baseline_nu_oracle_rel_rmse_nu001": oracle,
        "beats_nu_oracle": beats_nu_oracle,
        "effective_rank": effective_rank,
        "reconstruction_rmse_test": rec_rmse,
        "monotone_fraction_test": monotone,
        "structural_ok": struct_ok,
        "verdict": verdict,
    }


# =============================================================================
# 8. Pipeline complet par protocole
# =============================================================================

def train_and_evaluate_a_seq(
    protocol: str,
    train_path: str | Path,
    val_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    window_size: int = 25,
    heldout_nu_values: Sequence[float] = (0.05, 0.2),
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 200,
    patience: int = 10,
    weight_decay: float = 0.0,
    latent_dim: int = 8,
    conv_channels: Sequence[int] = (16, 32, 32, 64, 64),
    kernel_size: int = 5,
    conv_stride: int = 2,
    trunk_dims: Sequence[int] = (256,),
    decoder_hidden_dims: Sequence[int] = (64, 128),
    predictor_hidden_dims: Sequence[int] = (64, 64),
    scalar_head_hidden_dims: Sequence[int] = (),
    activation: str = "relu",
    dropout: float = 0.0,
    reconstruction_weight: float = 1.0,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
    variance_weight: float = 0.01,
    min_latent_variance: float = 0.01,
    monotonicity_mode: str = "squared_hinge",
    prediction_mode: str = "mse",
    eval_times: Sequence[float] = (1.0, 2.0),
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: int = 0,
    print_every: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Pipeline window-A pour un protocole : datasets -> entraînement (early
    stopping) -> meilleur checkpoint -> prédiction test -> métriques z vs nu*D
    + baselines + rang latent + rec RMSE + monotonie -> sauvegarde."""
    if protocol not in ("standard", "heldout_nu"):
        raise ValueError("`protocol` doit être 'standard' ou 'heldout_nu'.")

    set_global_seed(seed)
    device_obj = torch.device(device or "cpu")
    output_dir = ensure_dir(output_dir)

    ec_datasets = _load_experiment_c_datasets_module()
    ec_models = _load_experiment_c_models_module()
    models_module = _load_models_module()

    if verbose:
        print_section(f"Window-A (A-Sequence) : fenêtre w={window_size}, protocole={protocol}")

    loss_kwargs = _loss_kwargs(
        reconstruction_weight, prediction_weight, monotonicity_weight,
        variance_weight, min_latent_variance, monotonicity_mode, prediction_mode,
    )

    # 1) datasets
    nu_filter_train = tuple(float(v) for v in heldout_nu_values) if protocol == "heldout_nu" else None
    bundle = build_a_seq_datasets(
        train_path, val_path, test_path,
        window_size=window_size,
        nu_filter_train=nu_filter_train,
    )
    train_ds = bundle["train"]
    nx = train_ds.nx
    e0_quartile_edges = np.quantile(
        np.asarray(train_ds.energy0, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    # 2) modèle + entraînement
    model = ec_models.build_window_dynamics_autoencoder(
        window_size=window_size,
        nx=nx,
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        trunk_dims=trunk_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        predictor_hidden_dims=predictor_hidden_dims,
        scalar_head_hidden_dims=scalar_head_hidden_dims,
        activation=activation,
        dropout=dropout,
    )
    model_config = {
        "model": "window_dynamics_autoencoder",
        "window_size": int(window_size),
        "nx": int(nx),
        "latent_dim": latent_dim,
        "conv_channels": list(conv_channels),
        "kernel_size": kernel_size,
        "conv_stride": conv_stride,
        "trunk_dims": list(trunk_dims),
        "decoder_hidden_dims": list(decoder_hidden_dims),
        "predictor_hidden_dims": list(predictor_hidden_dims),
        "scalar_head_hidden_dims": list(scalar_head_hidden_dims),
        "activation": activation,
        "dropout": dropout,
        "loss": dict(loss_kwargs),
    }
    if verbose:
        print(f"Modèle : {models_module.summarize_model(model, 'window_dynamics_autoencoder').to_dict()}")

    train_loader, val_loader, test_loader = build_a_seq_dataloaders(
        bundle, batch_size=batch_size, num_workers=num_workers
    )

    train_result = train_window_autoencoder(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        patience=patience,
        weight_decay=weight_decay,
        device=device_obj,
        loss_kwargs=loss_kwargs,
        model_config=model_config,
        input_standardizer=bundle["input_standardizer"],
        print_every=print_every,
        verbose=verbose,
    )

    load_best_model_a_seq(model, train_result["best_model_path"], device_obj)

    # 3) prédiction sur le test complet
    pred = predict_a_seq(model, bundle["test"], device=device_obj, batch_size=batch_size, num_workers=num_workers)
    z, h, rec, u, meta = pred["z"], pred["h"], pred["rec"], pred["u"], pred["meta"]

    reconstruction_rmse_test = float(np.sqrt(np.mean((rec - u) ** 2)))

    # rang latent (PCA de h sur le test)
    pca = models_module.latent_pca_stats(torch.as_tensor(h, dtype=torch.float32))
    latent_rank = {
        "effective_rank": float(pca["effective_rank"]),
        "eigenvalues": [float(v) for v in np.asarray(pca["eigenvalues"]).reshape(-1)],
        "max_offline_residual": float(pca["max_offline_residual"]),
    }

    # loss + monotonie sur le test (paires)
    test_eval = evaluate_loss_a_seq(model, test_loader, device_obj, loss_kwargs)
    monotone_fraction_test = float(test_eval["monotone_fraction"])

    # 4) métriques z vs nu*D + baselines
    model_label = f"window_a_w{window_size}"
    nested, rows = evaluate_z_vs_nuD(
        z, meta, e0_quartile_edges, eval_times, protocol, model_label
    )
    baselines, baseline_rows = baseline_z_vs_nuD(
        train_ds, meta, e0_quartile_edges, eval_times, protocol
    )

    # 5) sauvegarde
    results: Dict[str, Any] = {
        "experiment": "experiment_a_seq",
        "protocol": protocol,
        "window_size": int(window_size),
        "heldout_nu_values": list(heldout_nu_values) if protocol == "heldout_nu" else None,
        "model_config": model_config,
        "best_epoch": train_result["best_epoch"],
        "best_val_loss": train_result["best_val_loss"],
        "stopped_early": train_result["stopped_early"],
        "history": train_result["history"],
        "checkpoint_path": train_result["best_model_path"],
        "reconstruction_rmse_test": reconstruction_rmse_test,
        "latent_rank": latent_rank,
        "monotone_fraction_test": monotone_fraction_test,
        "test_losses": {k: v for k, v in test_eval.items() if k != "monotone_fraction"},
        "e0_quartile_edges": e0_quartile_edges.tolist(),
        "metrics": nested,
        "baselines": baselines,
    }

    metrics_json_path = output_dir / f"metrics_window_a_w{window_size}_{protocol}.json"
    save_json(metrics_json_path, results)
    pd.DataFrame(rows + baseline_rows).to_csv(
        output_dir / f"metrics_long_window_a_w{window_size}_{protocol}.csv", index=False
    )
    save_json(
        output_dir / "standardizers.json",
        {"input_standardizer": bundle["input_standardizer"].to_dict()},
    )

    if verbose:
        print(f"Métriques sauvegardées : {metrics_json_path}")
        m = nested["global"]["nu_dissipation"]
        print(
            f"  global nu_dissipation: rmse={m['rmse']:.4e} rel_rmse={m['rel_rmse_mean']:.4e} "
            f"r2={m['r2']:.4f} spearman={m['spearman']:.4f} cal_slope={m['calibration_slope']:.4f}"
        )
        print(f"  rec RMSE (test)={reconstruction_rmse_test:.4e} | rang effectif={latent_rank['effective_rank']} "
              f"| monotone={monotone_fraction_test:.4f}")

    return results
