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

def set_global_seed(seed: int, num_threads: Optional[int] = None, deterministic: bool = True) -> None:
    """Seed global + correctif de déterminisme (prereg TD 515cf5b, §9) :
    `torch.use_deterministic_algorithms(True)` et nombre de threads fixé
    (par défaut 4) pour que deux exécutions identiques soient bit-à-bit
    reproductibles sur CPU. `num_threads=None` laisse le défaut torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))


# =============================================================================
# 2b. Traçabilité (prereg TD 515cf5b, §9) : hash git, working tree, versions
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent


def _git(args: List[str]) -> Optional[str]:
    import subprocess
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, timeout=30,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_traceability() -> Dict[str, Any]:
    """Hash git, état du working tree, versions.

    Seuls les fichiers SUIVIS modifiés comptent (`--untracked-files=no`) :
    les artefacts non suivis (npz data, logs, modèles, sorties de runs) ne
    bloquent jamais. Les __pycache__/*.pyc (suivis ici) sont exclus."""
    dirty_files = []
    status = _git(["status", "--porcelain", "--untracked-files=no"]) or ""
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip().strip('"')
        if "__pycache__" in path or path.endswith(".pyc"):
            continue  # caches bytecode : non significatifs pour la reproductibilité du code
        dirty_files.append(path)
    return {
        "git_hash": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "working_tree_dirty": bool(dirty_files),
        "dirty_files": dirty_files,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": __import__("sys").version,
        "num_threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
    }


def assert_clean_working_tree(trace: Dict[str, Any], allow_smoke: bool = False) -> None:
    """Refuse de démarrer un run de production si le working tree est modifié
    (règle 0.3, prereg TD 515cf5b §9). Les __pycache__/*.pyc sont exclus."""
    if allow_smoke:
        return
    if trace.get("working_tree_dirty"):
        raise SystemExit(
            "working tree modifié (fichiers suivis hors __pycache__) : "
            + ", ".join(trace["dirty_files"])
            + " — run refusé (règle 0.3). Commiter ou restaurer ces fichiers, "
            "ou lancer avec --smoke."
        )


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
    "monotonicity_loss", "variance_loss", "z_loss",
)

# Modes de cible physique pour z (Experiment A-Physics) :
#   "none"             -> pas de terme z (window-A historique)
#   "energy_centered"  -> cible = nuD_est_t (différence centrée de E, auto-supervisé)
#   "supervised"       -> cible = nu_dissipation_t stocké (contrôle supervisé)
Z_TARGET_MODES: Tuple[str, ...] = ("none", "energy_centered", "supervised")


def _z_target_key(z_target_mode: str) -> Optional[str]:
    if z_target_mode == "none":
        return None
    if z_target_mode == "energy_centered":
        return "nuD_est_t"
    if z_target_mode == "supervised":
        return "nu_dissipation_t"
    raise ValueError(f"`z_target_mode` inconnu: {z_target_mode!r} (attendu {Z_TARGET_MODES})")


def _loss_kwargs(
    reconstruction_weight: float,
    prediction_weight: float,
    monotonicity_weight: float,
    variance_weight: float,
    min_latent_variance: float,
    monotonicity_mode: str,
    prediction_mode: str,
    z_weight: float = 0.0,
) -> Dict[str, Any]:
    return {
        "reconstruction_weight": reconstruction_weight,
        "prediction_weight": prediction_weight,
        "monotonicity_weight": monotonicity_weight,
        "variance_weight": variance_weight,
        "min_latent_variance": min_latent_variance,
        "monotonicity_mode": monotonicity_mode,
        "prediction_mode": prediction_mode,
        "z_weight": z_weight,
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
        # train complet (sans filtre) : sert UNIQUEMENT au calcul de la baseline
        # nu_oracle (convention README:342 — oracle de nu sur le FULL train).
        "train_full": _make(train_raw, None),
        "input_standardizer": input_standardizer,
    }


def build_a_seq_dataloaders(
    bundle: Dict[str, Any],
    batch_size: int = 256,
    num_workers: int = 0,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Générateur explicite seedé pour le shuffle (correctif déterminisme,
    # prereg TD 515cf5b §9) : deux exécutions de même graine ont le même ordre.
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(int(seed))

    def _loader(ds: Any, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=gen if shuffle else None,
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
    z_target_mode: str = "none",
    z_target_transform: Optional[Tuple[float, float]] = None,
) -> Dict[str, float]:
    """Une époque : paires de fenêtres (window_t, window_tp), cible de
    reconstruction = snapshot final u_t. Loss = latent_dynamics_autoencoder_total_loss
    (+ terme physique z optionnel si z_target_mode != "none").

    `z_target_transform` = (mean, std) : si fourni, la cible z du batch est
    standardisée (t - mean)/std avant la loss (mêmes stats pour tous les modes)."""
    losses_module = _load_losses_module()
    z_key = _z_target_key(z_target_mode)
    model.train()
    total = {key: 0.0 for key in LOSS_KEYS}
    total_samples = 0

    for batch in dataloader:
        window_t = batch["window_t"].to(device)
        window_tp = batch["window_tp"].to(device)
        u_t = batch["u_t"].to(device)
        if z_key is not None:
            z_target = batch[z_key].to(device)
            if z_target_transform is not None:
                mean, std = z_target_transform
                z_target = (z_target - mean) / max(std, 1e-8)
        else:
            z_target = None

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
            z_target=z_target,
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
    z_target_mode: str = "none",
    z_target_transform: Optional[Tuple[float, float]] = None,
    monotone_rtol: float = 0.0,
) -> Dict[str, Any]:
    """Loss de validation + fraction monotone (z_{t+1} <= z_t + atol sur les paires,
    atol = monotone_rtol * mean|z_t| si monotone_rtol > 0, sinon 0)."""
    losses_module = _load_losses_module()
    z_key = _z_target_key(z_target_mode)
    model.eval()
    total = {key: 0.0 for key in LOSS_KEYS}
    total_samples = 0
    monotone_ok = 0
    monotone_count = 0
    monotone_atol = 0.0

    for batch in dataloader:
        window_t = batch["window_t"].to(device)
        window_tp = batch["window_tp"].to(device)
        u_t = batch["u_t"].to(device)
        if z_key is not None:
            z_target = batch[z_key].to(device)
            if z_target_transform is not None:
                mean, std = z_target_transform
                z_target = (z_target - mean) / max(std, 1e-8)
        else:
            z_target = None

        outputs = model(window_t, window_tp)
        losses = losses_module.latent_dynamics_autoencoder_total_loss(
            reconstruction=outputs["reconstruction"],
            target_u=u_t,
            predicted_latent=outputs["h_tp_pred"],
            target_latent=outputs["h_tp"],
            z_t=outputs["z_t"],
            z_tp=outputs["z_tp"],
            h_t=outputs["h_t"],
            z_target=z_target,
            **loss_kwargs,
        )

        bs = window_t.shape[0]
        total_samples += bs
        for key in LOSS_KEYS:
            total[key] += float(losses[key].item()) * bs

        z_t = outputs["z_t"].detach().reshape(-1).cpu().numpy()
        z_tp = outputs["z_tp"].detach().reshape(-1).cpu().numpy()
        if monotone_rtol > 0:
            monotone_atol = float(monotone_rtol) * float(np.mean(np.abs(z_t)))
        monotone_ok += int(np.sum(z_tp <= z_t + monotone_atol))
        monotone_count += int(z_t.shape[0])

    metrics: Dict[str, Any] = {key: value / max(total_samples, 1) for key, value in total.items()}
    metrics["monotone_fraction"] = float(monotone_ok / max(monotone_count, 1))
    metrics["monotone_atol"] = monotone_atol
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
    z_target_mode: str = "none",
    z_target_transform: Optional[Tuple[float, float]] = None,
    monotone_rtol: float = 0.0,
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
        train_metrics = train_one_epoch_a_seq(model, train_loader, optimizer, device_obj, loss_kwargs, z_target_mode, z_target_transform)
        val_metrics = evaluate_loss_a_seq(model, val_loader, device_obj, loss_kwargs, z_target_mode, z_target_transform, monotone_rtol)

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
                 "nu_dissipation_t", "nuD_est_t", "energy_t", "dissipation_t", "ic_type_id")
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
# 5b. Sauvegarde des prédictions par fenêtre (format standard, tous modes)
# =============================================================================

def _subset_metrics(pred: np.ndarray, targ: np.ndarray) -> Dict[str, Any]:
    """Métriques de régression (spearman, pearson, R², rel_rmse, pente) en numpy/pandas."""
    pred = np.asarray(pred, dtype=float).reshape(-1)
    targ = np.asarray(targ, dtype=float).reshape(-1)
    if len(targ) < 2:
        return {"n": int(len(targ))}
    rx = pd.Series(pred).rank().to_numpy()
    ry = pd.Series(targ).rank().to_numpy()
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    pearson = float(np.corrcoef(pred, targ)[0, 1])
    ss_res = float(np.sum((pred - targ) ** 2))
    ss_tot = float(np.sum((targ - targ.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rel_rmse = float(np.sqrt(np.mean((pred - targ) ** 2)) / np.mean(np.abs(targ)))
    slope = float(np.polyfit(targ, pred, 1)[0])
    return {
        "n": int(len(targ)),
        "spearman": spearman,
        "pearson": pearson,
        "r2": r2,
        "r2_identity": r2,  # alias explicite : r2 (lignes 594-595) est une R2 identite 1-ss_res/ss_tot, pas affine (README Step 9)
        "rel_rmse": rel_rmse,
        "calibration_slope": slope,
    }


def range_stratified_metrics(
    z: np.ndarray,
    meta: Dict[str, np.ndarray],
    nuD_train_max: float,
    nu_value: float = 0.01,
) -> Dict[str, Any]:
    """Métriques z vs nu*D sur la strate nu=nu_value, stratifiées selon que nuD
    tombe dans la plage vue à l'entraînement (<= nuD_train_max) ou hors plage."""
    z = np.asarray(z, dtype=float).reshape(-1)
    nu = np.asarray(meta["nu"], dtype=float).reshape(-1)
    nd = np.asarray(meta["nu_dissipation_t"], dtype=float).reshape(-1)
    m = np.isclose(nu, nu_value)
    in_range = m & (nd <= nuD_train_max)
    out_range = m & (nd > nuD_train_max)
    return {
        "nu": float(nu_value),
        "nuD_train_max": float(nuD_train_max),
        "global": _subset_metrics(z[m], nd[m]),
        "in_range": _subset_metrics(z[in_range], nd[in_range]),
        "out_of_range": _subset_metrics(z[out_range], nd[out_range]),
        "frac_out_of_range": float(out_range.sum() / max(int(m.sum()), 1)),
    }


def save_predictions_npz(
    path: str | Path,
    *,
    z: np.ndarray,
    h: np.ndarray,
    meta: Dict[str, np.ndarray],
    u: Optional[np.ndarray] = None,
    dx: Optional[float] = None,
    split: str,
    protocol: str,
    seed: int,
    window_size: int,
    z_target_mode: str,
    run_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Sauvegarde les prédictions par fenêtre dans un npz au format standard.

    Champs : z (unités physiques), h, ic_type_id, E, D, max_abs_ux, nu, time_t,
    trajectory_index, time_index_t, energy0, nu_dissipation_t, nuD_est_t +
    scalaires (split, protocol, seed, window_size, z_target_mode, run_name).
    Écrit pour TOUS les modes, y compris z_target_mode="none".
    """
    path = Path(path)
    z = np.asarray(z, dtype=float).reshape(-1)
    n = len(z)
    if u is not None and dx is not None:
        max_abs_ux = np.abs(np.gradient(np.asarray(u, dtype=float), float(dx), axis=1)).max(axis=1)
    else:
        max_abs_ux = np.full(n, np.nan)
    ic = meta.get("ic_type_id")
    data: Dict[str, Any] = {
        "z": z,
        "h": np.asarray(h, dtype=float),
        "ic_type_id": np.asarray(ic if ic is not None else np.full(n, -1), dtype=np.int64),
        "E": np.asarray(meta["energy_t"], dtype=float),
        "D": np.asarray(meta["dissipation_t"], dtype=float),
        "max_abs_ux": np.asarray(max_abs_ux, dtype=float),
        "nu": np.asarray(meta["nu"], dtype=float),
        "time_t": np.asarray(meta["time_t"], dtype=float),
        "trajectory_index": np.asarray(meta["trajectory_index"], dtype=np.int64),
        "time_index_t": np.asarray(meta["time_index_t"], dtype=np.int64),
        "energy0": np.asarray(meta["energy0"], dtype=float),
        "nu_dissipation_t": np.asarray(meta["nu_dissipation_t"], dtype=float),
        "nuD_est_t": np.asarray(meta["nuD_est_t"], dtype=float),
        "split": np.array(str(split)),
        "protocol": np.array(str(protocol)),
        "seed": np.array(int(seed)),
        "window_size": np.array(int(window_size)),
        "z_target_mode": np.array(str(z_target_mode)),
        "run_name": np.array(str(run_name)),
    }
    if extra:
        data.update(extra)
    np.savez(path, **data)
    return path


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
    train_dataset_full: Any = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Baselines triviales pour z vs nu*D.

    - global_mean : moyenne globale sur le train FILTRÉ du protocole ;
    - nu_oracle   : moyenne par strate de nu calculée sur le TRAIN COMPLET
      (`train_dataset_full`, convention README:342 — oracle qui connaît nu,
      y compris la strate held-out) ; fallback sur le train filtré si non fourni.
    """
    mt = _load_train_experiment_c_module()
    baselines: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    for kind in ("global_mean", "nu_oracle"):
        src = train_dataset_full if (kind == "nu_oracle" and train_dataset_full is not None) else train_dataset
        train_targets = np.asarray(
            [[src.nu_dissipation[i, t]] for (i, t) in src.index_map],
            dtype=float,
        )
        train_nu = np.asarray([src.nu_values[i] for (i, _t) in src.index_map], dtype=float)
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
    z_target_mode: str = "none",
    verdict_version: str = "v1",
    nu_eval: float = 0.01,
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

    Version v2 (z_target_mode != "none", Experiment A-Physics) : le critère de
    monotonie est NOT APPLICABLE — νD n'est pas monotone à faible ν (mesuré sur
    test nu_sweep_v1 : 68.8% de pas montants à ν=0.01, 100% des trajectoires
    concernées ; une z parfaite alignée sur νD aurait monotone_fraction ≈ 0.64).
    structural_ok porte alors sur rang + rec RMSE uniquement.
    """
    nu_key = f"nu={nu_eval}"
    h = heldout_results["metrics"][nu_key]["nu_dissipation"]
    r_holdout = float(h["spearman"])
    r2_holdout = float(h["r2"])
    slope = float(h["calibration_slope"])
    rel_rmse_holdout = float(h["rel_rmse_mean"])
    oracle = float(
        heldout_results["baselines"]["nu_oracle"][nu_key]["nu_dissipation"]["rel_rmse_mean"]
    )
    beats_nu_oracle = bool(rel_rmse_holdout < oracle)

    s = standard_results
    effective_rank = float(s["latent_rank"]["effective_rank"])
    rec_rmse = float(s["reconstruction_rmse_test"])
    monotone = float(s["monotone_fraction_test"])
    if z_target_mode != "none":
        # v2 : z ancré sur νD (non monotone à faible ν) -> critère de monotonie
        # non applicable ; structural_ok = rang + rec RMSE.
        monotone_check = "not_applicable"
        struct_ok = bool(
            effective_rank >= thresholds["rank_min"]
            and rec_rmse <= thresholds["rec_rmse_max"]
        )
    else:
        monotone_check = "applied"
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
        "verdict_version": verdict_version,
        "z_target_mode": z_target_mode,
        "nu_eval": float(nu_eval),
        "thresholds": dict(thresholds),
        "r_holdout": r_holdout,
        "r2_holdout_nu001": r2_holdout,
        "calibration_slope_nu001": slope,
        "rel_rmse_holdout_nu001": rel_rmse_holdout,
        "baseline_nu_oracle_rel_rmse_nu001": oracle,
        "beats_nu_oracle": beats_nu_oracle,
        "effective_rank": effective_rank,
        "reconstruction_rmse_test": rec_rmse,
        "monotone_fraction_test": monotone,
        "monotone_check": monotone_check,
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
    z_target_mode: str = "none",
    z_weight: float = 0.0,
    standardize_z_target: bool = True,
    monotone_rtol: float = 0.0,
    eval_times: Sequence[float] = (1.0, 2.0),
    nu_strat: float = 0.01,
    seed: int = 42,
    device: Optional[str] = None,
    num_workers: int = 0,
    print_every: int = 10,
    verbose: bool = True,
    encoder_mode: str = "td0",
    run_tag: Optional[str] = None,
    num_threads: int = 4,
    require_clean_tree: bool = True,
    smoke: bool = False,
    expected_train_nu: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Pipeline window-A pour un protocole : datasets -> entraînement (early
    stopping) -> meilleur checkpoint -> prédiction test -> métriques z vs nu*D
    + baselines + rang latent + rec RMSE + monotonie -> sauvegarde.

    `z_target_mode` (Experiment A-Physics) : "none" (window-A historique),
    "energy_centered" (z ancré sur nuD_est_t = différence centrée de E,
    auto-supervisé) ou "supervised" (z ancré sur nu_dissipation_t stocké).

    `encoder_mode` (expérience TD, prereg 515cf5b) : "td0" = encodeur Conv1D
    actuel (défaut, inchangé) ; "td1"/"td2"/"td3" = encodeurs TD (différences
    temporelles, GAP, canal produit). `run_tag` (ex. "td2") préfixe le suffixe
    de run ; None = nommage historique `w{w}_{protocole}`.
    `require_clean_tree` : refuse le run si le working tree est modifié
    (règle 0.3, sauf smoke=True).
    `expected_train_nu` : si fourni, ensemble ν attendu pour train/val ; le run
    est REFUSÉ avant tout entraînement si l'ensemble ν effectivement filtré
    (train_ds.nu_values) diffère (correction bug filtre de protocole). L'ensemble
    effectif et le standardiseur de cible sont consignés dans `traceability`."""
    if protocol not in ("standard", "heldout_nu", "heldout_nu005"):
        raise ValueError("`protocol` doit être 'standard', 'heldout_nu' ou 'heldout_nu005'.")

    set_global_seed(seed, num_threads=num_threads)
    traceability = git_traceability()
    traceability["start_time"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    assert_clean_working_tree(traceability, allow_smoke=smoke or not require_clean_tree)
    if verbose:
        print(f"[traçabilité] git={traceability['git_hash']} dirty={traceability['working_tree_dirty']} "
              f"torch={traceability['torch_version']} numpy={traceability['numpy_version']} "
              f"threads={traceability['num_threads']}")

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
        z_weight=z_weight,
    )

    # 1) datasets
    nu_filter_train = tuple(float(v) for v in heldout_nu_values) if protocol.startswith("heldout") else None
    bundle = build_a_seq_datasets(
        train_path, val_path, test_path,
        window_size=window_size,
        nu_filter_train=nu_filter_train,
    )
    train_ds = bundle["train"]
    nx = train_ds.nx

    # Garde-fou protocole (correction bug filtre) : ensemble ν effectif du
    # train/val == expected_train_nu, sinon refus AVANT tout entraînement.
    # L'ensemble effectif est lu sur index_map (paires réellement chargées,
    # déjà filtrées par nu_mask) — PAS sur train_ds.nu_values (tableau brut
    # non filtré des 600 trajectoires).
    train_val_nu_actual = sorted({float(train_ds.nu_values[i]) for (i, _t) in train_ds.index_map})
    if expected_train_nu is not None:
        expected = sorted(float(v) for v in expected_train_nu)
        if len(train_val_nu_actual) != len(expected) or not all(
            bool(np.isclose(a, b)) for a, b in zip(train_val_nu_actual, expected)
        ):
            raise ValueError(
                f"filtre nu incoherent : train/val effectif {train_val_nu_actual} "
                f"!= attendu {expected} (expected_train_nu). Run refuse avant entraînement."
            )
    traceability["train_val_nu"] = train_val_nu_actual
    e0_quartile_edges = np.quantile(
        np.asarray(train_ds.energy0, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    # Standardisation de la cible z (identique pour tous les modes) : stats fit sur
    # le split TRAIN du protocole courant uniquement. L'évaluation de z vs nu*D
    # reste en unités physiques (inverse transform de z).
    z_key = _z_target_key(z_target_mode)
    z_target_transform: Optional[Tuple[float, float]] = None
    z_target_standardizer = None
    if z_key is not None and standardize_z_target:
        if z_key == "nuD_est_t":
            arr = train_ds.energy  # (n_traj, n_times), énergie brute
            vals = np.empty(len(train_ds.index_map), dtype=float)
            for k, (i, t) in enumerate(train_ds.index_map):
                tp = t + train_ds.stride
                if t >= 1:
                    vals[k] = (arr[i, t - 1] - arr[i, tp]) / (2.0 * train_ds.dt)
                else:
                    vals[k] = (arr[i, t] - arr[i, tp]) / train_ds.dt
        else:
            arr = train_ds.nu_dissipation
            vals = np.asarray([arr[i, t] for (i, t) in train_ds.index_map], dtype=float)
        z_mean = float(np.mean(vals))
        z_std = float(np.std(vals))
        z_target_transform = (z_mean, max(z_std, 1e-8))
        z_target_standardizer = {"mean": z_mean, "std": z_std, "fit_on": "train", "key": z_key}
    traceability["z_target_standardizer"] = z_target_standardizer

    # 2) modèle + entraînement
    if encoder_mode == "td0":
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
    else:
        model = ec_models.build_window_dynamics_autoencoder_td(
            encoder_mode=encoder_mode,
            window_size=window_size,
            nx=nx,
            dt=float(train_ds.dt),
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
        "encoder_mode": encoder_mode,
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
        bundle, batch_size=batch_size, num_workers=num_workers, seed=seed
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
        z_target_mode=z_target_mode,
        z_target_transform=z_target_transform,
        monotone_rtol=monotone_rtol,
        print_every=print_every,
        verbose=verbose,
    )

    load_best_model_a_seq(model, train_result["best_model_path"], device_obj)

    # 3) prédiction sur le test complet
    pred = predict_a_seq(model, bundle["test"], device=device_obj, batch_size=batch_size, num_workers=num_workers)
    z, h, rec, u, meta = pred["z"], pred["h"], pred["rec"], pred["u"], pred["meta"]
    # z en unités PHYSIQUES pour l'évaluation (inverse de la standardisation de cible)
    if z_target_transform is not None:
        z_mean, z_std = z_target_transform
        z = z * z_std + z_mean

    reconstruction_rmse_test = float(np.sqrt(np.mean((rec - u) ** 2)))

    # rang latent (PCA de h sur le test)
    pca = models_module.latent_pca_stats(torch.as_tensor(h, dtype=torch.float32))
    latent_rank = {
        "effective_rank": float(pca["effective_rank"]),
        "eigenvalues": [float(v) for v in np.asarray(pca["eigenvalues"]).reshape(-1)],
        "max_offline_residual": float(pca["max_offline_residual"]),
    }

    # loss + monotonie sur le test (paires)
    test_eval = evaluate_loss_a_seq(model, test_loader, device_obj, loss_kwargs, z_target_mode, z_target_transform, monotone_rtol)
    monotone_fraction_test = float(test_eval["monotone_fraction"])

    # 4) métriques z vs nu*D + baselines
    model_label = f"window_a_w{window_size}"
    nested, rows = evaluate_z_vs_nuD(
        z, meta, e0_quartile_edges, eval_times, protocol, model_label
    )
    baselines, baseline_rows = baseline_z_vs_nuD(
        train_ds, meta, e0_quartile_edges, eval_times, protocol,
        train_dataset_full=bundle["train_full"],
    )

    # stratification de la strate nu=0.01 selon la plage de nuD vue à l'entraînement
    # (population exacte des paires train du protocole courant)
    nuD_train_max = float(max(train_ds.nu_dissipation[i, t] for (i, t) in train_ds.index_map))
    range_strat = range_stratified_metrics(z, meta, nuD_train_max, nu_value=nu_strat)

    # 5) sauvegarde
    traceability["end_time"] = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    results: Dict[str, Any] = {
        "experiment": "experiment_a_seq",
        "protocol": protocol,
        "window_size": int(window_size),
        "encoder_mode": encoder_mode,
        "run_tag": run_tag,
        "num_threads": int(num_threads),
        "traceability": traceability,
        "heldout_nu_values": list(heldout_nu_values) if protocol.startswith("heldout") else None,
        "z_target_mode": z_target_mode,
        "standardize_z_target": bool(standardize_z_target and z_key is not None),
        "z_target_standardizer": z_target_standardizer,
        "monotone_rtol": float(monotone_rtol),
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
        "nuD_range_stratification": range_strat,
    }

    # suffixe de run : "w{w}_{protocol}" (+ "_z{mode}" si terme physique actif)
    # run_tag (expérience TD) remplace le préfixe "w{w}" : "td{bras}_{protocol}"
    run_suffix = f"{run_tag}_{protocol}" if run_tag else f"w{window_size}_{protocol}"
    if z_target_mode != "none":
        run_suffix += f"_z{z_target_mode}"
    metrics_json_path = output_dir / f"metrics_window_a_{run_suffix}.json"
    save_json(metrics_json_path, results)
    pd.DataFrame(rows + baseline_rows).to_csv(
        output_dir / f"metrics_long_window_a_{run_suffix}.csv", index=False
    )
    save_json(
        output_dir / "standardizers.json",
        {
            "input_standardizer": bundle["input_standardizer"].to_dict(),
            "z_target_standardizer": z_target_standardizer,
        },
    )
    # prédictions par fenêtre (tous modes, y compris z_target_mode="none") :
    # z en unités physiques, pour bootstrap a posteriori par trajectoire
    save_predictions_npz(
        output_dir / "predictions_test.npz",
        z=z, h=h, meta=meta, u=u, dx=train_ds.dx,
        split="test", protocol=protocol, seed=seed,
        window_size=window_size, z_target_mode=z_target_mode,
        run_name=f"{run_suffix}_s{seed}",
        extra={"checkpoint_path": np.array(str(train_result["best_model_path"]))},
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
