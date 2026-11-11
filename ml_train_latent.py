"""
ml/train_latent.py
==================

Entraînement de modèles latents pour le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module fournit un pipeline clair et progressif pour entraîner deux familles
principales de modèles latents :

1. un autoencodeur latent avec tête scalaire optionnelle,
2. un modèle de dynamique latente avec prédiction h_t -> h_{t+1}.

Ces modèles peuvent être entraînés avec des pertes combinant :
- reconstruction,
- prédiction latente,
- monotonie d'une observable scalaire apprise,
- régularisation de variance pour éviter le collapse.

Philosophie
-----------
- code lisible et abondamment commenté,
- séparation nette entre :
  * chargement des modules,
  * préparation des données,
  * boucle d'entraînement,
  * évaluation,
  * sauvegarde des résultats,
- robustesse aux imports dans un prototype encore en construction.

Remarque
--------
Ce module est écrit pour fonctionner dans le contexte actuel du prototype, même
si tous les fichiers ne sont pas encore importables comme un package Python
classique. Les imports critiques sont donc résolus de manière flexible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import importlib.util
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# =============================================================================
# 1. Chargement flexible des modules du projet
# =============================================================================

def _load_module_from_candidates(module_name: str, candidates: List[str]):
    """
    Charge dynamiquement un module depuis une liste de chemins candidats.

    Paramètres
    ----------
    module_name : str
        Nom logique du module à charger.
    candidates : list[str]
        Liste de chemins fichiers à tester, dans l'ordre.

    Retour
    ------
    module
        Module Python chargé.
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
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module

    raise ModuleNotFoundError(
        f"Impossible de charger le module {module_name!r}. Candidats testés : {candidates}"
    )



def load_project_symbols():
    """
    Charge les symboles nécessaires à l'entraînement latent.

    Retour
    ------
    tuple
        Contient notamment :
        - get_default_config
        - BurgersSnapshotDataset
        - BurgersPairDataset
        - LatentAutoencoder
        - LatentDynamicsModel
        - summarize_model
        - autoencoder_total_loss
        - latent_dynamics_total_loss
        - describe_dataset
    """
    # config
    try:
        from config import get_default_config  # type: ignore
    except ModuleNotFoundError:
        config_module = _load_module_from_candidates('config', ['config.py'])
        get_default_config = config_module.get_default_config

    datasets_module = _load_module_from_candidates(
        'ml_datasets', ['ml_datasets.py', 'ml/datasets.py']
    )
    models_module = _load_module_from_candidates(
        'ml_models', ['ml_models.py', 'ml/models.py']
    )
    losses_module = _load_module_from_candidates(
        'ml_losses', ['ml_losses.py', 'ml/losses.py']
    )

    return (
        get_default_config,
        datasets_module.BurgersSnapshotDataset,
        datasets_module.BurgersPairDataset,
        datasets_module.describe_dataset,
        models_module.LatentAutoencoder,
        models_module.LatentDynamicsModel,
        models_module.summarize_model,
        losses_module.autoencoder_total_loss,
        losses_module.latent_dynamics_total_loss,
    )


# =============================================================================
# 2. Utilitaires généraux
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Fixe les graines principales pour la reproductibilité."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def ensure_dir(path: str | Path) -> Path:
    """Crée un dossier s'il n'existe pas et renvoie le Path correspondant."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    """Sauvegarde un dictionnaire dans un fichier JSON lisible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def print_section(title: str) -> None:
    """Affiche un titre lisible dans le terminal."""
    line = '=' * len(title)
    print(f"\n{title}\n{line}")


# =============================================================================
# 3. Construction des DataLoaders
# =============================================================================

def build_autoencoder_dataloaders(
    train_dataset_path: str | Path,
    val_dataset_path: str | Path,
    batch_size: int,
    normalize: bool = False,
    num_workers: int = 0,
):
    """
    Construit les DataLoaders pour l'apprentissage de type autoencodeur.

    Chaque item correspond à un snapshot individuel via `BurgersSnapshotDataset`.
    """
    _, BurgersSnapshotDataset, _, _, _, _, _, _, _ = load_project_symbols()

    train_dataset = BurgersSnapshotDataset(
        train_dataset_path,
        normalize=normalize,
        standardizer=None,
        return_metadata=True,
    )
    val_dataset = BurgersSnapshotDataset(
        val_dataset_path,
        normalize=normalize,
        standardizer=train_dataset.standardizer,
        return_metadata=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_dataset, val_dataset



def build_dynamics_dataloaders(
    train_dataset_path: str | Path,
    val_dataset_path: str | Path,
    batch_size: int,
    normalize: bool = False,
    stride: int = 1,
    num_workers: int = 0,
):
    """
    Construit les DataLoaders pour l'apprentissage de dynamique latente.

    Chaque item correspond à une paire temporelle via `BurgersPairDataset`.
    """
    _, _, BurgersPairDataset, _, _, _, _, _, _ = load_project_symbols()

    train_dataset = BurgersPairDataset(
        train_dataset_path,
        normalize=normalize,
        standardizer=None,
        stride=stride,
        return_metadata=True,
    )
    val_dataset = BurgersPairDataset(
        val_dataset_path,
        normalize=normalize,
        standardizer=train_dataset.standardizer,
        stride=stride,
        return_metadata=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_dataset, val_dataset


# =============================================================================
# 4. Boucle autoencodeur latent
# =============================================================================

def train_one_epoch_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reconstruction_weight: float = 1.0,
    monotonicity_weight: float = 0.0,
    variance_weight: float = 0.0,
    monotonicity_mode: str = 'squared_hinge',
    min_scalar_variance: float = 1.0e-3,
) -> Dict[str, float]:
    """
    Exécute une époque d'entraînement pour un autoencodeur latent.

    Stratégie
    ---------
    - on entraîne sur des snapshots individuels,
    - la reconstruction est toujours active,
    - si le modèle possède une tête scalaire, on peut ajouter une régularisation
      de variance batch-wise,
    - la monotonie n'est pas activée ici car il n'y a pas de paire temporelle.
    """
    _, _, _, _, _, _, _, autoencoder_total_loss, _ = load_project_symbols()

    model.train()
    total = {'total_loss': 0.0, 'reconstruction_loss': 0.0, 'monotonicity_loss': 0.0, 'variance_loss': 0.0}
    total_samples = 0

    for batch in dataloader:
        x = batch['u'].to(device)

        optimizer.zero_grad()
        outputs = model(x)

        scalar = outputs.get('scalar', None)
        losses = autoencoder_total_loss(
            reconstruction=outputs['reconstruction'],
            target=x,
            scalar_t=scalar,
            scalar_tp=None,
            reconstruction_weight=reconstruction_weight,
            monotonicity_weight=0.0,  # pas de paire temporelle ici
            variance_weight=variance_weight,
            monotonicity_mode=monotonicity_mode,
            min_scalar_variance=min_scalar_variance,
        )
        loss = losses['total_loss']
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_samples += batch_size
        for key in total.keys():
            total[key] += float(losses[key].item()) * batch_size

    return {key: value / max(total_samples, 1) for key, value in total.items()}


@torch.no_grad()
def evaluate_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    reconstruction_weight: float = 1.0,
    variance_weight: float = 0.0,
    monotonicity_mode: str = 'squared_hinge',
    min_scalar_variance: float = 1.0e-3,
) -> Dict[str, float]:
    """
    Évalue un autoencodeur latent sur un DataLoader.
    """
    _, _, _, _, _, _, _, autoencoder_total_loss, _ = load_project_symbols()

    model.eval()
    total = {'total_loss': 0.0, 'reconstruction_loss': 0.0, 'monotonicity_loss': 0.0, 'variance_loss': 0.0}
    total_samples = 0

    for batch in dataloader:
        x = batch['u'].to(device)
        outputs = model(x)

        scalar = outputs.get('scalar', None)
        losses = autoencoder_total_loss(
            reconstruction=outputs['reconstruction'],
            target=x,
            scalar_t=scalar,
            scalar_tp=None,
            reconstruction_weight=reconstruction_weight,
            monotonicity_weight=0.0,
            variance_weight=variance_weight,
            monotonicity_mode=monotonicity_mode,
            min_scalar_variance=min_scalar_variance,
        )

        batch_size = x.shape[0]
        total_samples += batch_size
        for key in total.keys():
            total[key] += float(losses[key].item()) * batch_size

    return {key: value / max(total_samples, 1) for key, value in total.items()}


# =============================================================================
# 5. Boucle dynamique latente
# =============================================================================

def train_one_epoch_dynamics(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 0.0,
    variance_weight: float = 0.0,
    prediction_mode: str = 'mse',
    monotonicity_mode: str = 'squared_hinge',
    min_scalar_variance: float = 1.0e-3,
) -> Dict[str, float]:
    """
    Exécute une époque d'entraînement pour un modèle de dynamique latente.
    """
    _, _, _, _, _, _, _, _, latent_dynamics_total_loss = load_project_symbols()

    model.train()
    total = {'total_loss': 0.0, 'prediction_loss': 0.0, 'monotonicity_loss': 0.0, 'variance_loss': 0.0}
    total_samples = 0

    for batch in dataloader:
        u_t = batch['u_t'].to(device)
        u_tp = batch['u_tp'].to(device)

        optimizer.zero_grad()
        outputs = model(u_t, u_tp)

        losses = latent_dynamics_total_loss(
            predicted_latent=outputs['h_tp_pred'],
            target_latent=outputs['h_tp'],
            z_t=outputs.get('z_t', None),
            z_tp=outputs.get('z_tp', None),
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            prediction_mode=prediction_mode,
            monotonicity_mode=monotonicity_mode,
            min_scalar_variance=min_scalar_variance,
        )
        loss = losses['total_loss']
        loss.backward()
        optimizer.step()

        batch_size = u_t.shape[0]
        total_samples += batch_size
        for key in total.keys():
            total[key] += float(losses[key].item()) * batch_size

    return {key: value / max(total_samples, 1) for key, value in total.items()}


@torch.no_grad()
def evaluate_dynamics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 0.0,
    variance_weight: float = 0.0,
    prediction_mode: str = 'mse',
    monotonicity_mode: str = 'squared_hinge',
    min_scalar_variance: float = 1.0e-3,
) -> Dict[str, float]:
    """
    Évalue un modèle de dynamique latente sur un DataLoader.
    """
    _, _, _, _, _, _, _, _, latent_dynamics_total_loss = load_project_symbols()

    model.eval()
    total = {'total_loss': 0.0, 'prediction_loss': 0.0, 'monotonicity_loss': 0.0, 'variance_loss': 0.0}
    total_samples = 0

    for batch in dataloader:
        u_t = batch['u_t'].to(device)
        u_tp = batch['u_tp'].to(device)

        outputs = model(u_t, u_tp)
        losses = latent_dynamics_total_loss(
            predicted_latent=outputs['h_tp_pred'],
            target_latent=outputs['h_tp'],
            z_t=outputs.get('z_t', None),
            z_tp=outputs.get('z_tp', None),
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            prediction_mode=prediction_mode,
            monotonicity_mode=monotonicity_mode,
            min_scalar_variance=min_scalar_variance,
        )

        batch_size = u_t.shape[0]
        total_samples += batch_size
        for key in total.keys():
            total[key] += float(losses[key].item()) * batch_size

    return {key: value / max(total_samples, 1) for key, value in total.items()}


# =============================================================================
# 6. Entraînement complet : autoencodeur latent
# =============================================================================

def train_latent_autoencoder(
    train_dataset_path: str | Path,
    val_dataset_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 50,
    latent_dim: int = 8,
    encoder_hidden_dims: Tuple[int, ...] = (128, 64),
    decoder_hidden_dims: Tuple[int, ...] = (64, 128),
    activation: str = 'relu',
    dropout: float = 0.0,
    use_scalar_head: bool = True,
    scalar_head_hidden_dims: Tuple[int, ...] = (),
    positive_scalar_output: bool = False,
    reconstruction_weight: float = 1.0,
    variance_weight: float = 0.0,
    min_scalar_variance: float = 1.0e-3,
    normalize: bool = False,
    device: Optional[str] = None,
    seed: int = 42,
    weight_decay: float = 0.0,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Entraîne un `LatentAutoencoder` sur des snapshots individuels.
    """
    (
        _, _, _, describe_dataset,
        LatentAutoencoder, _, summarize_model,
        _, _
    ) = load_project_symbols()

    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)

    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)

    train_loader, val_loader, train_dataset, val_dataset = build_autoencoder_dataloaders(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        batch_size=batch_size,
        normalize=normalize,
        num_workers=num_workers,
    )

    input_dim = int(train_dataset.nx)
    model = LatentAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        activation=activation,
        dropout=dropout,
        use_scalar_head=use_scalar_head,
        scalar_head_hidden_dims=scalar_head_hidden_dims,
        positive_scalar_output=positive_scalar_output,
    ).to(device_obj)

    model_summary = summarize_model(model).to_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {
        'train_total_loss': [],
        'train_reconstruction_loss': [],
        'train_variance_loss': [],
        'val_total_loss': [],
        'val_reconstruction_loss': [],
        'val_variance_loss': [],
    }

    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = Path(output_dir) / 'best_latent_autoencoder.pt'
    last_model_path = Path(output_dir) / 'last_latent_autoencoder.pt'

    print_section('Début de l\'entraînement latent : autoencodeur')
    print(f'Device                 : {device_obj}')
    print(f'Input dimension        : {input_dim}')
    print(f'Batch size             : {batch_size}')
    print(f'Learning rate          : {learning_rate}')
    print(f'Number of epochs       : {num_epochs}')
    print(f'Normalize snapshots    : {normalize}')
    print(f'Model summary          : {model_summary}')

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch_autoencoder(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device_obj,
            reconstruction_weight=reconstruction_weight,
            monotonicity_weight=0.0,
            variance_weight=variance_weight,
            monotonicity_mode='squared_hinge',
            min_scalar_variance=min_scalar_variance,
        )
        val_metrics = evaluate_autoencoder(
            model=model,
            dataloader=val_loader,
            device=device_obj,
            reconstruction_weight=reconstruction_weight,
            variance_weight=variance_weight,
            monotonicity_mode='squared_hinge',
            min_scalar_variance=min_scalar_variance,
        )

        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_reconstruction_loss'].append(train_metrics['reconstruction_loss'])
        history['train_variance_loss'].append(train_metrics['variance_loss'])
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
        history['val_variance_loss'].append(val_metrics['variance_loss'])

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"train_total={train_metrics['total_loss']:.6e} | "
            f"val_total={val_metrics['total_loss']:.6e} | "
            f"val_reconstruction={val_metrics['reconstruction_loss']:.6e}"
        )

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': {
                        'input_dim': input_dim,
                        'latent_dim': latent_dim,
                        'encoder_hidden_dims': list(encoder_hidden_dims),
                        'decoder_hidden_dims': list(decoder_hidden_dims),
                        'activation': activation,
                        'dropout': dropout,
                        'use_scalar_head': use_scalar_head,
                        'scalar_head_hidden_dims': list(scalar_head_hidden_dims),
                        'positive_scalar_output': positive_scalar_output,
                    },
                    'standardizer': train_dataset.standardizer.to_dict() if train_dataset.standardizer is not None else None,
                },
                best_model_path,
            )

    torch.save(
        {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config': {
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'encoder_hidden_dims': list(encoder_hidden_dims),
                'decoder_hidden_dims': list(decoder_hidden_dims),
                'activation': activation,
                'dropout': dropout,
                'use_scalar_head': use_scalar_head,
                'scalar_head_hidden_dims': list(scalar_head_hidden_dims),
                'positive_scalar_output': positive_scalar_output,
            },
            'standardizer': train_dataset.standardizer.to_dict() if train_dataset.standardizer is not None else None,
        },
        last_model_path,
    )

    summary = {
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
        'device': str(device_obj),
        'seed': int(seed),
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'num_epochs': int(num_epochs),
        'normalize': bool(normalize),
        'model_summary': model_summary,
        'train_dataset_description': describe_dataset(train_dataset_path),
        'val_dataset_description': describe_dataset(val_dataset_path),
        'training_mode': 'latent_autoencoder',
    }

    save_json(Path(output_dir) / 'history_latent_autoencoder.json', history)
    save_json(Path(output_dir) / 'training_summary_latent_autoencoder.json', summary)

    print_section('Fin de l\'entraînement latent : autoencodeur')
    print(f'Best epoch      : {best_epoch}')
    print(f'Best val loss   : {best_val_loss:.6e}')
    print(f'Best model path : {best_model_path}')
    print(f'Last model path : {last_model_path}')

    return {
        'history': history,
        'summary': summary,
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
    }


# =============================================================================
# 7. Entraînement complet : dynamique latente
# =============================================================================

def train_latent_dynamics_model(
    train_dataset_path: str | Path,
    val_dataset_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 50,
    latent_dim: int = 8,
    encoder_hidden_dims: Tuple[int, ...] = (128, 64),
    predictor_hidden_dims: Tuple[int, ...] = (64, 64),
    activation: str = 'relu',
    dropout: float = 0.0,
    use_scalar_head: bool = True,
    scalar_head_hidden_dims: Tuple[int, ...] = (),
    positive_scalar_output: bool = False,
    residual_predictor: bool = False,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 0.0,
    variance_weight: float = 0.0,
    prediction_mode: str = 'mse',
    monotonicity_mode: str = 'squared_hinge',
    min_scalar_variance: float = 1.0e-3,
    normalize: bool = False,
    stride: int = 1,
    device: Optional[str] = None,
    seed: int = 42,
    weight_decay: float = 0.0,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Entraîne un `LatentDynamicsModel` sur des paires temporelles.
    """
    (
        _, _, _, describe_dataset,
        _, LatentDynamicsModel, summarize_model,
        _, _
    ) = load_project_symbols()

    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)

    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)

    train_loader, val_loader, train_dataset, val_dataset = build_dynamics_dataloaders(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        batch_size=batch_size,
        normalize=normalize,
        stride=stride,
        num_workers=num_workers,
    )

    input_dim = int(train_dataset.nx)
    model = LatentDynamicsModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        predictor_hidden_dims=predictor_hidden_dims,
        activation=activation,
        dropout=dropout,
        use_scalar_head=use_scalar_head,
        scalar_head_hidden_dims=scalar_head_hidden_dims,
        positive_scalar_output=positive_scalar_output,
        residual_predictor=residual_predictor,
    ).to(device_obj)

    model_summary = summarize_model(model).to_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {
        'train_total_loss': [],
        'train_prediction_loss': [],
        'train_monotonicity_loss': [],
        'train_variance_loss': [],
        'val_total_loss': [],
        'val_prediction_loss': [],
        'val_monotonicity_loss': [],
        'val_variance_loss': [],
    }

    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = Path(output_dir) / 'best_latent_dynamics_model.pt'
    last_model_path = Path(output_dir) / 'last_latent_dynamics_model.pt'

    print_section('Début de l\'entraînement latent : dynamique')
    print(f'Device                 : {device_obj}')
    print(f'Input dimension        : {input_dim}')
    print(f'Batch size             : {batch_size}')
    print(f'Learning rate          : {learning_rate}')
    print(f'Number of epochs       : {num_epochs}')
    print(f'Normalize snapshots    : {normalize}')
    print(f'Stride                 : {stride}')
    print(f'Model summary          : {model_summary}')

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch_dynamics(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device_obj,
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            prediction_mode=prediction_mode,
            monotonicity_mode=monotonicity_mode,
            min_scalar_variance=min_scalar_variance,
        )
        val_metrics = evaluate_dynamics(
            model=model,
            dataloader=val_loader,
            device=device_obj,
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            prediction_mode=prediction_mode,
            monotonicity_mode=monotonicity_mode,
            min_scalar_variance=min_scalar_variance,
        )

        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_prediction_loss'].append(train_metrics['prediction_loss'])
        history['train_monotonicity_loss'].append(train_metrics['monotonicity_loss'])
        history['train_variance_loss'].append(train_metrics['variance_loss'])
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_prediction_loss'].append(val_metrics['prediction_loss'])
        history['val_monotonicity_loss'].append(val_metrics['monotonicity_loss'])
        history['val_variance_loss'].append(val_metrics['variance_loss'])

        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"train_total={train_metrics['total_loss']:.6e} | "
            f"val_total={val_metrics['total_loss']:.6e} | "
            f"val_prediction={val_metrics['prediction_loss']:.6e} | "
            f"val_mono={val_metrics['monotonicity_loss']:.6e}"
        )

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'model_config': {
                        'input_dim': input_dim,
                        'latent_dim': latent_dim,
                        'encoder_hidden_dims': list(encoder_hidden_dims),
                        'predictor_hidden_dims': list(predictor_hidden_dims),
                        'activation': activation,
                        'dropout': dropout,
                        'use_scalar_head': use_scalar_head,
                        'scalar_head_hidden_dims': list(scalar_head_hidden_dims),
                        'positive_scalar_output': positive_scalar_output,
                        'residual_predictor': residual_predictor,
                    },
                    'standardizer': train_dataset.standardizer.to_dict() if train_dataset.standardizer is not None else None,
                    'stride': int(stride),
                },
                best_model_path,
            )

    torch.save(
        {
            'epoch': num_epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'model_config': {
                'input_dim': input_dim,
                'latent_dim': latent_dim,
                'encoder_hidden_dims': list(encoder_hidden_dims),
                'predictor_hidden_dims': list(predictor_hidden_dims),
                'activation': activation,
                'dropout': dropout,
                'use_scalar_head': use_scalar_head,
                'scalar_head_hidden_dims': list(scalar_head_hidden_dims),
                'positive_scalar_output': positive_scalar_output,
                'residual_predictor': residual_predictor,
            },
            'standardizer': train_dataset.standardizer.to_dict() if train_dataset.standardizer is not None else None,
            'stride': int(stride),
        },
        last_model_path,
    )

    summary = {
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
        'device': str(device_obj),
        'seed': int(seed),
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'num_epochs': int(num_epochs),
        'normalize': bool(normalize),
        'stride': int(stride),
        'model_summary': model_summary,
        'train_dataset_description': describe_dataset(train_dataset_path),
        'val_dataset_description': describe_dataset(val_dataset_path),
        'training_mode': 'latent_dynamics',
    }

    save_json(Path(output_dir) / 'history_latent_dynamics.json', history)
    save_json(Path(output_dir) / 'training_summary_latent_dynamics.json', summary)

    print_section('Fin de l\'entraînement latent : dynamique')
    print(f'Best epoch      : {best_epoch}')
    print(f'Best val loss   : {best_val_loss:.6e}')
    print(f'Best model path : {best_model_path}')
    print(f'Last model path : {last_model_path}')

    return {
        'history': history,
        'summary': summary,
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
    }


# =============================================================================
# 7bis. Architecture A : symboles, époque, évaluation et monitoring
# =============================================================================

def load_architecture_a_symbols():
    """
    Charge les symboles nécessaires à l'Architecture A (autoencodeur de
    dynamique latente) SANS modifier `load_project_symbols`, dont le tuple
    de retour est déjà déballé par les fonctions existantes (rétrocompatibilité).

    Retour
    ------
    tuple
        - LatentDynamicsAutoencoder (ml_models)
        - latent_pca_stats (ml_models)
        - latent_dynamics_autoencoder_total_loss (ml_losses)
    """
    models_module = _load_module_from_candidates(
        'ml_models', ['ml_models.py', 'ml/models.py']
    )
    losses_module = _load_module_from_candidates(
        'ml_losses', ['ml_losses.py', 'ml/losses.py']
    )
    return (
        models_module.LatentDynamicsAutoencoder,
        models_module.latent_pca_stats,
        losses_module.latent_dynamics_autoencoder_total_loss,
    )


def train_one_epoch_dynamics_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    reconstruction_weight: float = 1.0,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
    variance_weight: float = 0.01,
    min_latent_variance: float = 0.01,
    monotonicity_mode: str = 'squared_hinge',
    prediction_mode: str = 'mse',
) -> Dict[str, float]:
    """
    Exécute une époque d'entraînement pour un `LatentDynamicsAutoencoder` (Architecture A).

    Stratégie
    ---------
    - paires temporelles (u_t, u_{t+1}) via `BurgersPairDataset`,
    - perte totale :
        λ_rec·MSE(û_t, u_t) + λ_pred·MSE(h_{t+1}^pred, h_{t+1})
        + λ_mono·hinge(z_t, z_tp) + λ_var·Σ_j relu(min_var − Var(h_t[:, j])),
    - la monotonie porte sur les encodages RÉELS z_t, z_tp (pas sur z_tp_pred),
    - le plancher de variance est PAR DIMENSION sur h_t.
    """
    _, _, latent_dynamics_autoencoder_total_loss = load_architecture_a_symbols()

    model.train()
    total = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'prediction_loss': 0.0,
        'monotonicity_loss': 0.0,
        'variance_loss': 0.0,
    }
    total_samples = 0

    for batch in dataloader:
        u_t = batch['u_t'].to(device)
        u_tp = batch['u_tp'].to(device)

        optimizer.zero_grad()
        outputs = model(u_t, u_tp)

        if outputs.get('z_t', None) is None or outputs.get('z_tp', None) is None:
            raise ValueError(
                "latent_dynamics_autoencoder_total_loss requiert une tête scalaire "
                "(use_scalar_head=True) pour fournir z_t et z_tp."
            )

        losses = latent_dynamics_autoencoder_total_loss(
            reconstruction=outputs['reconstruction'],
            target_u=u_t,
            predicted_latent=outputs['h_tp_pred'],
            target_latent=outputs['h_tp'],
            z_t=outputs['z_t'],
            z_tp=outputs['z_tp'],
            h_t=outputs['h_t'],
            reconstruction_weight=reconstruction_weight,
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            min_latent_variance=min_latent_variance,
            monotonicity_mode=monotonicity_mode,
            prediction_mode=prediction_mode,
        )
        loss = losses['total_loss']
        loss.backward()
        optimizer.step()

        batch_size = u_t.shape[0]
        total_samples += batch_size
        for key in total.keys():
            total[key] += float(losses[key].item()) * batch_size

    return {key: value / max(total_samples, 1) for key, value in total.items()}


@torch.no_grad()
def evaluate_dynamics_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    reconstruction_weight: float = 1.0,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
    variance_weight: float = 0.01,
    min_latent_variance: float = 0.01,
    monotonicity_mode: str = 'squared_hinge',
    prediction_mode: str = 'mse',
) -> Dict[str, Any]:
    """
    Évalue un `LatentDynamicsAutoencoder` sur un DataLoader.

    Retour
    ------
    Dict[str, Any]
        - les 5 termes de perte moyennés par échantillon,
        - 'monotone_fraction'  : fraction des paires avec z_{t+1} <= z_t,
        - 'corr_z_dissipation' : corrélation de Pearson entre z_t et la dissipation
          D_t (métadonnée `dissipation_t` du dataset), accumulée sur tout le
          DataLoader (None si non calculable, ex variance nulle).
    """
    _, _, latent_dynamics_autoencoder_total_loss = load_architecture_a_symbols()

    model.eval()
    total = {
        'total_loss': 0.0,
        'reconstruction_loss': 0.0,
        'prediction_loss': 0.0,
        'monotonicity_loss': 0.0,
        'variance_loss': 0.0,
    }
    total_samples = 0
    z_parts: List[np.ndarray] = []
    d_parts: List[np.ndarray] = []
    monotone_ok = 0
    monotone_count = 0

    for batch in dataloader:
        u_t = batch['u_t'].to(device)
        u_tp = batch['u_tp'].to(device)

        outputs = model(u_t, u_tp)
        losses = latent_dynamics_autoencoder_total_loss(
            reconstruction=outputs['reconstruction'],
            target_u=u_t,
            predicted_latent=outputs['h_tp_pred'],
            target_latent=outputs['h_tp'],
            z_t=outputs['z_t'],
            z_tp=outputs['z_tp'],
            h_t=outputs['h_t'],
            reconstruction_weight=reconstruction_weight,
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            min_latent_variance=min_latent_variance,
            monotonicity_mode=monotonicity_mode,
            prediction_mode=prediction_mode,
        )

        batch_size = u_t.shape[0]
        total_samples += batch_size
        for key in total.keys():
            total[key] += float(losses[key].item()) * batch_size

        z_t = outputs['z_t'].detach().reshape(-1).cpu().numpy().astype(np.float64)
        z_tp = outputs['z_tp'].detach().reshape(-1).cpu().numpy().astype(np.float64)
        z_parts.append(z_t)
        monotone_ok += int(np.sum(z_tp <= z_t))
        monotone_count += int(z_t.shape[0])
        if 'dissipation_t' in batch:
            d_parts.append(
                torch.as_tensor(batch['dissipation_t']).reshape(-1).cpu().numpy().astype(np.float64)
            )

    metrics: Dict[str, Any] = {key: value / max(total_samples, 1) for key, value in total.items()}
    metrics['monotone_fraction'] = float(monotone_ok / max(monotone_count, 1))

    if len(d_parts) > 0 and len(z_parts) > 0:
        z_cat = np.concatenate(z_parts)
        d_cat = np.concatenate(d_parts)
        if z_cat.shape[0] == d_cat.shape[0] and float(np.std(z_cat)) > 0.0 and float(np.std(d_cat)) > 0.0:
            metrics['corr_z_dissipation'] = float(np.corrcoef(z_cat, d_cat)[0, 1])
        else:
            metrics['corr_z_dissipation'] = None
    else:
        metrics['corr_z_dissipation'] = None

    return metrics


@torch.no_grad()
def compute_latent_monitoring_stats(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 1,
) -> Dict[str, Any]:
    """
    Statistiques de monitoring du latent h sur `max_batches` batch(s) de validation.

    Utilise `LatentDynamicsAutoencoder.monitoring_stats` (PCA de h + unités actives
    de la 2e couche de l'encodeur).

    Retour
    ------
    Dict[str, Any]
        Dictionnaire sérialisable en JSON :
        - 'effective_rank'          : rang effectif (95% de variance) de h,
        - 'max_offline_residual'    : résidu max hors 1re composante principale
          (détecteur d'effondrement exact rang 1),
        - 'alive_layer2_units'      : unités de la 2e couche linéaire de l'encodeur
          actives (pré-activation > 0) sur au moins un échantillon,
        - 'latent_variance_per_dim' : variance par dimension de h (liste),
        - 'pca_eigenvalues'         : spectre PCA de h (liste, ordre décroissant),
        - 'monitoring_batch_size'   : nombre d'échantillons utilisés.
    """
    model.eval()
    u_parts: List[torch.Tensor] = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        u_parts.append(batch['u_t'].to(device))
    if len(u_parts) == 0:
        return {}

    u_batch = torch.cat(u_parts, dim=0)
    mon = model.monitoring_stats(u_batch)
    return {
        'effective_rank': int(mon['effective_rank']),
        'max_offline_residual': float(mon['max_offline_residual']),
        'alive_layer2_units': int(mon['alive_layer2_units']),
        'latent_variance_per_dim': [float(v) for v in mon['latent_variance_per_dim'].tolist()],
        'pca_eigenvalues': [float(v) for v in mon['eigenvalues'].tolist()],
        'monitoring_batch_size': int(u_batch.shape[0]),
    }


# =============================================================================
# 7ter. Entraînement complet : autoencodeur de dynamique latente (Architecture A)
# =============================================================================

def _save_dynamics_autoencoder_checkpoint(
    path: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_val_loss: float,
    model_config: Dict[str, Any],
    standardizer: Optional[Dict[str, Any]],
    stride: int,
) -> None:
    """
    Sauvegarde un checkpoint au MÊME format top-level que les checkpoints existants
    ('epoch', 'model_state_dict', 'optimizer_state_dict', 'best_val_loss',
    'model_config', 'standardizer', 'stride') pour rester compatible avec le
    code d'évaluation. `model_config` inclut en plus 'model_type', 'stride' et
    'loss_config' (λ + min_latent_variance) pour reconstruire le modèle.
    """
    torch.save(
        {
            'epoch': int(epoch),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': float(best_val_loss),
            'model_config': model_config,
            'standardizer': standardizer,
            'stride': int(stride),
        },
        path,
    )


def train_latent_dynamics_autoencoder(
    train_dataset_path: str | Path,
    val_dataset_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 64,
    learning_rate: float = 1.0e-3,
    num_epochs: int = 50,
    latent_dim: int = 8,
    encoder_hidden_dims: Tuple[int, ...] = (128, 64),
    decoder_hidden_dims: Tuple[int, ...] = (64, 128),
    predictor_hidden_dims: Tuple[int, ...] = (64, 64),
    activation: str = 'relu',
    dropout: float = 0.0,
    use_scalar_head: bool = True,
    scalar_head_hidden_dims: Tuple[int, ...] = (),
    positive_scalar_output: bool = False,
    residual_predictor: bool = False,
    reconstruction_weight: float = 1.0,
    prediction_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
    variance_weight: float = 0.01,
    min_latent_variance: float = 0.01,
    prediction_mode: str = 'mse',
    monotonicity_mode: str = 'squared_hinge',
    normalize: bool = False,
    stride: int = 1,
    device: Optional[str] = None,
    seed: int = 42,
    weight_decay: float = 0.0,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """
    Entraîne un `LatentDynamicsAutoencoder` (Architecture A) sur des paires temporelles.

    Architecture A
    --------------
    L = λ_rec·MSE(û_t, u_t) + λ_pred·MSE(h_{t+1}^pred, h_{t+1})
      + λ_mono·mean(relu(z_{t+1} − z_t)²) + λ_var·Σ_j relu(min_var − Var(h_t[:, j]))

    avec λ_rec = λ_pred = λ_mono = 1.0, λ_var = 0.01, min_var = 0.01 par défaut.

    Monitoring par époque (1 batch de val) : rang effectif, spectre PCA,
    variance par dimension de h, unités actives couche 2, fraction de monotonie
    val, corrélation corr(z, D) — le tout sauvegardé dans le history JSON.
    """
    (
        _, _, _, describe_dataset,
        _, _, summarize_model,
        _, _
    ) = load_project_symbols()
    LatentDynamicsAutoencoder, _, _ = load_architecture_a_symbols()

    set_global_seed(seed)
    output_dir = ensure_dir(output_dir)

    if device is None:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device_obj = torch.device(device)

    train_loader, val_loader, train_dataset, val_dataset = build_dynamics_dataloaders(
        train_dataset_path=train_dataset_path,
        val_dataset_path=val_dataset_path,
        batch_size=batch_size,
        normalize=normalize,
        stride=stride,
        num_workers=num_workers,
    )

    input_dim = int(train_dataset.nx)
    model = LatentDynamicsAutoencoder(
        input_dim=input_dim,
        latent_dim=latent_dim,
        encoder_hidden_dims=encoder_hidden_dims,
        decoder_hidden_dims=decoder_hidden_dims,
        predictor_hidden_dims=predictor_hidden_dims,
        activation=activation,
        dropout=dropout,
        use_scalar_head=use_scalar_head,
        scalar_head_hidden_dims=scalar_head_hidden_dims,
        positive_scalar_output=positive_scalar_output,
        residual_predictor=residual_predictor,
    ).to(device_obj)

    model_summary = summarize_model(model).to_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    standardizer_dict = (
        train_dataset.standardizer.to_dict() if train_dataset.standardizer is not None else None
    )
    model_config: Dict[str, Any] = {
        'model_type': 'latent_dynamics_autoencoder',
        'input_dim': input_dim,
        'latent_dim': latent_dim,
        'encoder_hidden_dims': list(encoder_hidden_dims),
        'decoder_hidden_dims': list(decoder_hidden_dims),
        'predictor_hidden_dims': list(predictor_hidden_dims),
        'activation': activation,
        'dropout': dropout,
        'use_scalar_head': use_scalar_head,
        'scalar_head_hidden_dims': list(scalar_head_hidden_dims),
        'positive_scalar_output': positive_scalar_output,
        'residual_predictor': residual_predictor,
        'stride': int(stride),
        'loss_config': {
            'reconstruction_weight': float(reconstruction_weight),
            'prediction_weight': float(prediction_weight),
            'monotonicity_weight': float(monotonicity_weight),
            'variance_weight': float(variance_weight),
            'min_latent_variance': float(min_latent_variance),
            'monotonicity_mode': monotonicity_mode,
            'prediction_mode': prediction_mode,
        },
    }

    history: Dict[str, List[Any]] = {
        'train_total_loss': [],
        'train_reconstruction_loss': [],
        'train_prediction_loss': [],
        'train_monotonicity_loss': [],
        'train_variance_loss': [],
        'val_total_loss': [],
        'val_reconstruction_loss': [],
        'val_prediction_loss': [],
        'val_monotonicity_loss': [],
        'val_variance_loss': [],
        'val_monotone_fraction': [],
        'val_corr_z_dissipation': [],
        'val_effective_rank': [],
        'val_max_offline_residual': [],
        'val_alive_layer2_units': [],
        'val_latent_variance_per_dim': [],
        'val_pca_eigenvalues': [],
    }

    best_val_loss = float('inf')
    best_epoch = -1
    best_model_path = Path(output_dir) / 'best_latent_dynamics_autoencoder.pt'
    last_model_path = Path(output_dir) / 'last_latent_dynamics_autoencoder.pt'

    print_section('Début de l\'entraînement latent : autoencodeur de dynamique (Architecture A)')
    print(f'Device                 : {device_obj}')
    print(f'Input dimension        : {input_dim}')
    print(f'Batch size             : {batch_size}')
    print(f'Learning rate          : {learning_rate}')
    print(f'Number of epochs       : {num_epochs}')
    print(f'Normalize snapshots    : {normalize}')
    print(f'Stride                 : {stride}')
    print(f'λ_rec / λ_pred         : {reconstruction_weight} / {prediction_weight}')
    print(f'λ_mono / λ_var         : {monotonicity_weight} / {variance_weight}')
    print(f'Min var (par dim, h)   : {min_latent_variance}')
    print(f'Model summary          : {model_summary}')

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch_dynamics_autoencoder(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device_obj,
            reconstruction_weight=reconstruction_weight,
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            min_latent_variance=min_latent_variance,
            monotonicity_mode=monotonicity_mode,
            prediction_mode=prediction_mode,
        )
        val_metrics = evaluate_dynamics_autoencoder(
            model=model,
            dataloader=val_loader,
            device=device_obj,
            reconstruction_weight=reconstruction_weight,
            prediction_weight=prediction_weight,
            monotonicity_weight=monotonicity_weight,
            variance_weight=variance_weight,
            min_latent_variance=min_latent_variance,
            monotonicity_mode=monotonicity_mode,
            prediction_mode=prediction_mode,
        )
        monitoring = compute_latent_monitoring_stats(model, val_loader, device_obj, max_batches=1)

        history['train_total_loss'].append(train_metrics['total_loss'])
        history['train_reconstruction_loss'].append(train_metrics['reconstruction_loss'])
        history['train_prediction_loss'].append(train_metrics['prediction_loss'])
        history['train_monotonicity_loss'].append(train_metrics['monotonicity_loss'])
        history['train_variance_loss'].append(train_metrics['variance_loss'])
        history['val_total_loss'].append(val_metrics['total_loss'])
        history['val_reconstruction_loss'].append(val_metrics['reconstruction_loss'])
        history['val_prediction_loss'].append(val_metrics['prediction_loss'])
        history['val_monotonicity_loss'].append(val_metrics['monotonicity_loss'])
        history['val_variance_loss'].append(val_metrics['variance_loss'])
        history['val_monotone_fraction'].append(val_metrics['monotone_fraction'])
        history['val_corr_z_dissipation'].append(val_metrics['corr_z_dissipation'])
        history['val_effective_rank'].append(monitoring.get('effective_rank'))
        history['val_max_offline_residual'].append(monitoring.get('max_offline_residual'))
        history['val_alive_layer2_units'].append(monitoring.get('alive_layer2_units'))
        history['val_latent_variance_per_dim'].append(monitoring.get('latent_variance_per_dim'))
        history['val_pca_eigenvalues'].append(monitoring.get('pca_eigenvalues'))

        corr_z_d = val_metrics['corr_z_dissipation']
        corr_str = 'nan' if corr_z_d is None else f'{corr_z_d:.4f}'
        print(
            f"[Epoch {epoch:03d}/{num_epochs:03d}] "
            f"train_total={train_metrics['total_loss']:.6e} | "
            f"val_total={val_metrics['total_loss']:.6e} | "
            f"val_rec={val_metrics['reconstruction_loss']:.6e} | "
            f"val_pred={val_metrics['prediction_loss']:.6e} | "
            f"val_mono={val_metrics['monotonicity_loss']:.6e} | "
            f"rank={monitoring.get('effective_rank')} | "
            f"alive_L2={monitoring.get('alive_layer2_units')} | "
            f"mono_frac={val_metrics['monotone_fraction']:.4f} | "
            f"corr(z,D)={corr_str}"
        )

        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            best_epoch = epoch
            _save_dynamics_autoencoder_checkpoint(
                best_model_path, epoch, model, optimizer, best_val_loss,
                model_config, standardizer_dict, stride,
            )

    _save_dynamics_autoencoder_checkpoint(
        last_model_path, num_epochs, model, optimizer, best_val_loss,
        model_config, standardizer_dict, stride,
    )

    summary = {
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
        'device': str(device_obj),
        'seed': int(seed),
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'num_epochs': int(num_epochs),
        'normalize': bool(normalize),
        'stride': int(stride),
        'loss_config': model_config['loss_config'],
        'model_summary': model_summary,
        'train_dataset_description': describe_dataset(train_dataset_path),
        'val_dataset_description': describe_dataset(val_dataset_path),
        'training_mode': 'latent_dynamics_autoencoder',
    }

    save_json(Path(output_dir) / 'history_latent_dynamics_autoencoder.json', history)
    save_json(Path(output_dir) / 'training_summary_latent_dynamics_autoencoder.json', summary)

    print_section('Fin de l\'entraînement latent : autoencodeur de dynamique (Architecture A)')
    print(f'Best epoch      : {best_epoch}')
    print(f'Best val loss   : {best_val_loss:.6e}')
    print(f'Best model path : {best_model_path}')
    print(f'Last model path : {last_model_path}')

    return {
        'history': history,
        'summary': summary,
        'best_model_path': str(best_model_path),
        'last_model_path': str(last_model_path),
    }


# =============================================================================
# 8. Interfaces pratiques à partir de la config
# =============================================================================

def train_latent_autoencoder_from_config(config: Any, output_subdir: str = 'latent_autoencoder') -> Dict[str, Any]:
    """
    Lance l'entraînement de l'autoencodeur latent à partir de la configuration globale.
    """
    train_path = Path(config.paths.data_dir) / config.dataset.train_filename
    val_path = Path(config.paths.data_dir) / config.dataset.val_filename
    output_dir = Path(config.paths.models_dir) / output_subdir

    return train_latent_autoencoder(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        output_dir=output_dir,
        batch_size=int(config.ml.batch_size),
        learning_rate=float(config.ml.learning_rate),
        num_epochs=int(config.ml.num_epochs),
        latent_dim=int(config.ml.latent_dim),
        encoder_hidden_dims=(128, 64),
        decoder_hidden_dims=(64, 128),
        activation='relu',
        dropout=0.0,
        use_scalar_head=True,
        scalar_head_hidden_dims=(),
        positive_scalar_output=False,
        reconstruction_weight=float(getattr(config.losses, 'reconstruction_weight', 1.0)) if hasattr(config, 'losses') else 1.0,
        variance_weight=float(getattr(config.losses, 'variance_weight', 0.0)) if hasattr(config, 'losses') else 0.0,
        min_scalar_variance=float(getattr(config.losses, 'min_latent_variance', 1.0e-3)) if hasattr(config, 'losses') else 1.0e-3,
        normalize=bool(getattr(config.normalization, 'enabled', False)) if hasattr(config, 'normalization') else False,
        device=str(config.ml.device) if hasattr(config, 'ml') else None,
        seed=int(config.reproducibility.seed),
        weight_decay=float(getattr(config.ml, 'weight_decay', 0.0)),
        num_workers=0,
    )



def train_latent_dynamics_from_config(config: Any, output_subdir: str = 'latent_dynamics', stride: int = 1) -> Dict[str, Any]:
    """
    Lance l'entraînement du modèle de dynamique latente à partir de la configuration globale.
    """
    train_path = Path(config.paths.data_dir) / config.dataset.train_filename
    val_path = Path(config.paths.data_dir) / config.dataset.val_filename
    output_dir = Path(config.paths.models_dir) / output_subdir

    return train_latent_dynamics_model(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        output_dir=output_dir,
        batch_size=int(config.ml.batch_size),
        learning_rate=float(config.ml.learning_rate),
        num_epochs=int(config.ml.num_epochs),
        latent_dim=int(config.ml.latent_dim),
        encoder_hidden_dims=(128, 64),
        predictor_hidden_dims=(64, 64),
        activation='relu',
        dropout=0.0,
        use_scalar_head=True,
        scalar_head_hidden_dims=(),
        positive_scalar_output=False,
        residual_predictor=False,
        prediction_weight=float(getattr(config.losses, 'prediction_weight', 1.0)) if hasattr(config, 'losses') else 1.0,
        monotonicity_weight=float(getattr(config.losses, 'monotonicity_weight', 0.0)) if hasattr(config, 'losses') else 0.0,
        variance_weight=float(getattr(config.losses, 'variance_weight', 0.0)) if hasattr(config, 'losses') else 0.0,
        prediction_mode='mse',
        monotonicity_mode=str(getattr(config.losses, 'monotonicity_mode', 'squared_hinge')) if hasattr(config, 'losses') else 'squared_hinge',
        min_scalar_variance=float(getattr(config.losses, 'min_latent_variance', 1.0e-3)) if hasattr(config, 'losses') else 1.0e-3,
        normalize=bool(getattr(config.normalization, 'enabled', False)) if hasattr(config, 'normalization') else False,
        stride=int(stride),
        device=str(config.ml.device) if hasattr(config, 'ml') else None,
        seed=int(config.reproducibility.seed),
        weight_decay=float(getattr(config.ml, 'weight_decay', 0.0)),
        num_workers=0,
    )


# =============================================================================
# 8bis. Interface config : autoencodeur de dynamique latente (Architecture A)
# =============================================================================

def train_latent_dynamics_autoencoder_from_config(
    config: Any,
    output_subdir: str = 'latent_dynamics_autoencoder',
    stride: int = 1,
) -> Dict[str, Any]:
    """
    Lance l'entraînement de l'autoencodeur de dynamique latente (Architecture A)
    à partir de la configuration globale.

    Les poids λ sont lus dans `config.losses` (champs existants — config.py
    inchangé). `min_latent_variance` vaut 0.01 : c'est le plancher PAR DIMENSION
    sur h de l'Architecture A, distinct de l'ancien plancher 1e-3 sur z
    (`config.losses.min_latent_variance`, conservé pour les modes existants).
    """
    train_path = Path(config.paths.data_dir) / config.dataset.train_filename
    val_path = Path(config.paths.data_dir) / config.dataset.val_filename
    output_dir = Path(config.paths.models_dir) / output_subdir

    return train_latent_dynamics_autoencoder(
        train_dataset_path=train_path,
        val_dataset_path=val_path,
        output_dir=output_dir,
        batch_size=int(config.ml.batch_size),
        learning_rate=float(config.ml.learning_rate),
        num_epochs=int(config.ml.num_epochs),
        latent_dim=int(config.ml.latent_dim),
        encoder_hidden_dims=(128, 64),
        decoder_hidden_dims=(64, 128),
        predictor_hidden_dims=(64, 64),
        activation='relu',
        dropout=0.0,
        use_scalar_head=True,
        scalar_head_hidden_dims=(),
        positive_scalar_output=False,
        residual_predictor=False,
        reconstruction_weight=float(getattr(config.losses, 'reconstruction_weight', 1.0)) if hasattr(config, 'losses') else 1.0,
        prediction_weight=float(getattr(config.losses, 'prediction_weight', 1.0)) if hasattr(config, 'losses') else 1.0,
        monotonicity_weight=float(getattr(config.losses, 'monotonicity_weight', 1.0)) if hasattr(config, 'losses') else 1.0,
        variance_weight=float(getattr(config.losses, 'variance_weight', 0.01)) if hasattr(config, 'losses') else 0.01,
        min_latent_variance=0.01,
        prediction_mode='mse',
        monotonicity_mode=str(getattr(config.losses, 'monotonicity_mode', 'squared_hinge')) if hasattr(config, 'losses') else 'squared_hinge',
        normalize=bool(getattr(config.normalization, 'enabled', False)) if hasattr(config, 'normalization') else False,
        stride=int(stride),
        device=str(config.ml.device) if hasattr(config, 'ml') else None,
        seed=int(config.reproducibility.seed),
        weight_decay=float(getattr(config.ml, 'weight_decay', 0.0)),
        num_workers=0,
    )


# =============================================================================
# 9. Bloc de test minimal
# =============================================================================

if __name__ == '__main__':
    print('ml/train_latent.py chargé avec succès.')
    print('Fonctions disponibles :')
    print('- build_autoencoder_dataloaders')
    print('- build_dynamics_dataloaders')
    print('- train_one_epoch_autoencoder')
    print('- evaluate_autoencoder')
    print('- train_one_epoch_dynamics')
    print('- evaluate_dynamics')
    print('- train_latent_autoencoder')
    print('- train_latent_dynamics_model')
    print('- train_latent_autoencoder_from_config')
    print('- train_latent_dynamics_from_config')
    print('- load_architecture_a_symbols')
    print('- train_one_epoch_dynamics_autoencoder')
    print('- evaluate_dynamics_autoencoder')
    print('- compute_latent_monitoring_stats')
    print('- train_latent_dynamics_autoencoder')
    print('- train_latent_dynamics_autoencoder_from_config')
