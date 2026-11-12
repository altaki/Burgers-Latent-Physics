"""
ml/models_experiment_c.py
=========================

Architectures PyTorch pour l'expérience C : baseline supervisée de
séparabilité des observables physiques sur le dataset `nu_sweep_v1`.

Trois variantes, toutes multi-tâches (encodeur partagé + 5 têtes scalaires) :

- C1 `ExperimentCMLP`       : snapshot u_t (B, Nx)      -> MLPEncoder      -> 5 têtes
- C2 `ExperimentCCNN`       : snapshot u_t (B, Nx)      -> Conv1DEncoder(1 canal)  -> 5 têtes
- C3 `ExperimentCWindowCNN` : fenêtre (B, w, Nx)        -> Conv1DEncoder(w canaux) -> 5 têtes

Cibles, dans l'ordre de `EXPERIMENT_C_TARGET_NAMES` (ml_datasets_experiment_c) :
    energy, nu_dissipation, dissipation, tv, max_abs_ux

Conventions
-----------
- Entrées  : (B, Nx) pour C1/C2, (B, window_size, Nx) pour C3.
- Sorties  : dict {"latent": (B, latent_dim), "preds": (B, 5)} où la colonne j
  de `preds` correspond à la j-ème cible de `EXPERIMENT_C_TARGET_NAMES`.
- Les cibles étant standardisées pendant l'entraînement, les têtes n'imposent
  pas de positivité (`positive_output=False`).

Réutilisation
-------------
Les briques existantes de `ml_models.py` sont réutilisées telles quelles :
`build_mlp`, `MLPEncoder`, `ScalarHead`, `get_activation`, `count_parameters`,
`summarize_model`. Seul l'encodeur convolutif `Conv1DEncoder` est nouveau
(aucune brique conv n'existait dans `ml_models.py`).

Ce module est additif : aucun fichier existant n'est modifié.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import importlib.util

import torch
import torch.nn as nn


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


def _load_models_module():
    """Charge `ml_models` (racine ou package `ml/`)."""
    return _load_module_from_candidates(
        module_name='ml_models',
        candidates=['ml_models.py', 'ml/models.py'],
    )


def _load_experiment_c_datasets_module():
    """Charge `ml_datasets_experiment_c` (source unique des noms de cibles)."""
    return _load_module_from_candidates(
        module_name='ml_datasets_experiment_c',
        candidates=['ml_datasets_experiment_c.py'],
    )


def _default_target_names() -> Tuple[str, ...]:
    """Noms des cibles par défaut, lus dans `ml_datasets_experiment_c`."""
    return tuple(_load_experiment_c_datasets_module().EXPERIMENT_C_TARGET_NAMES)


#: Variantes d'entrée disponibles pour l'expérience C.
EXPERIMENT_C_VARIANTS: Tuple[str, ...] = ("c1_mlp", "c2_cnn", "c3_window")


# =============================================================================
# 2. Encodeur convolutif 1D (nouvelle brique, absente de ml_models.py)
# =============================================================================

class Conv1DEncoder(nn.Module):
    """
    Encodeur 1D-CNN : (B, C_in, Nx) -> latent (B, latent_dim).

    Architecture
    ------------
    blocs [Conv1d -> activation -> (dropout)] x n, puis flatten et trunk MLP
    vers le latent.

    La longueur de sortie après les convolutions est calculée exactement en
    __init__ (formule des Conv1d), ce qui rend `flatten_dim` déterministe.
    """

    def __init__(
        self,
        in_channels: int,
        nx: int,
        latent_dim: int = 128,
        channels: Sequence[int] = (16, 32, 32, 64, 64),
        kernel_size: int = 5,
        stride: int = 2,
        trunk_dims: Sequence[int] = (256,),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        models_module = _load_models_module()

        if in_channels < 1 or nx < 1:
            raise ValueError("`in_channels` et `nx` doivent être >= 1.")
        if len(channels) < 1:
            raise ValueError("`channels` doit contenir au moins un élément.")

        self.in_channels = int(in_channels)
        self.nx = int(nx)
        self.latent_dim = int(latent_dim)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

        # blocs conv
        layers: List[nn.Module] = []
        c_in = self.in_channels
        length = self.nx
        padding = self.kernel_size // 2
        for c_out in self.channels:
            layers.append(nn.Conv1d(c_in, c_out, self.kernel_size, stride=self.stride, padding=padding))
            layers.append(models_module.get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            c_in = c_out
            length = (length + 2 * padding - self.kernel_size) // self.stride + 1

        self.conv = nn.Sequential(*layers)
        self.out_length = length
        self.flatten_dim = length * self.channels[-1]

        self.to_latent = models_module.build_mlp(
            input_dim=self.flatten_dim,
            hidden_dims=trunk_dims,
            output_dim=self.latent_dim,
            activation=activation,
            dropout=dropout,
            final_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paramètres
        ----------
        x : torch.Tensor
            Entrée de forme (B, C_in, Nx).

        Retour
        ------
        torch.Tensor
            Latent de forme (B, latent_dim).
        """
        feats = self.conv(x)                       # (B, C_out, L_out)
        flat = feats.reshape(feats.shape[0], -1)   # (B, flatten_dim)
        return self.to_latent(flat)


# =============================================================================
# 3. Modèle multi-tâches : encodeur partagé + une tête scalaire par cible
# =============================================================================

class ExperimentCMultiHeadModel(nn.Module):
    """
    Encodeur partagé + 5 têtes scalaires (une par cible de l'expérience C).

    `forward` renvoie :
    - "latent" : (B, latent_dim)
    - "preds"  : (B, n_targets), colonne j = cible `target_names[j]`
    """

    def __init__(
        self,
        encoder: nn.Module,
        latent_dim: int,
        head_hidden_dims: Sequence[int] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        target_names: Optional[Sequence[str]] = None,
        variant: str = "c2_cnn",
    ) -> None:
        super().__init__()
        models_module = _load_models_module()

        self.variant = str(variant)
        self.encoder = encoder
        self.latent_dim = int(latent_dim)
        self.target_names = tuple(target_names) if target_names is not None else _default_target_names()

        # positive_output=False : les cibles sont standardisées (peuvent être
        # négatives) ; on prédit dans l'espace standardisé.
        self.heads = nn.ModuleDict({
            name: models_module.ScalarHead(
                latent_dim=self.latent_dim,
                hidden_dims=head_hidden_dims,
                activation=activation,
                dropout=dropout,
                positive_output=False,
            )
            for name in self.target_names
        })

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self._encode(x)
        preds = torch.stack(
            [self.heads[name](h).squeeze(-1) for name in self.target_names],
            dim=1,
        )  # (B, n_targets)
        return {
            "latent": h,
            "preds": preds,
        }


# =============================================================================
# 4. Variante C1 : MLP sur snapshot unique
# =============================================================================

class ExperimentCMLP(ExperimentCMultiHeadModel):
    """
    C1 : snapshot u_t (B, Nx) -> MLPEncoder -> 5 têtes.

    Variante la plus proche du `EnergyRegressor` historique, mais avec
    Nx=1024 et 5 sorties.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 128,
        encoder_hidden_dims: Sequence[int] = (512, 256),
        head_hidden_dims: Sequence[int] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        target_names: Optional[Sequence[str]] = None,
    ) -> None:
        models_module = _load_models_module()
        encoder = models_module.MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        super().__init__(
            encoder=encoder,
            latent_dim=latent_dim,
            head_hidden_dims=head_hidden_dims,
            activation=activation,
            dropout=dropout,
            target_names=target_names,
            variant="c1_mlp",
        )
        self.input_dim = int(input_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, Nx)
        return self.encoder(x)


# =============================================================================
# 5. Variante C2 : CNN 1D sur snapshot unique
# =============================================================================

class ExperimentCCNN(ExperimentCMultiHeadModel):
    """
    C2 : snapshot u_t (B, Nx) -> Conv1DEncoder(1 canal) -> 5 têtes.
    """

    def __init__(
        self,
        nx: int,
        in_channels: int = 1,
        latent_dim: int = 128,
        conv_channels: Sequence[int] = (16, 32, 32, 64, 64),
        kernel_size: int = 5,
        conv_stride: int = 2,
        trunk_dims: Sequence[int] = (256,),
        head_hidden_dims: Sequence[int] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        target_names: Optional[Sequence[str]] = None,
    ) -> None:
        encoder = Conv1DEncoder(
            in_channels=in_channels,
            nx=nx,
            latent_dim=latent_dim,
            channels=conv_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            trunk_dims=trunk_dims,
            activation=activation,
            dropout=dropout,
        )
        super().__init__(
            encoder=encoder,
            latent_dim=latent_dim,
            head_hidden_dims=head_hidden_dims,
            activation=activation,
            dropout=dropout,
            target_names=target_names,
            variant="c2_cnn",
        )
        self.nx = int(nx)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, Nx) -> (B, 1, Nx)
        return self.encoder(x[:, None, :])


# =============================================================================
# 6. Variante C3 : CNN 1D sur fenêtre temporelle de w snapshots
# =============================================================================

class ExperimentCWindowCNN(ExperimentCMultiHeadModel):
    """
    C3 : fenêtre (B, window_size, Nx) -> Conv1DEncoder(window_size canaux) -> 5 têtes.

    La fenêtre (u_{t-w+1}, ..., u_t) est interprétée comme w canaux d'une
    entrée (B, w, Nx) ; les cibles sont celles de l'instant final t.
    """

    def __init__(
        self,
        nx: int,
        window_size: int = 5,
        latent_dim: int = 128,
        conv_channels: Sequence[int] = (16, 32, 32, 64, 64),
        kernel_size: int = 5,
        conv_stride: int = 2,
        trunk_dims: Sequence[int] = (256,),
        head_hidden_dims: Sequence[int] = (64,),
        activation: str = "relu",
        dropout: float = 0.0,
        target_names: Optional[Sequence[str]] = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("`window_size` doit être >= 1.")
        encoder = Conv1DEncoder(
            in_channels=int(window_size),
            nx=nx,
            latent_dim=latent_dim,
            channels=conv_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            trunk_dims=trunk_dims,
            activation=activation,
            dropout=dropout,
        )
        super().__init__(
            encoder=encoder,
            latent_dim=latent_dim,
            head_hidden_dims=head_hidden_dims,
            activation=activation,
            dropout=dropout,
            target_names=target_names,
            variant="c3_window",
        )
        self.nx = int(nx)
        self.window_size = int(window_size)

    def _encode(self, window: torch.Tensor) -> torch.Tensor:
        # window : (B, w, Nx) = (B, C_in, L)
        return self.encoder(window)


# =============================================================================
# 7. Fabrique de modèles
# =============================================================================

def build_experiment_c_model(
    variant: str,
    nx: int,
    window_size: int = 5,
    latent_dim: int = 128,
    encoder_hidden_dims: Sequence[int] = (512, 256),
    conv_channels: Sequence[int] = (16, 32, 32, 64, 64),
    kernel_size: int = 5,
    conv_stride: int = 2,
    trunk_dims: Sequence[int] = (256,),
    head_hidden_dims: Sequence[int] = (64,),
    activation: str = "relu",
    dropout: float = 0.0,
    target_names: Optional[Sequence[str]] = None,
) -> ExperimentCMultiHeadModel:
    """
    Construit un modèle de l'expérience C.

    Paramètres
    ----------
    variant : str
        "c1_mlp" (snapshot MLP), "c2_cnn" (snapshot CNN) ou
        "c3_window" (fenêtre temporelle CNN, `window_size` canaux).
    nx : int
        Taille spatiale des snapshots (1024 pour nu_sweep_v1).
    window_size : int
        Taille de la fenêtre pour la variante "c3_window".

    Retour
    ------
    ExperimentCMultiHeadModel
        Modèle multi-tâches (encodeur partagé + 5 têtes).
    """
    if variant == "c1_mlp":
        return ExperimentCMLP(
            input_dim=nx,
            latent_dim=latent_dim,
            encoder_hidden_dims=encoder_hidden_dims,
            head_hidden_dims=head_hidden_dims,
            activation=activation,
            dropout=dropout,
            target_names=target_names,
        )
    if variant == "c2_cnn":
        return ExperimentCCNN(
            nx=nx,
            in_channels=1,
            latent_dim=latent_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            trunk_dims=trunk_dims,
            head_hidden_dims=head_hidden_dims,
            activation=activation,
            dropout=dropout,
            target_names=target_names,
        )
    if variant == "c3_window":
        return ExperimentCWindowCNN(
            nx=nx,
            window_size=window_size,
            latent_dim=latent_dim,
            conv_channels=conv_channels,
            kernel_size=kernel_size,
            conv_stride=conv_stride,
            trunk_dims=trunk_dims,
            head_hidden_dims=head_hidden_dims,
            activation=activation,
            dropout=dropout,
            target_names=target_names,
        )
    raise ValueError(
        f"Variante inconnue : {variant!r}. Choisir parmi {EXPERIMENT_C_VARIANTS}."
    )


# =============================================================================
# 8. Window-A (Architecture A-Sequence) : autoencodeur de dynamique latente
#    avec encodeur convolutif sur fenêtre temporelle (w snapshots = w canaux)
# =============================================================================

def build_window_dynamics_autoencoder(
    window_size: int,
    nx: int,
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
) -> Any:
    """
    Construit le modèle window-A (Step 5) : `LatentDynamicsAutoencoder`
    (Architecture A, ml_models.py) avec un encodeur `Conv1DEncoder` prenant
    la fenêtre (B, window_size, Nx) — le temps entre comme canaux, comme dans
    la variante c3_window de l'Experiment C/C2.

    - encodeur : Conv1DEncoder(in_channels=window_size, nx=nx) -> h (B, latent_dim)
    - décodeur : MLPDecoder(h -> u_t final de la fenêtre, output_dim=nx)
    - prédicteur latent + tête scalaire z : identiques à Architecture A.

    La loss utilisée à l'entraînement est `latent_dynamics_autoencoder_total_loss`
    (ml_losses.py, inchangée) ; la cible de reconstruction est le snapshot final
    de la fenêtre (fourni par `NuSweepWindowPairDataset` sous la clé "u_t").
    """
    models_module = _load_models_module()
    if window_size < 1:
        raise ValueError("`window_size` doit être >= 1.")
    encoder = Conv1DEncoder(
        in_channels=int(window_size),
        nx=int(nx),
        latent_dim=latent_dim,
        channels=conv_channels,
        kernel_size=kernel_size,
        stride=conv_stride,
        trunk_dims=trunk_dims,
        activation=activation,
        dropout=dropout,
    )
    return models_module.LatentDynamicsAutoencoder(
        input_dim=int(nx),
        latent_dim=latent_dim,
        encoder=encoder,
        output_dim=int(nx),
        decoder_hidden_dims=decoder_hidden_dims,
        predictor_hidden_dims=predictor_hidden_dims,
        activation=activation,
        dropout=dropout,
        use_scalar_head=True,
        scalar_head_hidden_dims=scalar_head_hidden_dims,
        positive_scalar_output=False,
        residual_predictor=False,
    )


# =============================================================================
# 8b. Encodeurs TD (expérience TD, prereg outputs/prereg/td_encoder_prereg.json,
#     commit 515cf5b) — ADDITIF, td0 = chemin existant inchangé.
#
# TDChannelTransform : (B, w, Nx) -> (B, C, Nx)
#   - canaux de la fenêtre brute (w canaux, inchangés, en tête)
#   - + canaux de différences temporelles Δ_k u = (u_t - u_{t-k}) / (k*dt),
#     k ∈ diff_ks, avec u_t = dernier canal temporel (dim 0 = temps)
#   - + canal produit u_t ⊙ Δ_1 u si add_product (TD3 : injection architecturale
#     de l'identité dE/dt = -νD via mean_x(u_t ⊙ ∂_t u) ≈ (1/Nx) dE/dt)
# Conv1DEncoderTD : Conv1DEncoder avec pooling "flatten" (identique à Conv1DEncoder)
#   ou "gap" (global average pooling spatial, moyenne sur la longueur de sortie).
# =============================================================================

TD_DIFF_KS: Tuple[int, ...] = (1, 2, 4, 8, 16)
TD_ENCODER_MODES: Tuple[str, ...] = ("td0", "td1", "td2", "td3")


class TDChannelTransform(nn.Module):
    """(B, w, Nx) -> (B, C, Nx) : fenêtre brute + canaux de différences (+ produit)."""

    def __init__(
        self,
        window_size: int,
        dt: float,
        diff_ks: Sequence[int] = TD_DIFF_KS,
        add_product: bool = False,
    ) -> None:
        super().__init__()
        if window_size < 2:
            raise ValueError("`window_size` doit être >= 2 pour les différences temporelles.")
        self.window_size = int(window_size)
        self.dt = float(dt)
        self.diff_ks = tuple(int(k) for k in diff_ks)
        for k in self.diff_ks:
            if k < 1 or k >= window_size:
                raise ValueError(f"chaque k de diff_ks doit vérifier 1 <= k < window_size ({window_size}), reçu {k}.")
        self.add_product = bool(add_product)
        self.out_channels = self.window_size + len(self.diff_ks) + (1 if self.add_product else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, w, Nx), temps sur la dim 1, u_t = x[:, -1, :]
        u_t = x[:, -1, :]
        chans: List[torch.Tensor] = [x]
        for k in self.diff_ks:
            du = (u_t - x[:, -1 - k, :]) / (k * self.dt)
            chans.append(du.unsqueeze(1))
        if self.add_product:
            du1 = (u_t - x[:, -2, :]) / self.dt
            chans.append((u_t * du1).unsqueeze(1))
        return torch.cat(chans, dim=1)


class Conv1DEncoderTD(nn.Module):
    """
    Encodeur TD : TDChannelTransform -> blocs conv (mêmes hyperparamètres que
    Conv1DEncoder) -> pooling ("flatten" | "gap") -> trunk MLP -> latent.

    - pooling="flatten" : identique à Conv1DEncoder (TD0/TD1).
    - pooling="gap"     : moyenne spatiale sur la longueur de sortie conv
      (TD2/TD3) : biais inductif d'invariance par translation, comme les
      observables cibles (νD = ν∫u_x² dx est invariant par translation).
    """

    def __init__(
        self,
        in_channels: int,
        nx: int,
        latent_dim: int = 128,
        channels: Sequence[int] = (16, 32, 32, 64, 64),
        kernel_size: int = 5,
        stride: int = 2,
        trunk_dims: Sequence[int] = (256,),
        pooling: str = "flatten",
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        models_module = _load_models_module()
        if pooling not in ("flatten", "gap"):
            raise ValueError("`pooling` doit être 'flatten' ou 'gap'.")
        self.pooling = pooling
        self.in_channels = int(in_channels)
        self.nx = int(nx)
        self.latent_dim = int(latent_dim)
        self.channels = tuple(int(c) for c in channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

        layers: List[nn.Module] = []
        c_in = self.in_channels
        length = self.nx
        padding = self.kernel_size // 2
        for c_out in self.channels:
            layers.append(nn.Conv1d(c_in, c_out, self.kernel_size, stride=self.stride, padding=padding))
            layers.append(models_module.get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            c_in = c_out
            length = (length + 2 * padding - self.kernel_size) // self.stride + 1

        self.conv = nn.Sequential(*layers)
        self.out_length = length
        self.flatten_dim = length * self.channels[-1]
        trunk_input = self.channels[-1] if pooling == "gap" else self.flatten_dim
        self.to_latent = models_module.build_mlp(
            input_dim=trunk_input,
            hidden_dims=trunk_dims,
            output_dim=self.latent_dim,
            activation=activation,
            dropout=dropout,
            final_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.conv(x)  # (B, C_out, L_out)
        if self.pooling == "gap":
            pooled = feats.mean(dim=2)  # (B, C_out)
        else:
            pooled = feats.reshape(feats.shape[0], -1)  # (B, flatten_dim)
        return self.to_latent(pooled)


def build_td_encoder(
    encoder_mode: str,
    window_size: int,
    nx: int,
    dt: float,
    latent_dim: int = 8,
    conv_channels: Sequence[int] = (16, 32, 32, 64, 64),
    kernel_size: int = 5,
    conv_stride: int = 2,
    trunk_dims: Sequence[int] = (256,),
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Module:
    """
    Construit l'encodeur d'un bras TD (prereg td_encoder_prereg.json, commit 515cf5b) :

    - "td0" : Conv1DEncoder(in_channels=window_size) — encodeur actuel, inchangé.
    - "td1" : fenêtre brute + Δ_k u (k ∈ {1,2,4,8,16}), flatten.
    - "td2" : idem td1 + global average pooling spatial.
    - "td3" : td2 + canal produit u_t ⊙ Δ_1 u (injection architecturale, diagnostic).
    """
    if encoder_mode not in TD_ENCODER_MODES:
        raise ValueError(f"`encoder_mode` doit être dans {TD_ENCODER_MODES}, reçu {encoder_mode!r}.")
    if encoder_mode == "td0":
        return Conv1DEncoder(
            in_channels=int(window_size),
            nx=int(nx),
            latent_dim=latent_dim,
            channels=conv_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            trunk_dims=trunk_dims,
            activation=activation,
            dropout=dropout,
        )
    transform = TDChannelTransform(
        window_size=window_size,
        dt=dt,
        diff_ks=TD_DIFF_KS,
        add_product=(encoder_mode == "td3"),
    )
    pooling = "gap" if encoder_mode in ("td2", "td3") else "flatten"
    conv = Conv1DEncoderTD(
        in_channels=transform.out_channels,
        nx=int(nx),
        latent_dim=latent_dim,
        channels=conv_channels,
        kernel_size=kernel_size,
        stride=conv_stride,
        trunk_dims=trunk_dims,
        pooling=pooling,
        activation=activation,
        dropout=dropout,
    )

    class _TDEncoder(nn.Module):
        """Chaîne transform -> conv pour être branchée comme `encoder` du modèle."""

        def __init__(self, transform: TDChannelTransform, conv: Conv1DEncoderTD) -> None:
            super().__init__()
            self.transform = transform
            self.conv = conv

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(self.transform(x))

    return _TDEncoder(transform, conv)


def build_window_dynamics_autoencoder_td(
    encoder_mode: str,
    window_size: int,
    nx: int,
    dt: float,
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
) -> Any:
    """Variante TD de `build_window_dynamics_autoencoder` (même modèle, encodeur TD)."""
    models_module = _load_models_module()
    encoder = build_td_encoder(
        encoder_mode=encoder_mode,
        window_size=window_size,
        nx=nx,
        dt=dt,
        latent_dim=latent_dim,
        conv_channels=conv_channels,
        kernel_size=kernel_size,
        conv_stride=conv_stride,
        trunk_dims=trunk_dims,
        activation=activation,
        dropout=dropout,
    )
    return models_module.LatentDynamicsAutoencoder(
        input_dim=int(nx),
        latent_dim=latent_dim,
        encoder=encoder,
        output_dim=int(nx),
        decoder_hidden_dims=decoder_hidden_dims,
        predictor_hidden_dims=predictor_hidden_dims,
        activation=activation,
        dropout=dropout,
        use_scalar_head=True,
        scalar_head_hidden_dims=scalar_head_hidden_dims,
        positive_scalar_output=False,
        residual_predictor=False,
    )


# =============================================================================
# 9. Bloc de test minimal (smoke test CPU)
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    models_module = _load_models_module()

    batch_size = 4
    nx = 256          # nx réduit pour le smoke test CPU (1024 en production)
    window_size = 5

    x = torch.randn(batch_size, nx)          # snapshot (B, Nx)
    win = torch.randn(batch_size, window_size, nx)  # fenêtre (B, w, Nx)

    print("Smoke test ml_models_experiment_c.py")
    print("- cibles :", _default_target_names())

    for variant in EXPERIMENT_C_VARIANTS:
        model = build_experiment_c_model(
            variant,
            nx=nx,
            window_size=window_size,
            latent_dim=32,
            encoder_hidden_dims=(64,),
            conv_channels=(8, 16),
            trunk_dims=(32,),
            head_hidden_dims=(16,),
        )
        inp = win if variant == "c3_window" else x
        out = model(inp)
        n_params = models_module.count_parameters(model)
        print(
            f"- {variant:10s} preds {tuple(out['preds'].shape)} "
            f"latent {tuple(out['latent'].shape)} params={n_params}"
        )
        assert out["preds"].shape == (batch_size, len(_default_target_names()))
        assert out["latent"].shape == (batch_size, 32)

    # Vérification du gradient (environnement torch réel uniquement).
    model = build_experiment_c_model("c2_cnn", nx=nx, latent_dim=32,
                                    conv_channels=(8, 16), trunk_dims=(32,),
                                    head_hidden_dims=(16,))
    out = model(x)
    loss = out["preds"].sum()
    try:
        loss.backward()
        grads_ok = any(
            getattr(p, "grad", None) is not None for p in model.parameters()
        )
        print("- backward OK, au moins un gradient non nul :", grads_ok)
    except AttributeError:
        print("- backward ignoré (environnement de test sans autograd)")

    print("- résumé :", models_module.summarize_model(model, "c2_cnn").to_dict())
