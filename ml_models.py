"""
ml/models.py
============

Architectures PyTorch simples, lisibles et commentées pour le projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module regroupe les briques de base nécessaires pour les premières
expériences de machine learning sur les trajectoires de Burgers :

1. encodeur MLP pour transformer un snapshot spatial u en latent h,
2. tête scalaire pour produire une observable candidate z,
3. décodeur MLP pour reconstruire un état depuis un latent,
4. régressseur d'énergie supervisé,
5. prédicteur latent pour la dynamique h_t -> h_{t+1},
6. modèles composés (autoencodeur, dynamique latente, etc.).

Philosophie
-----------
- Pas d'architecture complexe à ce stade.
- Priorité à la lisibilité et à la modularité.
- Composants réutilisables et facilement combinables.
- Code adapté à un prototype de recherche, pas encore à une grosse base de code.

Conventions
-----------
- Les entrées sont généralement des tenseurs de forme (batch_size, Nx).
- Les latents sont de forme (batch_size, latent_dim).
- Les sorties scalaires sont de forme (batch_size, 1).
- Les activations par défaut sont ReLU, mais elles peuvent être changées.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn


# =============================================================================
# 1. Utilitaires généraux
# =============================================================================

def get_activation(name: str) -> nn.Module:
    """
    Retourne une fonction d'activation PyTorch à partir de son nom.

    Paramètres
    ----------
    name : str
        Nom de l'activation. Possibilités actuelles :
        - "relu"
        - "gelu"
        - "tanh"
        - "elu"
        - "leaky_relu"

    Retour
    ------
    nn.Module
        Module d'activation correspondant.
    """
    name = name.lower()

    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)

    raise ValueError(
        f"Activation inconnue : {name!r}. "
        "Choisir parmi {'relu', 'gelu', 'tanh', 'elu', 'leaky_relu'}."
    )



def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Compte le nombre de paramètres d'un modèle.

    Paramètres
    ----------
    model : nn.Module
        Modèle PyTorch.
    trainable_only : bool
        Si True, ne compte que les paramètres avec `requires_grad=True`.

    Retour
    ------
    int
        Nombre de paramètres.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())



def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
    final_activation: Optional[str] = None,
) -> nn.Sequential:
    """
    Construit un MLP fully-connected simple.

    Paramètres
    ----------
    input_dim : int
        Dimension d'entrée.
    hidden_dims : Sequence[int]
        Dimensions des couches cachées.
    output_dim : int
        Dimension de sortie.
    activation : str
        Activation utilisée entre les couches cachées.
    dropout : float
        Taux de dropout appliqué après les activations cachées.
    final_activation : str ou None
        Activation optionnelle appliquée après la dernière couche.

    Retour
    ------
    nn.Sequential
        Réseau MLP prêt à l'emploi.
    """
    if input_dim < 1 or output_dim < 1:
        raise ValueError("`input_dim` et `output_dim` doivent être >= 1.")
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("`dropout` doit vérifier 0 <= dropout < 1.")

    layers: List[nn.Module] = []
    dims = [input_dim, *hidden_dims, output_dim]

    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

    # Dernière couche linéaire
    layers.append(nn.Linear(dims[-2], dims[-1]))

    if final_activation is not None:
        layers.append(get_activation(final_activation))

    return nn.Sequential(*layers)


# =============================================================================
# 2. Encodeur MLP
# =============================================================================

class MLPEncoder(nn.Module):
    """
    Encodeur MLP : transforme un snapshot spatial u -> latent h.

    Exemple typique
    ---------------
    Entrée  : u  R^{Nx}
    Sortie  : h  R^{latent_dim}
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation_name = activation
        self.dropout = float(dropout)

        self.network = build_mlp(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.latent_dim,
            activation=self.activation_name,
            dropout=self.dropout,
            final_activation=None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Paramètres
        ----------
        x : torch.Tensor
            Tenseur d'entrée de forme (batch_size, input_dim).

        Retour
        ------
        torch.Tensor
            Latent de forme (batch_size, latent_dim).
        """
        return self.network(x)


# =============================================================================
# 3. Tête scalaire
# =============================================================================

class ScalarHead(nn.Module):
    """
    Tête scalaire simple : h -> z.

    Utilité
    -------
    Produire une observable candidate z à partir du latent h,
    par exemple :
    - énergie prédite,
    - entropie candidate,
    - quantité monotone apprise.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int] = (),
        activation: str = "relu",
        dropout: float = 0.0,
        positive_output: bool = False,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation_name = activation
        self.dropout = float(dropout)
        self.positive_output = bool(positive_output)

        final_activation = None
        self.network = build_mlp(
            input_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            activation=self.activation_name,
            dropout=self.dropout,
            final_activation=final_activation,
        )

        # Option simple pour forcer la positivité de la sortie si souhaité.
        self.output_transform = nn.Softplus() if self.positive_output else nn.Identity()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.network(h)
        return self.output_transform(z)


# =============================================================================
# 4. Décodeur MLP
# =============================================================================

class MLPDecoder(nn.Module):
    """
    Décodeur MLP : latent h -> état reconstruit u_hat.

    Exemple typique
    ---------------
    Entrée  : h  R^{latent_dim}
    Sortie  :  R^{Nx}
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (64, 128),
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation_name = activation
        self.dropout = float(dropout)

        self.network = build_mlp(
            input_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
            activation=self.activation_name,
            dropout=self.dropout,
            final_activation=None,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.network(h)


# =============================================================================
# 5. Régressseur d'énergie supervisé
# =============================================================================

class EnergyRegressor(nn.Module):
    """
    Modèle supervisé simple pour approximer l'énergie d'un snapshot.

    Architecture
    ------------
    u -> encoder -> h -> scalar head -> E_hat
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        encoder_hidden_dims: Sequence[int] = (128, 64),
        head_hidden_dims: Sequence[int] = (),
        activation: str = "relu",
        dropout: float = 0.0,
        positive_output: bool = True,
    ) -> None:
        super().__init__()

        self.encoder = MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.head = ScalarHead(
            latent_dim=latent_dim,
            hidden_dims=head_hidden_dims,
            activation=activation,
            dropout=dropout,
            positive_output=positive_output,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        energy_hat = self.head(h)
        return {
            "latent": h,
            "energy_hat": energy_hat,
        }


# =============================================================================
# 6. Prédicteur latent
# =============================================================================

class LatentPredictor(nn.Module):
    """
    Prédicteur de dynamique latente : h_t -> h_{t+1}.

    Utilité
    -------
    Cette brique est adaptée aux expériences de type :
    - prédiction à un pas en espace latent,
    - world model minimal,
    - encodeur + prédicteur + variable scalaire monotone.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.activation_name = activation
        self.dropout = float(dropout)
        self.residual = bool(residual)

        self.network = build_mlp(
            input_dim=self.latent_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.latent_dim,
            activation=self.activation_name,
            dropout=self.dropout,
            final_activation=None,
        )

    def forward(self, h_t: torch.Tensor) -> torch.Tensor:
        delta_or_next = self.network(h_t)
        if self.residual:
            return h_t + delta_or_next
        return delta_or_next


# =============================================================================
# 7. Autoencodeur latent simple
# =============================================================================

class LatentAutoencoder(nn.Module):
    """
    Autoencodeur simple avec tête scalaire optionnelle.

    Architecture
    ------------
    u -> encoder -> h -> decoder -> u_hat
                    \-> scalar head -> z (optionnel)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        encoder_hidden_dims: Sequence[int] = (128, 64),
        decoder_hidden_dims: Sequence[int] = (64, 128),
        activation: str = "relu",
        dropout: float = 0.0,
        use_scalar_head: bool = True,
        scalar_head_hidden_dims: Sequence[int] = (),
        positive_scalar_output: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )

        self.use_scalar_head = bool(use_scalar_head)
        if self.use_scalar_head:
            self.scalar_head = ScalarHead(
                latent_dim=latent_dim,
                hidden_dims=scalar_head_hidden_dims,
                activation=activation,
                dropout=dropout,
                positive_output=positive_scalar_output,
            )
        else:
            self.scalar_head = None

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        x_hat = self.decoder(h)

        outputs: Dict[str, torch.Tensor] = {
            "latent": h,
            "reconstruction": x_hat,
        }

        if self.scalar_head is not None:
            outputs["scalar"] = self.scalar_head(h)

        return outputs


# =============================================================================
# 8. Modèle de dynamique latente
# =============================================================================

class LatentDynamicsModel(nn.Module):
    """
    Modèle minimal pour l'apprentissage de dynamique latente.

    Architecture
    ------------
    u_t   -> encoder -> h_t   -> predictor -> h_{t+1}^{pred}
    u_{t+1} -> encoder -> h_{t+1}

    En option, une tête scalaire applique :
        h_t      -> z_t
        h_{t+1}  -> z_{t+1}

    Utilité
    -------
    Ce type de modèle est bien adapté à :
    - la prédiction à un pas,
    - l'analyse de monotonie d'une observable apprise,
    - la comparaison entre dynamique physique et dynamique latente.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        encoder_hidden_dims: Sequence[int] = (128, 64),
        predictor_hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        use_scalar_head: bool = True,
        scalar_head_hidden_dims: Sequence[int] = (),
        positive_scalar_output: bool = False,
        residual_predictor: bool = False,
    ) -> None:
        super().__init__()

        self.encoder = MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dims=predictor_hidden_dims,
            activation=activation,
            dropout=dropout,
            residual=residual_predictor,
        )

        self.use_scalar_head = bool(use_scalar_head)
        if self.use_scalar_head:
            self.scalar_head = ScalarHead(
                latent_dim=latent_dim,
                hidden_dims=scalar_head_hidden_dims,
                activation=activation,
                dropout=dropout,
                positive_output=positive_scalar_output,
            )
        else:
            self.scalar_head = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, u_t: torch.Tensor, u_tp: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h_t = self.encoder(u_t)
        h_tp_pred = self.predictor(h_t)

        outputs: Dict[str, torch.Tensor] = {
            "h_t": h_t,
            "h_tp_pred": h_tp_pred,
        }

        if u_tp is not None:
            h_tp = self.encoder(u_tp)
            outputs["h_tp"] = h_tp

        if self.scalar_head is not None:
            outputs["z_t"] = self.scalar_head(h_t)
            outputs["z_tp_pred"] = self.scalar_head(h_tp_pred)
            if u_tp is not None:
                outputs["z_tp"] = self.scalar_head(outputs["h_tp"])

        return outputs


# =============================================================================
# 8bis. Autoencodeur de dynamique latente (Architecture A)
# =============================================================================

def latent_pca_stats(h: torch.Tensor, var_threshold: float = 0.95) -> Dict[str, Any]:
    """
    Statistiques PCA d'un batch de latents h (détection d'effondrement de rang).

    Utilité
    -------
    Surveiller si le latent h s'effondre sur un sous-espace de petite dimension
    (dans les checkpoints existants, h est exactement de rang 1).

    Paramètres
    ----------
    h : torch.Tensor
        Latent de forme (batch, latent_dim).
    var_threshold : float
        Seuil de variance expliquée cumulée utilisé pour le rang effectif
        (0.95 par défaut).

    Retour
    ------
    Dict[str, Any]
        - 'eigenvalues' : torch.Tensor de forme (latent_dim,), valeurs propres
          de la matrice de covariance de h, triées par ordre décroissant,
        - 'effective_rank' : int, plus petit rang k tel que les k premières
          valeurs propres expliquent au moins `var_threshold` de la variance
          totale (0 si la variance totale est nulle),
        - 'max_offline_residual' : float, norme maximale du résidu de chaque
          échantillon hors de la première composante principale ; détecteur de
          rang exact 1 (~0, ex 3e-16, si h = s . v exactement).
    """
    if h.ndim != 2:
        raise ValueError(f"`h` doit être 2D (batch, latent_dim), obtenu {tuple(h.shape)}")
    if not (0.0 < float(var_threshold) <= 1.0):
        raise ValueError("`var_threshold` doit vérifier 0 < var_threshold <= 1.")

    h_centered = h - h.mean(dim=0, keepdim=True)
    _, singular_values, vt = torch.linalg.svd(h_centered, full_matrices=False)
    eigenvalues = singular_values ** 2 / max(h.shape[0] - 1, 1)

    total_variance = eigenvalues.sum()
    if float(total_variance) <= 0.0:
        effective_rank = 0
    else:
        cumulative = torch.cumsum(eigenvalues, dim=0) / total_variance
        effective_rank = int((cumulative < float(var_threshold)).sum().item()) + 1

    # Résidu hors première composante principale (détecteur de rang exact 1).
    v1 = vt[0]
    projection = (h_centered @ v1).unsqueeze(1) * v1.unsqueeze(0)
    residuals = (h_centered - projection).norm(dim=1)
    max_offline_residual = float(residuals.max().item()) if h.shape[0] > 0 else 0.0

    return {
        'eigenvalues': eigenvalues,
        'effective_rank': effective_rank,
        'max_offline_residual': max_offline_residual,
    }


class LatentDynamicsAutoencoder(nn.Module):
    """
    Autoencodeur + dynamique latente + tête scalaire (Architecture A).

    Architecture
    ------------
    u_t     -> encoder -> h_t -> decoder     -> u_hat_t (reconstruction)
                          \\-> predictor   -> h_{t+1}^{pred}
                          \\-> scalar head -> z_t
    u_{t+1} -> encoder -> h_{t+1} (cible de prédiction)
                          \\-> scalar head -> z_{t+1}

    Objectif
    --------
    Par rapport à LatentDynamicsModel (prédiction uniquement en espace latent),
    ce modèle ajoute une reconstruction en espace d'état : le latent h est forcé
    de conserver l'information d'état, ce qui vise à éviter l'effondrement de
    rang observé sur LatentDynamicsModel (h exactement de rang 1).

    Les clés du state_dict suivent le motif habituel :
    encoder.network.{0,2,4}.*, decoder.network.{0,2,4}.*,
    predictor.network.{0,2,4}.*, scalar_head.network.0.*
    (avec dropout=0 et les hidden_dims par défaut).
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        encoder_hidden_dims: Sequence[int] = (128, 64),
        decoder_hidden_dims: Sequence[int] = (64, 128),
        predictor_hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        dropout: float = 0.0,
        use_scalar_head: bool = True,
        scalar_head_hidden_dims: Sequence[int] = (),
        positive_scalar_output: bool = False,
        residual_predictor: bool = False,
        encoder: Optional[nn.Module] = None,
        output_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Encodeur interchangeable (window-A / A-Sequence) : si `encoder` est
        # fourni (ex. Conv1DEncoder avec in_channels=window_size), il est utilisé
        # tel quel et `output_dim` fixe la sortie du décodeur (défaut: input_dim).
        self.encoder = encoder if encoder is not None else MLPEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.decoder = MLPDecoder(
            latent_dim=latent_dim,
            output_dim=output_dim if output_dim is not None else input_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        self.predictor = LatentPredictor(
            latent_dim=latent_dim,
            hidden_dims=predictor_hidden_dims,
            activation=activation,
            dropout=dropout,
            residual=residual_predictor,
        )

        self.use_scalar_head = bool(use_scalar_head)
        if self.use_scalar_head:
            self.scalar_head = ScalarHead(
                latent_dim=latent_dim,
                hidden_dims=scalar_head_hidden_dims,
                activation=activation,
                dropout=dropout,
                positive_output=positive_scalar_output,
            )
        else:
            self.scalar_head = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, u_t: torch.Tensor, u_tp: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        h_t = self.encoder(u_t)
        reconstruction = self.decoder(h_t)
        h_tp_pred = self.predictor(h_t)

        outputs: Dict[str, torch.Tensor] = {
            "h_t": h_t,
            "h_tp_pred": h_tp_pred,
            "reconstruction": reconstruction,
        }

        if u_tp is not None:
            h_tp = self.encoder(u_tp)
            outputs["h_tp"] = h_tp

        if self.scalar_head is not None:
            outputs["z_t"] = self.scalar_head(h_t)
            outputs["z_tp_pred"] = self.scalar_head(h_tp_pred)
            if u_tp is not None:
                outputs["z_tp"] = self.scalar_head(outputs["h_tp"])

        return outputs

    def monitoring_stats(self, u_batch: torch.Tensor) -> Dict[str, Any]:
        """
        Statistiques de monitoring sur un batch (typiquement un batch de val).

        Paramètres
        ----------
        u_batch : torch.Tensor
            Batch d'états de forme (batch, input_dim).

        Retour
        ------
        Dict[str, Any]
            - 'latent_variance_per_dim' : torch.Tensor (latent_dim,), variance
              de chaque dimension de h_t sur le batch,
            - 'eigenvalues', 'effective_rank', 'max_offline_residual' :
              statistiques PCA de h_t (voir latent_pca_stats),
            - 'alive_layer2_units' : int, nombre d'unités de la 2e couche
              linéaire de l'encodeur (encoder.network[2] quand dropout = 0)
              dont la pré-activation est > 0 pour au moins un échantillon.
        """
        with torch.no_grad():
            h = self.encoder(u_batch)
            pca = latent_pca_stats(h)
            var_per_dim = torch.var(h, dim=0, unbiased=False)

            # Pré-activation de la 2e couche linéaire de l'encodeur
            # (identique à encoder.network[2](...) quand dropout = 0).
            out = u_batch
            linear_count = 0
            for module in self.encoder.network:
                out = module(out)
                if isinstance(module, nn.Linear):
                    linear_count += 1
                    if linear_count == 2:
                        break
            alive_layer2_units = int((out > 0).any(dim=0).sum().item())

        return {
            'latent_variance_per_dim': var_per_dim,
            'eigenvalues': pca['eigenvalues'],
            'effective_rank': pca['effective_rank'],
            'max_offline_residual': pca['max_offline_residual'],
            'alive_layer2_units': alive_layer2_units,
        }


# =============================================================================
# 9. Configuration de modèles type (optionnelle)
# =============================================================================

@dataclass
class ModelSummary:
    """
    Petit conteneur pratique pour résumer un modèle.
    """

    model_name: str
    total_parameters: int
    trainable_parameters: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_parameters": int(self.total_parameters),
            "trainable_parameters": int(self.trainable_parameters),
        }



def summarize_model(model: nn.Module, model_name: Optional[str] = None) -> ModelSummary:
    """
    Retourne un résumé simple du nombre de paramètres d'un modèle.
    """
    name = model_name or model.__class__.__name__
    return ModelSummary(
        model_name=name,
        total_parameters=count_parameters(model, trainable_only=False),
        trainable_parameters=count_parameters(model, trainable_only=True),
    )


# =============================================================================
# 10. Bloc de test minimal
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 4
    input_dim = 128
    latent_dim = 8

    x = torch.randn(batch_size, input_dim)
    y = torch.randn(batch_size, input_dim)

    encoder = MLPEncoder(input_dim=input_dim, latent_dim=latent_dim)
    head = ScalarHead(latent_dim=latent_dim, positive_output=True)
    decoder = MLPDecoder(latent_dim=latent_dim, output_dim=input_dim)
    regressor = EnergyRegressor(input_dim=input_dim, latent_dim=latent_dim)
    autoencoder = LatentAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    dynamics = LatentDynamicsModel(input_dim=input_dim, latent_dim=latent_dim)

    h = encoder(x)
    z = head(h)
    x_hat = decoder(h)
    reg_out = regressor(x)
    ae_out = autoencoder(x)
    dyn_out = dynamics(x, y)

    print("Tests rapides sur ml/models.py")
    print("- h shape            :", tuple(h.shape))
    print("- z shape            :", tuple(z.shape))
    print("- reconstruction     :", tuple(x_hat.shape))
    print("- regressor keys     :", list(reg_out.keys()))
    print("- autoencoder keys   :", list(ae_out.keys()))
    print("- dynamics keys      :", list(dyn_out.keys()))

    print("\nRésumé du modèle LatentDynamicsModel :")
    print(summarize_model(dynamics).to_dict())

    dynamics_ae = LatentDynamicsAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    dyn_ae_out = dynamics_ae(x, y)
    dyn_ae_out_single = dynamics_ae(x)

    print("\nTests rapides sur LatentDynamicsAutoencoder :")
    print("- keys avec u_tp     :", list(dyn_ae_out.keys()))
    print("- keys sans u_tp     :", list(dyn_ae_out_single.keys()))
    print("- h_t shape          :", tuple(dyn_ae_out["h_t"].shape))
    print("- reconstruction     :", tuple(dyn_ae_out["reconstruction"].shape))
    print("- z_t shape          :", tuple(dyn_ae_out["z_t"].shape))

    mon = dynamics_ae.monitoring_stats(x)
    print("- monitoring rank    :", mon['effective_rank'])
    print("- monitoring alive L2:", mon['alive_layer2_units'])
    print("- monitoring var/dim :", tuple(mon['latent_variance_per_dim'].shape))

    s = torch.randn(batch_size, 1)
    v = torch.randn(1, latent_dim)
    pca_rank1 = latent_pca_stats(s @ v)
    print("- PCA rang 1         : effective_rank =", pca_rank1['effective_rank'],
          ", max_offline_residual =", pca_rank1['max_offline_residual'])

    print("\nRésumé du modèle LatentDynamicsAutoencoder :")
    print(summarize_model(dynamics_ae).to_dict())
