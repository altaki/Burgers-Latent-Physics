"""
config.py
==========

Fichier de configuration central du projet :
"Découverte de variables physiques latentes sur l'équation de Burgers visqueuse".

Objectif
--------
Ce module regroupe tous les paramètres importants du projet afin d'éviter
la dispersion des constantes dans plusieurs fichiers. L'idée est d'avoir
un point d'entrée unique pour :

1. la simulation de l'équation de Burgers,
2. la génération des données,
3. l'entraînement des modèles de machine learning,
4. la reproductibilité,
5. l'organisation des dossiers de sortie.

Philosophie
-----------
- Code lisible et abondamment commenté.
- Paramètres regroupés par blocs logiques.
- Utilisation de dataclasses pour un accès clair via la notation pointée.
- Fonction utilitaire pour créer automatiquement les dossiers de sortie.

Exemple d'utilisation
---------------------
>>> from config import get_default_config
>>> cfg = get_default_config()
>>> print(cfg.pde.nu)
>>> print(cfg.paths.data_dir)

Remarque
--------
Cette première version est volontairement simple et stable.
Elle est très adaptée à un prototype de recherche.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, Any, List, Optional


# -----------------------------------------------------------------------------
# 1. Configuration des chemins du projet
# -----------------------------------------------------------------------------

@dataclass
class PathsConfig:
    """
    Regroupe tous les chemins utiles au projet.

    Tous les dossiers sont construits à partir d'un dossier racine `output_root`.
    Cela facilite l'organisation du projet et évite d'écrire des chemins en dur
    dans les scripts.
    """

    # Dossier racine dans lequel seront enregistrés les résultats.
    output_root: Path = Path("outputs")

    # Sous-dossiers principaux.
    data_dir: Path = field(default_factory=lambda: Path("outputs") / "data")
    models_dir: Path = field(default_factory=lambda: Path("outputs") / "models")
    figures_dir: Path = field(default_factory=lambda: Path("outputs") / "figures")
    logs_dir: Path = field(default_factory=lambda: Path("outputs") / "logs")

    def create_directories(self) -> None:
        """
        Crée tous les dossiers nécessaires s'ils n'existent pas déjà.
        """
        for path in [self.output_root, self.data_dir, self.models_dir,
                     self.figures_dir, self.logs_dir]:
            path.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. Configuration PDE : équation de Burgers visqueuse
# -----------------------------------------------------------------------------

@dataclass
class PDEConfig:
    """
    Paramètres de la simulation de l'équation de Burgers visqueuse 1D.

    Équation cible :
        u_t + u u_x = nu * u_xx

    sur un domaine périodique [0, L].
    """

    # Longueur du domaine spatial.
    L: float = 2.0 * 3.141592653589793

    # Nombre de points de discrétisation en espace.
    Nx: int = 128

    # Viscosité.
    nu: float = 0.05

    # Temps final de simulation.
    T: float = 2.0

    # Pas de temps numérique.
    dt: float = 2.0e-4

    # Sauvegarder un snapshot tous les `save_every` pas de temps.
    save_every: int = 100

    # Type de conditions aux limites. Ici, on vise uniquement le cas périodique.
    boundary_condition: str = "periodic"

    # Choix du solveur.
    # Version initiale conseillée : différences finies + RK4 explicite.
    spatial_scheme: str = "finite_difference"
    time_scheme: str = "rk4"

    @property
    def dx(self) -> float:
        """
        Pas d'espace déduit automatiquement de L et Nx.
        """
        return self.L / self.Nx

    @property
    def n_steps(self) -> int:
        """
        Nombre total de pas de temps.
        """
        return int(self.T / self.dt)

    @property
    def n_saved_steps(self) -> int:
        """
        Nombre de snapshots sauvegardés, en incluant l'état initial.
        """
        return self.n_steps // self.save_every + 1


# -----------------------------------------------------------------------------
# 3. Configuration de génération du dataset
# -----------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """
    Paramètres liés à la génération et au découpage du dataset.
    """

    # Nombre de trajectoires par split.
    n_train: int = 500
    n_val: int = 100
    n_test: int = 100

    # Familles de conditions initiales autorisées.
    # On pourra tirer aléatoirement parmi ces types.
    ic_types: List[str] = field(default_factory=lambda: [
        "fourier",
        "gaussian_bumps",
        "random_smooth",
    ])

    # Paramètres de base pour les conditions initiales de type Fourier.
    max_fourier_mode: int = 3
    fourier_amplitude_scale: float = 0.5

    # Paramètres de base pour les bosses gaussiennes.
    n_gaussian_bumps: int = 2

    # Paramètres de lissage spectral pour les champs aléatoires lisses.
    spectral_decay: float = 3.0

    # Nom des fichiers de sortie par split.
    train_filename: str = "train_data.npz"
    val_filename: str = "val_data.npz"
    test_filename: str = "test_data.npz"


# -----------------------------------------------------------------------------
# 4. Configuration du prétraitement / normalisation
# -----------------------------------------------------------------------------

@dataclass
class NormalizationConfig:
    """
    Paramètres de normalisation des données.

    Pour préserver autant que possible l'information physique globale,
    on recommande une normalisation globale (moyenne / écart-type calculés
    sur le train), et non pas une normalisation indépendante trajectoire
    par trajectoire.
    """

    enabled: bool = True
    method: str = "global_standardization"  # autres possibilités futures : none, minmax
    eps: float = 1.0e-8


# -----------------------------------------------------------------------------
# 5. Configuration machine learning
# -----------------------------------------------------------------------------

@dataclass
class MLConfig:
    """
    Paramètres globaux pour les modèles d'apprentissage.
    """

    # Dimension du latent principal h_t.
    latent_dim: int = 8

    # Taille de batch.
    batch_size: int = 64

    # Nombre d'époques.
    num_epochs: int = 100

    # Taux d'apprentissage.
    learning_rate: float = 1.0e-3

    # Weight decay optionnel.
    weight_decay: float = 0.0

    # Nom de l'optimiseur.
    optimizer_name: str = "adam"

    # Device choisi automatiquement plus tard si disponible.
    device: str = "cpu"

    # Fraction éventuelle du dataset à utiliser pour des tests rapides.
    train_fraction: float = 1.0


# -----------------------------------------------------------------------------
# 6. Configuration des losses pour les modèles latents
# -----------------------------------------------------------------------------

@dataclass
class LossConfig:
    """
    Poids des différentes composantes de la fonction de coût.
    """

    # Poids de la reconstruction (si autoencodeur).
    reconstruction_weight: float = 1.0

    # Poids de la prédiction latente h_t -> h_{t+1}.
    prediction_weight: float = 1.0

    # Poids de la contrainte de monotonie sur la variable scalaire z_t.
    monotonicity_weight: float = 1.0

    # Poids pour éviter la solution triviale z = constante.
    variance_weight: float = 1.0e-2

    # Variance minimale souhaitée pour z.
    min_latent_variance: float = 1.0e-3

    # Type de pénalité de monotonie.
    monotonicity_mode: str = "squared_hinge"  # "hinge" ou "squared_hinge"


# -----------------------------------------------------------------------------
# 7. Configuration d'évaluation
# -----------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """
    Paramètres liés à l'évaluation et aux figures.
    """

    # Nombre de trajectoires à afficher dans les figures de contrôle.
    n_plot_trajectories: int = 5

    # Sauvegarder automatiquement les figures.
    save_figures: bool = True

    # Fréquence d'affichage de logs pendant l'entraînement.
    print_every: int = 10

    # Taille de police par défaut pour les figures matplotlib.
    matplotlib_fontsize: int = 11


# -----------------------------------------------------------------------------
# 8. Reproductibilité
# -----------------------------------------------------------------------------

@dataclass
class ReproConfig:
    """
    Paramètres de reproductibilité.
    """

    seed: int = 42
    deterministic_torch: bool = False


# -----------------------------------------------------------------------------
# 8bis. Configuration nu_sweep_v1
# -----------------------------------------------------------------------------

@dataclass
class NuSweepConfig:
    """
    Paramètres du dataset nu_sweep_v1 (balayage de viscosité).

    Protocole : trois viscosités stratifiées par split (200/50/50), une
    norme L2 cible tirée log-uniformément, des conditions initiales de
    moyenne nulle pour toutes les familles, et une configuration numérique
    validée (Nx=1024, dt=1e-4, save_every=200, RK4) assurant un nombre de
    Péclet de maille inférieur à 2 (critère de maille, et non de CFL).

    Les sorties sont écrites dans ``outputs/data`` sous les noms
    ``{train,val,test}_data_nu_sweep.npz`` ; les fichiers legacy
    ``*_data.npz`` restent inchangés.
    """

    # Viscosités balayées, stratifiées par split (200/50/50 par valeur de nu).
    nu_values: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.20])

    # Bornes de la norme L2 cible, tirée log-uniformément (E(0) dans [0.125, 2.0]).
    target_norm_range: List[float] = field(default_factory=lambda: [0.5, 2.0])

    # Moyenne nulle imposée pour toutes les familles de conditions initiales.
    zero_mean: bool = True

    # Discrétisation spatiale (Nx=1024 requis à nu=0.01 ; Nx=128/256 non sûrs).
    Nx: int = 1024

    # Pas de temps (à nu=0.20, dt=2e-4 serait instable en diffusion).
    dt: float = 1.0e-4

    # Sauvegarder un snapshot tous les `save_every` pas (200 -> 101 snapshots).
    save_every: int = 200

    # Longueur du domaine périodique.
    L: float = 2.0 * 3.141592653589793

    # Temps final de simulation.
    T: float = 2.0

    # Schéma d'intégration temporelle.
    time_scheme: str = "rk4"

    # Nombre de trajectoires par split (stratifiées 200/50/50 par nu).
    n_train: int = 600
    n_val: int = 150
    n_test: int = 150

    # Noms des fichiers de sortie par split (dataset nu_sweep_v1 dédié).
    train_filename: str = "train_data_nu_sweep.npz"
    val_filename: str = "val_data_nu_sweep.npz"
    test_filename: str = "test_data_nu_sweep.npz"

    # Version du dataset et chaîne de protocole (métadonnées des npz).
    dataset_version: str = "nu_sweep_v1"
    protocol: str = (
        "nu_sweep_v1: nu in {0.01,0.05,0.20} stratifie; target_l2_norm "
        "log-uniform(0.5,2.0); zero_mean=True pour toutes les familles de CI; "
        "Nx=1024, dt=1e-4, save_every=200, RK4, L=2pi, T=2.0, 101 snapshots"
    )

    # Paramètres du pilote de convergence (grille plus fine).
    pilot_n_per_nu: int = 20
    pilot_seed: int = 1234
    pilot_ladder: List[List[float]] = field(
        default_factory=lambda: [[1024, 1e-4, 200], [2048, 2.5e-5, 800]]
    )

    # Seuils de validation du pilote (portes de qualité).
    gate_monotone_frac: float = 0.99
    gate_monotone_atol_rel: float = 1e-8     # atol = 1e-8 * E(0)
    gate_pass_fraction: float = 0.95
    gate_max_abs_u: float = 5.0
    gate_dedt_rel_err: float = 1e-2          # médiane de |dE/dt + nu*D| / nu*D
    gate_max_cell_peclet: float = 2.0        # diagnostic


# -----------------------------------------------------------------------------
# 8ter. Configuration de l'expérience C (baseline supervisée de séparabilité)
# -----------------------------------------------------------------------------

@dataclass
class ExperimentCConfig:
    """
    Paramètres de l'expérience C : baseline supervisée multi-tâches mesurant
    la séparabilité des observables {E, nu*D, D, TV, max|u_x|} sur le dataset
    nu_sweep_v1, avant tout modèle latent.

    Variantes de modèles (ml_models_experiment_c) :
    - "c1_mlp"    : MLP sur snapshot unique,
    - "c2_cnn"    : CNN 1D sur snapshot unique,
    - "c3_window" : CNN 1D sur fenêtre de `window_size` snapshots.

    Protocoles :
    - "standard"   : entraînement sur les 3 strates de nu,
    - "heldout_nu" : entraînement restreint à `heldout_nu_values`, évaluation
                     sur la strate nu=0.01 du test.

    Seuils de décision pour le verdict d'identifiabilité de nu*D
    (voir main_experiment_c.evaluate_identifiability) :
    - IDENTIFIABLE     si r_holdout >= r_holdout_min, rho <= rho_max,
                       equiv < equiv_max et le modèle bat la baseline nu_oracle ;
    - NOT IDENTIFIABLE si r_holdout < r_holdout_fail ou rel_rmse_holdout > rel_rmse_fail ;
    - PARTIAL          sinon.
    """

    # Variantes et protocoles à exécuter.
    variants: List[str] = field(default_factory=lambda: ["c1_mlp", "c2_cnn", "c3_window"])
    protocols: List[str] = field(default_factory=lambda: ["standard", "heldout_nu"])

    # Strates de nu utilisées pour l'entraînement en protocole heldout_nu.
    heldout_nu_values: List[float] = field(default_factory=lambda: [0.05, 0.20])

    # Fenêtre temporelle de la variante c3_window.
    window_size: int = 5

    # Tailles de fenêtre testées pour c3_window (Experiment C2 : ex. [5, 25, 50]).
    # None = [window_size] (comportement historique : une seule fenêtre, labels
    # de runs inchangés). Si plusieurs tailles sont données, les runs sont
    # étiquetés c3_window_w{w} pour ne pas écraser les runs historiques.
    window_sizes: Optional[List[int]] = None

    # Fichiers de données d'entrée (dataset nu_sweep_v1).
    train_filename: str = "train_data_nu_sweep.npz"
    val_filename: str = "val_data_nu_sweep.npz"
    test_filename: str = "test_data_nu_sweep.npz"

    # Hyperparamètres d'entraînement (CPU-friendly).
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    num_epochs: int = 200
    patience: int = 10
    weight_decay: float = 0.0

    # Architecture des modèles.
    latent_dim: int = 128
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 32, 64, 64])
    kernel_size: int = 5
    conv_stride: int = 2
    trunk_dims: List[int] = field(default_factory=lambda: [256])
    head_hidden_dims: List[int] = field(default_factory=lambda: [64])
    activation: str = "relu"
    dropout: float = 0.0

    # Instants de temps évalués séparément (sous-ensembles "t=...").
    eval_times: List[float] = field(default_factory=lambda: [0.0, 1.0, 2.0])

    # Sous-dossier des modèles et noms des rapports globaux.
    models_subdir: str = "experiment_c"
    summary_filename: str = "experiment_c_summary.json"
    config_snapshot_filename: str = "experiment_c_config_snapshot.json"

    # Seuils de sanity du pipeline (R² global sur le test, snapshot unique).
    sanity_r2_energy: float = 0.98
    sanity_r2_other_targets: float = 0.95

    # Seuils du verdict d'identifiabilité de nu*D (protocole heldout_nu).
    r_holdout_min: float = 0.8      # Spearman(pred nuD, nuD) sur nu=0.01 held-out
    rho_max: float = 2.0            # ratio rel_rmse held-out / rel_rmse in-distribution
    equiv_max: float = 0.99         # R² max entre têtes D et nu*D (équivalence)
    r_holdout_fail: float = 0.5     # en dessous -> NOT IDENTIFIABLE
    rel_rmse_fail: float = 0.5      # rel_rmse held-out au-dessus -> NOT IDENTIFIABLE


# -----------------------------------------------------------------------------
# 8b. Configuration de l'Experiment A-Sequence (window-A, Step 5)
# -----------------------------------------------------------------------------

@dataclass
class ExperimentASeqConfig:
    """
    Architecture A-Sequence (window-A) : `LatentDynamicsAutoencoder` (Architecture A)
    avec un encodeur Conv1D sur fenêtre de `window_size` snapshots (le temps entre
    comme canaux, comme c3_window), sur le dataset nu_sweep_v1.

    - protocoles : "standard" (train 3 strates de nu) et "heldout_nu"
      (train nu = `heldout_nu_values`, évaluation sur nu=0.01 du test) ;
    - évaluation de z vs nu*D avec les sous-ensembles d'Experiment C ;
    - verdict (Step 5) sur le protocole heldout_nu, strate nu=0.01 :
      IDENTIFIABLE si r_holdout >= r_holdout_min ET R²(holdout) >= r2_holdout_min
      ET pente de calibration dans [slope_min, slope_max] ET bat la baseline
      nu_oracle ET critères structurels (rang, rec, monotonie) ;
      NOT IDENTIFIABLE si r_holdout < r_holdout_fail ; PARTIAL sinon.
    """

    protocols: List[str] = field(default_factory=lambda: ["standard", "heldout_nu"])
    heldout_nu_values: List[float] = field(default_factory=lambda: [0.05, 0.20])

    # Fenêtre temporelle (w snapshots = w canaux de l'encodeur Conv1D).
    window_size: int = 25

    # Hyperparamètres d'entraînement.
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    num_epochs: int = 200
    patience: int = 10
    weight_decay: float = 0.0

    # Architecture (encodeur Conv1D + décodeur/prédicteur/tête comme Architecture A).
    latent_dim: int = 8
    conv_channels: List[int] = field(default_factory=lambda: [16, 32, 32, 64, 64])
    kernel_size: int = 5
    conv_stride: int = 2
    trunk_dims: List[int] = field(default_factory=lambda: [256])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [64, 128])
    predictor_hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    scalar_head_hidden_dims: List[int] = field(default_factory=list)
    activation: str = "relu"
    dropout: float = 0.0

    # Poids de la loss Architecture A (identiques à ceux de l'Experiment A).
    reconstruction_weight: float = 1.0
    prediction_weight: float = 1.0
    monotonicity_weight: float = 1.0
    variance_weight: float = 0.01
    min_latent_variance: float = 0.01
    monotonicity_mode: str = "squared_hinge"
    prediction_mode: str = "mse"

    # Experiment A-Physics (Step 6) : cible physique optionnelle de z.
    #   "none"            -> window-A historique (pas de terme z)
    #   "energy_centered" -> z ancré sur nuD_est_t = (E_{t-1}-E_{t+1})/(2*dt) = -dE/dt (auto-supervisé)
    #   "supervised"      -> z ancré sur nu_dissipation_t stocké (contrôle)
    z_target_mode: str = "none"
    z_weight: float = 0.0

    # Instants évalués séparément (sous-ensembles "t=...").
    eval_times: List[float] = field(default_factory=lambda: [1.0, 2.0])

    # Sorties.
    models_subdir: str = "latent_dynamics_autoencoder_seq"
    summary_filename: str = "experiment_a_seq_summary.json"
    config_snapshot_filename: str = "experiment_a_seq_config_snapshot.json"

    # Seuils du verdict (strates nu=0.01 held-out, z vs nu*D).
    r_holdout_min: float = 0.8      # Spearman(z, nuD) @ nu=0.01 held-out
    r2_holdout_min: float = 0.5     # R² affine(nuD ~ z) @ nu=0.01 held-out
    slope_min: float = 0.5          # pente de calibration nuD ~ a*z + b
    slope_max: float = 2.0
    r_holdout_fail: float = 0.5     # en dessous -> NOT IDENTIFIABLE
    # Critères structurels (hérités d'Architecture A).
    rank_min: int = 3               # rang effectif de h (PCA)
    rec_rmse_max: float = 0.15      # RMSE de reconstruction sur le test
    monotone_min: float = 0.95      # fraction de paires z_{t+1} <= z_t


# -----------------------------------------------------------------------------
# 9. Configuration globale du projet
# -----------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """
    Configuration globale du projet.

    Cette classe agrège toutes les sous-configurations dans une seule structure.
    Ainsi, dans le reste du code, on pourra écrire par exemple :

        cfg.pde.nu
        cfg.dataset.n_train
        cfg.ml.latent_dim

    ce qui rend le code plus lisible.
    """

    project_name: str = "burgers_latent_physics"
    experiment_name: str = "baseline_v1"

    paths: PathsConfig = field(default_factory=PathsConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    reproducibility: ReproConfig = field(default_factory=ReproConfig)
    nu_sweep: NuSweepConfig = field(default_factory=NuSweepConfig)
    experiment_c: ExperimentCConfig = field(default_factory=ExperimentCConfig)
    experiment_a_seq: ExperimentASeqConfig = field(default_factory=ExperimentASeqConfig)

    def create_directories(self) -> None:
        """
        Crée tous les dossiers nécessaires pour l'expérience.
        """
        self.paths.create_directories()

    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit récursivement la configuration en dictionnaire Python.

        Utile pour :
        - sauvegarder la configuration dans un fichier JSON,
        - logger les expériences,
        - inspecter facilement les paramètres.
        """
        data = asdict(self)

        # Les objets Path sont convertis en chaînes pour être sérialisables facilement.
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            return obj

        return convert_paths(data)


# -----------------------------------------------------------------------------
# 10. Fonction utilitaire principale
# -----------------------------------------------------------------------------

def get_default_config(create_dirs: bool = True) -> ProjectConfig:
    """
    Construit et retourne la configuration par défaut du projet.

    Paramètres
    ----------
    create_dirs : bool
        Si True, crée automatiquement les dossiers de sortie.

    Retour
    ------
    ProjectConfig
        Objet de configuration prêt à être utilisé dans les scripts.
    """
    cfg = ProjectConfig()

    if create_dirs:
        cfg.create_directories()

    return cfg


# -----------------------------------------------------------------------------
# 11. Bloc de test simple
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = get_default_config(create_dirs=True)

    print("Configuration du projet chargée avec succès.\n")
    print("Nom du projet :", cfg.project_name)
    print("Nom de l'expérience :", cfg.experiment_name)
    print("Viscosité nu :", cfg.pde.nu)
    print("Domaine L :", cfg.pde.L)
    print("Nombre de points Nx :", cfg.pde.Nx)
    print("Pas d'espace dx :", cfg.pde.dx)
    print("Nombre total de pas de temps :", cfg.pde.n_steps)
    print("Nombre de snapshots sauvegardés :", cfg.pde.n_saved_steps)
    print("Dossier des données :", cfg.paths.data_dir)
    print("Dimension latente :", cfg.ml.latent_dim)
    print("nu_sweep Nx :", cfg.nu_sweep.Nx)
    print("nu_sweep dt :", cfg.nu_sweep.dt)
    print("nu_sweep save_every :", cfg.nu_sweep.save_every)
    print("nu_sweep nu_values :", cfg.nu_sweep.nu_values)
    print("nu_sweep train_filename :", cfg.nu_sweep.train_filename)
    print("experiment_c variants :", cfg.experiment_c.variants)
    print("experiment_c protocols :", cfg.experiment_c.protocols)
    print("experiment_c heldout_nu_values :", cfg.experiment_c.heldout_nu_values)
    print("experiment_c window_size :", cfg.experiment_c.window_size)
    print("experiment_c window_sizes :", cfg.experiment_c.window_sizes)
