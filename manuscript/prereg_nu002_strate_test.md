# Pré-enregistrement v4 — « nouvelle strate test ν = 0.02 » (COMMITÉ SEUL, AVANT TOUTE GÉNÉRATION)

**Statut** : ce fichier est commité SEUL, avant tout code de génération et avant toute
génération de données (même règle que `outputs/prereg/td_encoder_prereg.md:3`).
Aucune génération, aucun entraînement, aucune lecture de la nouvelle strate n'a été
faite avant ce commit ; elles restent soumises au **feu vert explicite** de
l'utilisateur. v4 = 4 états de verdict mutuellement exclusifs, sd a priori unique,
plages sur moyenne des 5 graines avec biais déclaré. Règles permanentes inchangées :
`main_generate_data.py` jamais lancé depuis le sandbox ; npz de `outputs/data/` en
lecture seule (`allow_pickle=True`) ; génération et inférence torch côté
Windows/PowerShell uniquement.

**Objectif** : strate de test à ν = **0.02** (choix fixé par l'utilisateur), valeur
jamais vue (train/test existants : ν ∈ {0.01, 0.05, 0.20}), indépendante de la strate
ν = 0.01 consultée de nombreuses fois depuis le Step 5 (README.md:24, 1220).
Évaluation **sans réentraînement** (inférence seule) des checkpoints existants
TD0 / TD1 / TD3, 5 graines × 2 modes, pour confirmer la campagne TD sur strate vierge.

**Analyses préalables read-only réalisées pour ce brouillon** (sandbox, numpy, aucun
artefact touché, log `/tmp/analysis_v3.txt` hors dépôt) : recomputation depuis les
`predictions_test.npz` commités — reproduction exacte du JSON (`max|écart| = 0.0` sur
les **40 runs z au total** (20 par protocole : 4 bras × 5 graines)) et des IC de
calibration publiées (D2 cal +0.0399 [+0.0112,+0.0687] ; D1 cal +0.0092
[−0.0476,+0.0660], README.md:1248-1250). Corrélation estimée : ρ(r² identité, r²
calibré) = **0.9903** run-level (n=20 runs heldout_nu z, ν=0.01) et **0.9905** sur les
différences appariées poolées (n=30) → **ρ retenu = 0.99**.

---

## §1. Choix de ν = 0.02 — critères fixés AVANT génération, valeurs calculées

Candidats examinés : interpolation 0.02 / 0.10, extrapolation 0.005.
**Choix retenu : ν = 0.02** (interpolation entre 0.01 et 0.05). Critères a priori,
avec dx = 2π/1024 = 6.1359e-3, dt = 1e-4, u_max observé 1.8528
(`data_generation_report_nu_sweep_train.json`) :

| # | Critère | Seuil / source | ν=0.02 | ν=0.10 | ν=0.005 |
|---|---|---|---|---|---|
| C1 | Nombre diffusif ν·dt/dx² = ν·2.656 (0.531 à ν=0.20 ; limite RK4 ≈ 2.78) | ≤ 0.531 (README.md:317 ; config.py:338) | **0.053 ✓** | 0.266 ✓ | 0.013 ✓ |
| C2 | CFL advectif u_max·dt/dx | ≤ 0.030 (README.md:317) | 0.030 ✓ (limite, identique au cas ν=0.20) | ✓ | ✓ |
| C3 | Péclet de maille max = u·dx/ν (gate 2.0 ; mesuré 1.14) | ≤ 2.0 (config.py:380) | **0.569 ✓** | 0.114 ✓ | 2.27 ✗ à Nx=1024 → ladder Nx=2048, dt=2.5e-5 (config.py:371) |
| C4 | Magnitude νD vs train : max νD(strate) vs plage train (max train complet 1.8121, README.md:26-28, 885-886) | mesuré sur pilote 20 traj | à mesurer | à mesurer | à mesurer |
| C5 | Coût génération | pilote chronométré × 7.5 (150 traj) | **×1** | ×1 | ×~8 |

ν=0.02 passe C1-C3 à Nx=1024, coût ×1 → retenu. C4 mesuré sur le pilote avant
génération complète (gate GO/NO-GO §3).

## §2. Spec de génération (strictement additive)

- Split **test_extra** : 150 trajectoires, ν = 0.02 uniquement.
- Mêmes familles de CI round-robin (fourier / gaussian_bumps / random_smooth),
  `zero_mean=True` toutes familles (config.py:329) ; `target_l2_norm`
  log-uniform(0.5, 2.0) (config.py:326) → E(0) = ½·norm² ∈ [0.125, 2.0].
- **Seeds disjoints** des 900 existants : bloc `default_rng(100000+offset)`,
  enregistrés dans `trajectory_seeds`, check « no overlap » (pde_dataset_builder.py:866-868).
- Numérique identique : Nx=1024, dt=1e-4, save_every=200 → 101 snapshots, RK4,
  L=2π, T=2.0, dt_saved=0.02 (config.py:331-348).
- Mêmes clés npz + `dataset_version="nu_sweep_v1_test_extra_nu0p02"` ; fichier
  `outputs/data/test_extra_data_nu002.npz` — **aucun fichier existant modifié**
  (abort si le nom existe).
- Point d'entrée **additif** `main_generate_data_nu_extra.py` (commité séparément,
  avec smoke sans écriture dans `outputs/`) ; `main_generate_data.py` par défaut ne
  touche pas aux npz existants.
- Commandes (Windows, PowerShell, python3, racine repo) :
  `python3 main_generate_data_nu_extra.py --pilot 20 --n 0 --report outputs/logs/nu002_pilot_report.json`
  puis `python3 main_generate_data_nu_extra.py --pilot 0 --n 150 2>&1 | Tee-Object -FilePath outputs/logs/gen_nu002.log` (log UTF-16).

## §3. Audit du npz AVANT toute évaluation (gates Step 3, README.md:310-335)

- SHA-256 (`Get-FileHash ... -Algorithm SHA256`) consigné ; rapport JSON commité.
- Gates : `all_finite` ; monotone fraction ≥ 0.99 (atol 1e-8·E0), pass ≥ 0.95 ;
  max|u| ≤ 5.0 ; médiane |dE/dt+νD|/(νD) ≤ 1e-2 ; max cell Péclet ≤ 2.0 (attendu ≈ 0.57,
  §1 C3) ; mass0 ≤ 3.5e-15 ; energy0 = ½·target² ; seeds uniques sans overlap ;
  C4 : max νD(strate) rapporté vs plage train (1.8121) — hors plage autorisé mais
  rapporté et stratifié (§7). Tout gate KO → NO-GO, rien n'est évalué.

## §4. Bras, protocoles, statuts confirmatoire/exploratoire ; comparaisons ; export validation

- **Bras évalués** : TD0, TD1, TD3 (TD2 **exclu** : D1 confondue par le pooling
  global, README.md:1212-1215). Modes `energy_centered` (z) + `none`. Graines
  {42, 7, 123, 2024, 999}. → 60 runs évalués en inférence seule.
- **Checkpoints existants** (`91199bc`, SHA-256 dans
  `outputs/logs/td_corrected/td_artifacts_sha256.txt`), **hash avant/après** (lecture
  seule prouvée). Aucun réentraînement, aucun `.lock`, aucune écriture dans
  `outputs/models/` ; exports additifs `predictions_test_extra_nu002.npz`.
- **Protocoles et statuts à ν = 0.02** :
  - `heldout_nu005` (train/val ν ∈ {0.01, 0.20}, `td_encoder_prereg.md:42-43`) :
    0.01 < 0.02 < 0.20 → **interpolation en ν** → **CONFIRMATOIRE** (15 runs z + 15 none).
  - `heldout_nu` (train/val ν ∈ {0.05, 0.20}) : 0.02 < 0.05 → **extrapolation sous la
    plage train** → **EXPLORATOIRE** (transfert ; 15 runs z + 15 none). Les verdicts
    ne portent que sur le protocole confirmatoire.
- **Comparaisons** (appariement strict par graine, n = 5, ν = 0.02) :
  - **D1' = TD1 − TD0** — CONFIRMATOIRE (canaux de différences temporelles seuls).
  - **D3' = TD3 − TD0** — EXPLORATOIRE, étiqueté INJECTION (canal produit u_t⊙Δ_1u,
    prereg md:26-28).
  - TD3 − TD1 — descriptif.
  - **D3 = TD3 − TD2 n'est pas calculable** (TD2 non évalué ici).
- **Export des prédictions de VALIDATION** (pour la calibration hors strate, §5) :
  `main_export_predictions.py` n'exporte que le split test (`split="test"`, ligne 89 ;
  val non exporté par le pipeline, `ml_train_experiment_a_seq.py:1182-1183`). Prévu :
  script **additif** `main_export_predictions_val.py` (commité séparément), inférence
  seule (lit `best_model.pt` + `standardizers.json`, aucun réentraînement), côté
  Windows, feu vert requis ; exports `predictions_val.npz` dans `outputs/predictions_val/`
  (jamais dans les répertoires de runs), hash des checkpoints avant/après, SHA-256 des
  exports + manifeste commités. **Si cet export est refusé ou impossible : la
  co-primaire (ii) est rétrogradée en proxy** (fit sur strates test autres que ν=0.02,
  comme README.md:1248-1250), déclaré en **limite** — jamais d'ajustement sur la strate
  évaluée ν=0.02.

## §5. Métriques, 4 états de verdict mutuellement exclusifs, puissances

- **Co-primaires déclarées a priori** (les deux requises) :
  (i) **R² identité** `1−Σ(z−y)²/Σ(y−ȳ)²` sur la strate ν=0.02 (définition amendée
  `182c8fc`, `main_evaluate_td.py:88-89`) ;
  (ii) **R² après calibration hors strate** : fit affine y ~ a·z+b sur les prédictions
  de **validation** exportées (§4), appliqué aux fenêtres ν=0.02 ; métrique = R²
  identité sur prédictions calibrées (catégorie 3, README.md:1183-1186). Jamais de
  fit sur la strate évaluée.
- **Agrégeage conjonctif** : L = min(borne inf. id, borne inf. cal) ; U = max(borne sup.
  id, borne sup. cal) ; IC unilatéral 95 %, t ddl=4 (t=2.1318), n=5, appariement par
  graine, δ_R = 0.05.
- **4 états mutuellement exclusifs et exhaustifs** (vérifié par MC : somme = 1.000 à
  Δ=0 et Δ=0.05) :
  - **E1 — POSITIF-BORNÉ** : L > 0 et U < δ_R → « effet > 0 démontré, et < δ_R »
    (cas attendu pour D1' : +0.027, L ≈ +0.023 > 0, U ≈ +0.031 < 0.05) ;
  - **E2 — POSITIF-NON-BORNÉ** : L > 0 et U ≥ δ_R → « effet > 0 démontré, magnitude
    ≥ δ_R non exclue » ;
  - **E3 — NÉGATIF** : L ≤ 0 et U < δ_R → « effet ≥ δ_R exclu, signe non établi » ;
  - **E4 — INDÉTERMINÉ** : L ≤ 0 et U ≥ δ_R.
  **La question confirmatoire « y a-t-il un effet > 0 ? » est répondue par E1 ∪ E2
  (L > 0 sur les deux co-primaires)** ; E3 répond à « pas d'effet actionnable ≥ δ_R ».
  La règle « mean ≥ δ_R » reste abandonnée (plafond P=0.5, README.md:1302-1304).
- **Une seule sd a priori pour le dimensionnement : sd_diff = 0.029** (conservatrice,
  sd historique D2 à ν=0.01) ; la dispersion empirique nu005 de D1' (0.0043/0.0062, §6)
  est **informative seulement**, non utilisée pour le dimensionnement.
- **Puissances (Monte-Carlo numpy, seed 20261123, sandbox, dépôt intact ; post-hoc
  assumé, aucune puissance a priori du prereg TD modifiée, cf. README.md:1203)** :
  - métrique seule, sd=0.029, n=5 : se=0.012969, MDE80 = **0.0399**, power à Δ=0.05 =
    0.931, P(POSITIF|Δ=0) = 0.05 ;
  - **joint, sd_id = sd_cal = 0.029, ρ = 0.99** (estimé sur runs TD existants) :
    **MDE80 joint (E1∪E2) = 0.0405** ; à Δ=0.05 : E1 = 0.033, E2 = 0.889, E3 = 0.009,
    E4 = 0.068 → P(POSITIF joint) = **0.922** ; à Δ=0 : E1 = 0.0335, E2 = 0.0094,
    E3 = 0.8885, E4 = 0.0685 → **FWER joint = 0.0429 ≤ 0.05** ;
  - référence ρ=0 : MDE80 joint = 0.047 → ρ ≈ 1 ne coûte presque rien (0.0405 vs 0.0399).

## §6. Prédictions chiffrées à ν = 0.02, par protocole, mêmes checkpoints, plages resserrées

Dérivation a priori : pour **chaque protocole séparément**, interpolation linéaire en ν
(poids w = 0.25) entre les R² identité à ν=0.01 et ν=0.05 **des mêmes checkpoints**
(recalculées depuis leurs `predictions_test.npz` commités — reproduction exacte du
JSON vérifiée, écart 0.0), par graine ; valeur à ν=0.02 d'une graine =
0.75·R²(0.01)+0.25·R²(0.05). **Les plages portent sur la MOYENNE des 5 graines** :
plage = moyenne ± 2·sd inter-graines de la valeur interpolée par graine (les valeurs
par run sont rapportées à part). Justification de la largeur : mean ± 2·sd couvre
~95 % de la distribution des moyennes de runs sous normalité, sans aller jusqu'à
l'intervalle de prédiction complet (≈ mean ± 3.04·sd) — compromis falsifiabilité /
prudence déclaré a priori ; R² plafonnée à 1.0. **Biais déclaré** : pour heldout_nu005,
l'interpolation mélange ν=0.01 **in-train** (valeurs hautes, ex. TD0 0.9513) et ν=0.05
held-out → biais vers le haut des prédictions ; pour heldout_nu, c'est l'inverse
(ν=0.01 held-out, ν=0.05 in-train). **Une valeur observée hors plage étroite teste
D'ABORD l'hypothèse de linéarité en ν** (recalculer en log-ν ou avec le voisin le plus
proche) avant toute conclusion de falsification.

**Protocole CONFIRMATOIRE heldout_nu005** (ν=0.01 in-train, ν=0.05 held-out ; biais haut) :

| quantité | ν=0.01 (mêmes runs) | ν=0.05 (held-out) | prédiction ν=0.02 (moy ± sd) | plage moy ± 2 sd |
|---|---|---|---|---|
| R² TD0 | 0.9513±0.0035 | 0.9238±0.0084 | **0.9444±0.0038** | [0.937, 0.952] |
| R² TD1 | 0.9799±0.0010 | 0.9476±0.0041 | **0.9718±0.0016** | [0.969, 0.975] |
| R² TD3 | 0.9957±0.0009 | 0.9883±0.0052 | **0.9938±0.0015** | [0.991, 0.997] |
| D1' = TD1−TD0 | +0.0286 | +0.0238 | **+0.0274±0.0043** | [+0.019, +0.036] |
| D3' = TD3−TD0 | +0.0444 | +0.0645 | **+0.0494±0.0049** | [+0.040, +0.059] |
| TD3−TD1 (descriptif) | +0.0158 | +0.0407 | **+0.0220±0.0015** | [+0.019, +0.025] |

Attendu confirmatoire : D1' = +0.027 → L ≈ +0.0233 > 0, U ≈ +0.0315 < 0.05 → **E1
POSITIF-BORNÉ attendu** (effet > 0 mais < δ_R, rapporté descriptif vs δ_R) ; D3' = +0.049
→ L ≈ +0.0447 > 0, U ≈ +0.0541 ≥ 0.05 → **E2 POSITIF-NON-BORNÉ attendu**
(exploratoire/injection). Mode none : |Spearman| < 0.3. C6'a attendu échec (historique
1.216, json:1564-1566) — exploratoire. Écart au plafond R²(TD3)−0.9995 rapporté.

**Protocole EXPLORATOIRE heldout_nu** (ν=0.01 held-out, ν=0.05 in-train) :

| quantité | ν=0.01 (held-out) | ν=0.05 (mêmes runs) | prédiction ν=0.02 (moy ± sd) | plage moy ± 2 sd |
|---|---|---|---|---|
| R² TD0 | 0.8747±0.0201 | 0.9641±0.0071 | **0.8971±0.0165** | [0.864, 0.930] |
| R² TD1 | 0.9057±0.0241 | 0.9889±0.0022 | **0.9265±0.0176** | [0.891, 0.962] |
| R² TD3 | 0.9773±0.0053 | 0.9951±0.0013 | **0.9818±0.0040** | [0.974, 0.990] |
| D1' = TD1−TD0 | +0.0310 | +0.0248 | **+0.0294±0.0220** | [−0.015, +0.073] |
| D3' = TD3−TD0 | +0.1026 | +0.0310 | **+0.0847±0.0173** | [+0.050, +0.119] |
| TD3−TD1 (descriptif) | +0.0716 | +0.0062 | **+0.0553±0.0202** | [+0.015, +0.096] |

**Hors plage = falsification rapportée, pas STOP** : évaluation en inférence seule,
sans réentraînement ; la clause STOP du prereg TD §8 (prereg md:115-116) concernait
une référence réentraînée. Toute valeur hors plage est publiée telle quelle, « falsifié »
(après check de linéarité en ν, ci-dessus).

## §7. Plage de magnitude de νD et stratification des métriques

- Pour chaque protocole, plage [min, max] de νD sur les fenêtres du **train** du
  protocole calculée depuis les npz (lecture seule) : heldout_nu train ν∈{0.05,0.20} ;
  heldout_nu005 train ν∈{0.01,0.20}. Référence : max νD test complet 1.4509 < max train
  complet 1.8121, 0 % hors plage à ν=0.01 (README.md:26-28, 885-886).
- Strate ν=0.02 (150 traj × 76 fenêtres w=25 = 11 400 fenêtres) : fraction de fenêtres
  **hors plage** de νD du train de chaque protocole — inconnue a priori (νD à ν=0.02
  ≈ 2× les valeurs ν=0.01 à D comparable).
- **Métriques stratifiées** in-plage / hors-plage (R² identité, R² calibrée, Spearman,
  pente, rel_rmse, M6), même esprit que `outputs/logs/aphysis_stratifications.json`
  (README.md:885-886). Le verdict (§5) porte sur la strate complète ; la version
  restreinte in-plage est rapportée à côté, exploratoire.

## §8. Traçabilité et reproductibilité

- Ce prereg commité **avant** la génération ; git hash + dirty=False + versions
  (torch/numpy/python) dans chaque log ; SHA-256 du npz, des checkpoints lus
  (avant/après) et des exports de prédictions (val + test_extra) ; `python3
  td_protocols.py --check` et `python3 main_determinism_check.py` exit 0 avant toute
  évaluation ; logs PowerShell UTF-16 (README.md:1341-1343) ; commandes canoniques
  consignées (README.md:1345-1360).
- Convention plan v2 (`0a19572`) : aucune comparaison ancien code / nouveau code dans
  le texte principal (README.md:1105-1113) ; TD0 corrigé 0.8747±0.0201 reste la
  référence à ν=0.01.

## §9. Statut

**Ce prereg est commité SEUL (ce commit).** Ordre après feu vert explicite :
(1) commit séparé des points d'entrée additifs (`main_generate_data_nu_extra.py`,
`main_export_predictions_val.py`) avec smoke sans écriture dans `outputs/` ;
(2) pilote 20 traj + C4 + gates §3 → GO/NO-GO ; (3) génération 150 traj + audit +
commit rapport/SHA ; (4) export des prédictions de validation des 60 runs (inférence
seule, §4) ; (5) évaluation seule §4-§7 côté Windows.
