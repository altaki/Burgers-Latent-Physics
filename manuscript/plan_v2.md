# Plan de manuscrit v2 — Burgers-Latent-Physics (VALIDÉ)

**Statut** : plan de rédaction uniquement. Aucun run, aucune génération de données,
aucun artefact existant modifié. HEAD attendu `4c2c402`, tag `v1.0-preservation`.
Le brouillon de pré-enregistrement « nouvelle strate test »
(`manuscript/prereg_nouvelle_strate_test_BROUILLON.md`) reste **non commité** et
inactif sans feu vert explicite.

---

## Titre

**« Testing whether latent scalars encode physical observables: identifiability,
analytic ceilings and injection in 1D viscous Burgers »**

Variante courte : « Latent scalars in 1D viscous Burgers: not discovered without
supervision, preserved when injected, still below the two-snapshot ceiling ».

La portée est strictement Burgers 1D (voir Limites) : aucun « generalization » dans le titre.

---

## Conventions de rédaction (ajustements validés, à appliquer partout)

1. **C2 vs aphys** — formuler : « en performance de z, C2 (1 graine, ancien code) tombe
   dans l'IC bootstrap des 3 runs aphys — indistinguables » (README.md:855-857).
   « L_z concentre h : **participation ratio** 1.6–2.5 (aphysis) vs 3.7–5.7 (none),
   protocole heldout_nu » (README.md:892). **Toujours nommer la métrique de rang** :
   participation ratio (PR) vs rang 95 % — B1 C5 utilise le **rang 95 % = 5.667**
   (README.md:976 ; correction `5e6f24e`, README.md:1026-1032), pas le PR.
2. **Chiffres de référence** : valeur principale = **TD0 corrigé, R² identité
   0.875 ± 0.020 (0.8747 ± 0.0201), code déterministe, 5 graines**
   (`outputs/prereg/td_encoder_prereg_amendment_1.md:106` ; README.md:1147 ;
   `outputs/logs/td_encoder_results.json` groupe `td0_heldout_nu_energy_centered`).
   Step 6 (R² 0.844, moyenne 0.8438 sur 3 graines, ancien code) en **background
   seulement** (`td_encoder_prereg_amendment_1.md:37` ; README.md:844-848).
   **Ne jamais comparer de chiffres ancien code vs nouveau code dans le texte
   principal** (règle du dépôt : README.md:1105-1113).
3. **Sous-puissance** : elle concerne le verdict **REPR** (plafond P(REPR|Δ=δ_R) ≤ 0.5,
   = 0.50 D2 / 0.38 D1), **pas** le verdict NON-REPR (P(NON-REPR|Δ=0) = **0.93** pour
   D2, 0.46 pour D1 ; MDE80 = 0.062 D2 / 0.082 D1, n=5) — README.md:1203-1210.
   D1 est sous-puissant à cause de la **variance de TD2** (sd 0.0599, causée par le
   collapse de reconstruction induit par le pooling), pas à cause de la règle —
   README.md:1212-1215.
4. **Paragraphe main text obligatoire** (changement de métrique primaire) :
   « selon le texte original du prereg (R² affine, `td_encoder_prereg.md:68`),
   D1 serait REPRÉSENTATIONNEL de justesse (+0.0503, marge +0.0003) ; la métrique
   primaire a été changée en R² identité par l'amendement `182c8fc`, rédigé **après
   lecture de TD0 seul, avant TD1–TD3** ; les verdicts publiés suivent la définition
   amendée et la sensibilité est rapportée en tableau » — README.md:1238-1256 ;
   `td_encoder_prereg_amendment_1.md` §0.
5. **Limites à inclure** : (a) strate test ν=0.01 consultée de nombreuses fois depuis
   le Step 5 (README.md:24, 1220) ; (b) **aucune baseline externe runnée**
   (SINDy, autoencodeurs de Koopman, etc. discutés qualitativement seulement ;
   seules les sondes linéaires internes existent, README.md:1075) ; (c) portée
   strictement Burgers 1D, T=2, 3 valeurs de ν, n≤5 graines (README.md:22-25,
   163-169).
6. **Table de citations fichier:ligne complète en annexe** (voir Annexe A ci-dessous).

---

## Structure du manuscrit

### 1. Introduction
- Problème : dégénérescence à ν fixé et E(0) constant — tout scalaire décroissant
  corrèle avec D, la corrélation « z ≈ dissipation » y est infalsifiable
  (README.md:163-165). Leçon de design : le dataset nu_sweep_v1 a été construit pour
  lever cette dégénérescence (README.md:114-124 ; config.py:303-364).
- Baseline analytique comme plafond obligatoire : estimateur 2 snapshots
  backward_w25, R² identité 0.9995 à ν=0.01 held-out
  (`outputs/logs/experiment_a_seq_analytic_baseline.json:162,184` ;
  README.md:15-16). Toute revendication sur des observables latentes doit être
  comparée à la formule fermée quand elle existe.
- Travaux connexes (à rédiger ; aucune baseline externe runnée — cf. Limites) :
  SINDy, autoencodeurs de Koopman, sondes linéaires d'interprétabilité,
  approches physics-informed.
- Question du papier : un scalaire latent z encode-t-il l'observable physique νD
  à ν held-out, sans injection ? avec injection ? et à quelle distance du plafond
  analytique reste l'encodeur appris ?

### 2. Méthodes
- Données : nu_sweep_v1, spec + SHA-256 (README.md:114-124, 1320-1328) ;
  identité exacte dE/dt = −νD dans les données (R² = 1.0, README.md:150).
- Modèles : table README.md:126-133 ; perte d'ancrage L_z = MSE(z_t,
  (E_{t−w}−E_t)/(w·dt)) (README.md:693).
- Protocoles : heldout_nu (train {0.05,0.20}, éval ν=0.01) et heldout_nu005
  (train {0.01,0.20}, éval ν=0.05) — `td_encoder_prereg.md:40-43`, `config.py:417`.
- Bras TD0–TD3 (prereg `td_encoder_prereg.md:18-27`) ; hyperparamètres figés
  (`:44-53`) ; anti-sélection test (`:54-58`).
- Règle de décision machine (prereg md `:78-104` ; `td_encoder_prereg.json`
  `/decision_rule` : δ_R = 0.05, IC unilatéral 95 %, t ddl=4 = 2.1318, n=5,
  appariement par graine) + **un paragraphe main text** sur la définition du R²
  (identité) et son amendement (convention 4 ci-dessus ; inventaire des 3
  catégories de R² : README.md:1171-1200).
- Métriques : R² identité (évaluations), Spearman, pente de calibration, rel_rmse,
  M6, rang de h (nommer PR vs rang 95 %), C6'a/C6'b (prereg md `:106-112`).

### 3. Résultats
- **3.1 Sans injection : échec robuste à ν held-out.** wina none, 3 graines
  s42/s7/s123 : Spearman 0.249 / 0.438 / 0.327 à ν=0.01 (README.md:851, 1075) ;
  νD présent dans h sous forme distribuée (sonde linéaire 0.53–0.66,
  README.md:1075 ; table par run README.md:675-680) mais non sélectionné par z.
- **3.2 Avec injection : préservation.** Step 6 aphys (background, ancien code) ;
  référence principale TD0 corrigé R² 0.875 ± 0.020 (convention 2) ; A1 verdict
  TRACEUR DYNAMIQUE 6/6 critères (README.md:919 ; M1_raid 0.974, M2 0.913,
  README.md:921-927) ; B1 interpolation ν=0.05 : Spearman 0.985, R² 0.927, verdict
  PARTIAL, C6 seul échec (README.md:969-979). C2 vs aphys selon convention 1.
- **3.3 Écart au plafond analytique et campagne TD (une section).**
  Verdicts D1/D2 INDÉTERMINÉS, D3 REPRÉSENTATIONNEL
  (`td_encoder_results.json:1480-1546` ; README.md:1128-1136) ; table par bras
  (README.md:1146-1150) ; écart au plafond 0.1248 → 0.0222 (TD3)
  (README.md:1163-1165) ; C6' (json `:1550-1621` ; README.md:1138-1140) ;
  bilan des prédictions prereg §8 (README.md:1152-1162 ; TD3 falsifié de 0.013,
  README.md:1159) ; borne sup D1 +0.054 présentée comme information
  (json `:1498`) ; **tableau de sensibilité 3 variantes du R²**
  (README.md:1245-1256). Puissance selon convention 3.

### 4. Discussion
- Ce qui est démontré / seulement suggéré / non démontré (matrice explicite) :
  la préservation sous injection est démontrée ; la « découverte » sans injection
  est réfutée (dans ce système) ; la cause de l'écart au plafond reste ouverte
  (README.md:1266-1268).
- « Pourquoi apprendre si une formule fermée fait mieux ? » : Burgers comme banc
  d'essai contrôlé où la vérité est connue (identité exacte), pas comme cas d'usage.
- Portée du transfert : ν uniquement, magnitudes dans la plage d'entraînement
  (0 % de fenêtres test hors plage, max νD test 1.4509 < max train 1.8121,
  README.md:26-28, 885-886).

### 5. Limites (section dédiée)
- Conventions 3 et 5 (puissances, strate test ν=0.01 relue, pas de baseline externe,
  portée Burgers 1D) ; n=5 graines ; dataset unique ; C6 seuil 0.15 calibré sur
  legacy Nx=128 (README.md:983-998) ; non-déterminisme historique s42
  (rec 0.1775 vs 0.1663, même commit `212db0f`, README.md:870-876) vs déterminisme
  bit à bit de la campagne TD (`outputs/logs/td_corrected/td_determinism_check_postfix.log`,
  README.md:1335-1338) ; run 964 min (README.md:1217-1218) ; D1 confondu
  (pooling, README.md:1212-1215).

### 6. Supplementary (gouvernance, hors texte principal)
- Preregs commités (`outputs/prereg/*`), amendment 1 + erratum, note d'archivage
  `td0_indist_nu001_*` ; écarts au prereg (README.md:1228-1268) ; bug de filtre
  `main_train_td.py:48` et preuve à 8 décimales (README.md:1098-1115) ; inventaire
  de reproductibilité (SHA-256 des npz et des 300 artefacts TD, lock
  d'environnement, commandes canoniques PowerShell, encodage UTF-16 des logs,
  mapping résultat→artefact/commit, note clé `r2_identity`) — README.md:1312-1382 ;
  `outputs/logs/td_corrected/td_artifacts_sha256.txt` (300 lignes).
- Proposition TD2' : perspective uniquement, **non pré-enregistrée**, à ne lancer
  que sur feu vert avec nouveau prereg et règle de puissance corrigée
  (README.md:1293-1310).

---

## Annexe A — Table de citations fichier:ligne (complète)

| Affirmation / chiffre | Source |
|---|---|
| Question du dépôt (Burgers 1D périodique) | `README.md:3-5` |
| Résultat central « préservation, pas découverte » | `README.md:10-13` |
| Plafond analytique R² identité 0.9995 @ν=0.01 (backward_w25, 0.9994744418161681) | `outputs/logs/experiment_a_seq_analytic_baseline.json:162,184` ; `README.md:15-16` |
| Campagne TD (80 runs) en synthèse | `README.md:17-21` |
| Négatifs, limites, portée ; strate ν=0.01 consultée depuis Step 5 | `README.md:22-28` (ligne 24) |
| Méthodologie : règle `git show --stat`, inventaire 3 catégories R² | `README.md:29-37` |
| Spec dataset nu_sweep_v1 (Nx=1024, dt=1e-4, save_every=200, RK4, ν∈{0.01,0.05,0.20}, 600/150/150, zero_mean, log-uniform(0.5,2.0)) | `README.md:114-124` ; `config.py:303-364` |
| Table modèles (params, objectifs, résultats legacy) | `README.md:126-133` |
| Audit latent rank-1 ; dE/dt = −νD exact (R²=1.0) | `README.md:142-150` (ligne 150) |
| Interprétation (z = étiquette statique…) | `README.md:152-161` |
| Dégénérescence à ν fixé ; E(0) constant | `README.md:163-165` |
| Step 5 window-A mono-graine (0.249 w=25, −0.035 w=50) | `README.md:650-651` |
| C2 supervisé référence (Spearman 0.973, R² 0.69 @ν=0.01) | `README.md:653,788` |
| Perte d'ancrage proposée L_z = MSE(z_t,(E_{t−w}−E_t)/(w·dt)) | `README.md:693` |
| wina none 3 graines : Spearman 0.249/0.438/0.327 | `README.md:851,1075` |
| Sonde linéaire h 0.53–0.66 ; table par run (0.655/0.09 w=25 heldout) | `README.md:1075 ; 675-680` |
| Step 6 table aphys (0.859/0.791/0.881 — ancien code, background) | `README.md:844-848` ; moyenne 0.8438 : `td_encoder_prereg_amendment_1.md:37` |
| C2 (1 graine) dans l'IC bootstrap des 3 runs aphys | `README.md:855-857` |
| Pente z~νD 0.87–0.96 ; médian z/D ≈ 0.013 ≈ ν | `README.md:863-865` |
| asup = aphys, cibles identiques à 0.35 % | `README.md:866-867` |
| Non-déterminisme historique s42 (rec 0.1775 vs 0.1663, commit 212db0f) | `README.md:870-876` |
| Participation ratio h : aphys 1.6–2.5 vs none 3.7–5.7 (heldout_nu) | `README.md:892` |
| Sonde h linéaire @ν=0.01 = 0.96–0.97 | `README.md:894` |
| A1 : verdict TRACEUR DYNAMIQUE, 6/6 critères ; M1_raid 0.974, M2 0.913, M4=1 pas, M6=0.81 | `README.md:919,921-927` |
| B1 protocole heldout_nu005 | `README.md:962-967` |
| B1 verdict PARTIAL ; C5 = **rang 95 %** 5.667 (pas PR) ; C6 seul échec (0.196>0.15) | `README.md:969-979` ; correction `5e6f24e` : `README.md:1026-1032` |
| B1 C6 : plancher PCA k=8 = 0.188 ; seuil 0.15 calibré legacy Nx=128 | `README.md:983-998` |
| Open questions répondues (window-A, A-Physics, A1, B1) | `README.md:1063-1096` |
| Step 8 : bug filtre `main_train_td.py:48`, preuve 8 décimales, archivage ; règle « jamais comparer ancien/nouveau code » | `README.md:1098-1115` |
| Step 9 protocole (80 runs, git=182c8fc8, SHA-256) | `README.md:1118-1126` |
| Verdicts D1 −0.0034 [−0.0606,+0.0537] INDET ; D2 +0.0310 [+0.0034,+0.0585] INDET ; D3 +0.1060 [+0.0433,+0.1687] REPR | `outputs/logs/td_encoder_results.json:1480-1500,1503-1523,1526-1546` ; `README.md:1128-1136` ; `verdict_principal_D1` json:1624 |
| C6'a/C6'b par bras (td0/1/2 false, td3 C6'a true) | `td_encoder_results.json:1550-1621` ; `README.md:1138-1140` |
| Table par bras ν=0.01 : R² 0.8747±0.0201 / 0.9057±0.0241 / 0.8713±0.0614 / 0.9773±0.0053 ; ν=0.05 : 0.9238/0.9476/0.9765/0.9883 | `README.md:1146-1150` ; ré-agrégation depuis json `runs` |
| Bilan prereg §8 : TD0 CONFIRMÉ ; TD3 FALSIFIÉ de 0.013 (attendu ≥0.99) | `README.md:1152-1162` (ligne 1159) ; prereg md:115,118-119 |
| Écart au plafond 0.1248/0.0938/0.1282/0.0222 | `README.md:1163-1165` |
| Inventaire 3 catégories de R² (identité / Pearson² in-sample / affine train→test) | `README.md:1171-1200` |
| Puissances post-hoc : P(REPR|Δ=δ_R)=0.50/0.38 ; P(NON-REPR|Δ=0)=0.93/0.46 ; MDE80 0.062/0.082 | `README.md:1203-1210` |
| D1 confondu (pooling global, rec 0.746 vs 0.197) | `README.md:1212-1215` |
| Strate test ν=0.01 relue depuis Step 5 | `README.md:1220` (et 24) |
| Écarts au prereg : métrique changée par amendement 182c8fc après lecture TD0, avant TD1–TD3 | `README.md:1238-1240` |
| Sensibilité 3 variantes R² : Pearson² in-sample D1 +0.0503 [+0.0366,+0.0641] REPR de justesse | `README.md:1245-1256` (ligne 1249) |
| Erratum en-tête BROUILLON dans 182c8fc | `README.md:1265-1267` ; `outputs/prereg/td_encoder_prereg_amendment_1_erratum.md:1-13` |
| Open questions TD (pourquoi TD3 < 0.99, etc.) | `README.md:1266-1271` |
| Bilan global 5 points | `README.md:1273-1291` |
| TD2' non preregistré ; règle « mean ≥ δ_R » plafonne P(REPR) à 0.5 ; MDE80 0.062 (n=5) / 0.058 (n=10) ; budget 9–10 h / 18–20 h CPU | `README.md:1293-1310` (1302-1308) |
| SHA-256 des 6 npz | `README.md:1320-1328` |
| Environnement lock (torch 2.12.0+cpu, numpy 2.4.1) ; déterminisme postfix | `README.md:1331-1339` ; `outputs/logs/td_corrected/td_determinism_check_postfix.log` |
| Encodage UTF-16 des logs Tee-Object | `README.md:1341-1343` |
| Commandes canoniques PowerShell | `README.md:1345-1360` |
| Mapping résultat→artefact/commit | `README.md:1362-1376` |
| Note clé `r2_identity` absente des artefacts existants | `README.md:1378-1382` |
| SHA-256 des 300 artefacts TD | `outputs/logs/td_corrected/td_artifacts_sha256.txt` (300 lignes) |
| Prereg TD md : commit seul, question, bras, protocoles, hyper, anti-sélection, métriques (R² « affine » ligne 68), règle §6, C6', §8 (TD0 [0.78,0.90] l.115 ; TD3 ≥0.99 l.118-119) | `outputs/prereg/td_encoder_prereg.md:3,8-13,18-27,40-43,44-53,54-58,60-66,68,78-104,106-112,114-122` |
| Prereg TD json : decision_rule (D1 td2−td0, delta_R=0.05, unilateral_95, t_ddl4=2.1318, n=5) ; verdicts 3 états ; D2/D3 ; puissance | `outputs/prereg/td_encoder_prereg.json` `/decision_rule`, `/verdicts`, `/comparaisons_secondaires`, `/puissance` |
| Amendment 1 : §0 R² identité fixée (citations code `main_evaluate_td.py:88-89`, `ml_train_experiment_a_seq.py:593-595`) ; TD0 0.8747±0.0201 ; §1 bug+preuve 8 déc. | `outputs/prereg/td_encoder_prereg_amendment_1.md:10-40,37,106,42-80` |
| Erratum amendment (en-tête BROUILLON commité) | `outputs/prereg/td_encoder_prereg_amendment_1_erratum.md:1-13` |
| Note archivage runs td0_indist_nu001_* (hors verdict) | `outputs/prereg/td0_indist_nu001_archive_note.md:1-26` |
| Code : R² identité + label erroné ; nommage évaluateur ; val non exportée | `main_evaluate_td.py:88-89,107-110,177-183` ; `ml_train_experiment_a_seq.py:593-595,1182-1183` ; `main_evaluate_b1.py:86-87,94` |
| Bug filtre heldout_nu ; config ; module protocoles | `main_train_td.py:48` ; `config.py:417` ; `td_protocols.py` |
| Gates audit Step 3 (monotonie, max|u|≤5, dedt ≤1e-2, Péclet maille ≤2.0, CFL) | `README.md:310-335` ; `config.py:366-380` ; `outputs/logs/data_generation_report_nu_sweep_{train,val,test}.json`, `nu_sweep_pilot_report.json` |
