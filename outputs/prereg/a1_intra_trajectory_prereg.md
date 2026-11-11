# Pré-enregistrement A1 — z est-il un traceur dynamique de νD intra-trajectoire ?

**Date** : session A1 (après commit 9325591, Step 6 A-Physics).
**Question** : à ν=0.01 held-out, z (unités physiques) suit-il la dynamique propre de
νD_vrai au sein de chaque trajectoire (montée/raidissement, pic, descente), ou n'est-il
qu'une étiquette statique bien calibrée inter-trajectoires (profil temporel commun ou
proxy de type E_t) ?
**Contrainte** : aucun entraînement. Lecture seule des `predictions_test.npz`, des npz
`nu_sweep_v1` et des checkpoints. Script additif unique : `main_analyze_a1_intra_trajectory.py`.
Aucune statistique impliquant z ou h n'a été produite avant le commit de ce fichier
(Phase 0 = vérité terrain E/D/νD + inventaire des clés + hashes uniquement).

## 1. Résultat de la Phase 0 (vérifications bloquantes)

### 1.0a — Anomalie `dissipation_physical` : RÉSOLUE, pas d'incohérence
- `pde_observables.py:564` : `nu_dissipation = nu_values[:, None] * dissipation` → la clé
  **`dissipation_physical` du npz = ν·D déjà** (et non D).
- `main_export_predictions.py:137,153` : `nuD = raw["dissipation_physical"]`,
  `"nu_dissipation_t": nuD[ti, tt]` → **`nu_dissipation_t` provient de la clé
  `dissipation_physical`** ; idem pipeline A-seq : `ml_datasets_experiment_c.py:222`
  (`"nu_dissipation": "dissipation_physical"`) et `:540-541`.
- L'écart signalé en Phase 0 (corr(dE/dt, −ν·dissipation_physical) = 0.777) était une
  **erreur de ma comparaison** (ν appliqué deux fois → −ν²D). Mesures sur
  `test_data_nu_sweep.npz` : max|dissipation_physical − ν·dissipation| = **0.0** ;
  corr(dE/dt, −dissipation_physical) = **0.999993** = corr(dE/dt, −ν·dissipation).
- Vérification dans chaque `predictions_test.npz` (jointure `trajectory_index` /
  `time_index_t`) : max rel err |nu_dissipation_t − ν·dissipation_t| = **8.1e-08**
  (précision float32), identique par strate ν ∈ {0.01, 0.05, 0.20} ;
  max|nuD_est_t − (E_{t−1}−E_{t+1})/2dt| = **2.9e-06** (float32).
  → `nu_dissipation_t == ν·D` à la précision machine. **Aucune incohérence : on continue.**

### 1.0b — Convention de ΔE_w
`main_analytic_nuD_baseline.py:6,59-62` : estimateur **causal** intra-fenêtre
`nuD_est_causal_w(t) = (E_{t−w+1} − E_t)/((w−1)·dt)` pour t ∈ [w−1, n_times) — couvre le
**passé** [t−w+1, t] de la fenêtre d'entrée [t−w+1, t] (`ml_datasets_experiment_c.py:573-578`,
`__getitem__` :464). La formule E[t+w]−E[t] (avenir) utilisée dans une note de Phase 0 est
**rejetée** ; A1 n'utilise que le causal (passé) et le centré `(E_{t−1}−E_{t+1})/2dt`
(`ml_datasets_experiment_c.py:612-614`, `main_export_predictions.py:38-46`).

### 1.0c — SHA-256 des entrées (intégrité)
Data (lecture seule, non commités — 718 MB) :
- `outputs/data/test_data_nu_sweep.npz`  e501f91f5bb73258eb63957bb7f54a091b1a4dd0a0db53a8469283cd881060bc (119.75 MB)
- `outputs/data/train_data_nu_sweep.npz` dab467cfbb3daccba6b355b353adb07225dbc56004aeee0018483a2ff494768f (478.97 MB)
- `outputs/data/val_data_nu_sweep.npz`   07b67e8b6ca9d84d7efbe792eaa03bba8ae1204fcab40c8eb5a24d417e2edf51 (119.73 MB)

Predictions (16 × ~1.8 MB ≈ 29 MB, **proposition : commiter ces 16 fichiers dans un commit
séparé après validation** — le prereg présent est commité SEUL) :
- window_w25_heldout_nu/predictions_test.npz                    3aa968613c58805f4a486c5cdc15b1c127ff131d0c2f557afc14d1b4b2328ade
- window_w25_heldout_nu_s123/predictions_test.npz               feb1038e37362dac05e8b7fc6f5e7960a4841b5c38939f7b180fb541015d428d
- window_w25_heldout_nu_s7/predictions_test.npz                 d010310aef9fb78fd433351df2f3c57db216b3e28ff0128b9a96263da72dd8f6
- window_w25_heldout_nu_zenergy_centered_s123/predictions_test.npz ca0626bc2f4b2e36e27b8a090e50e4548fe18e2c75aeffcc7f5de026abe7b98a
- window_w25_heldout_nu_zenergy_centered_s42/predictions_test.npz  1edb6109a793130368fe9a76e5a8cd987e0173982a18464cfb9e3b7ca1792056
- window_w25_heldout_nu_zenergy_centered_s7/predictions_test.npz   f775c9c0c3362fa7e9b8c6d13d8732cdbc1edbb974bcb2fcd37830d8f73368a5
- window_w25_heldout_nu_zsupervised_s42/predictions_test.npz      4d0f68a89b86636914aa1fabf6b877bed44d1983a1dbed43d779a7fc727d58ec
- window_w25_standard/predictions_test.npz                      484c66379bc1e761f4dccd298aaa0f0f73b456e6d3cd670cfb1b540f5065edd3
- window_w25_standard_s123/predictions_test.npz                 52674700d98ca81d2252960ef514e733ad911be0cb8721316723e399aa1ab9dc
- window_w25_standard_s7/predictions_test.npz                   53e8058deb3aeebf608b840b2a012eeb4b5a45b2c6dd2bbaf28cf61399c02ba7
- window_w25_standard_zenergy_centered_s123/predictions_test.npz d334c7b7620617bc2439b93562be3deec6f24f4a048184e4f3b2e11ddafb4d72
- window_w25_standard_zenergy_centered_s42/predictions_test.npz  c436f1c7deaa1e27969903d63525bca1c8c3342741859db47154dddeaf85b9c0
- window_w25_standard_zenergy_centered_s7/predictions_test.npz   e7b5a9ffe0918a3a87234e1a906d2d9d59f6a128fa05107e38fee0141db7baa4
- window_w25_standard_zsupervised_s42/predictions_test.npz      3b31220a63b6ebb423fbff5e52cd656d3c0622a55d987e2211556978bfb674dd
- window_w50_heldout_nu/predictions_test.npz                    941554804a243afa5b261f5b4f70f09f74c48e60f442186d7c9823b313f5ca92
- window_w50_standard/predictions_test.npz                      006eda471cff6f6d9d1915d38621670e8a9f1fc47d66cba9ff2a12cdc6cf8f22

### 1.1 — Inventaire des predictions_test.npz (16 runs A-seq)
Tous : t_index ∈ [24, 99] à w=25 (76 fenêtres/traj, 150 traj, 11 400 lignes, dont 3 800 à
ν=0.01 = 50 traj × 76) ; w=50 : t ∈ [49, 99] (51 fenêtres). Clés présentes : z (scalaire,
unités physiques), h, u, nu, time_t, trajectory_index, time_index_t, time_index_tp,
energy_t, dissipation_t, nu_dissipation_t, nuD_est_t, energy0, ic_type_id, split,
protocol, seed, window_size, z_target_mode, run_name (+ source/checkpoint_path sur les
runs sans suffixe de seed). Aucun champ manquant → pas de jointure nécessaire au-delà de
la vérification 1.0a.

### 1.2 — Couverture temporelle (vérité terrain, test, ν=0.01, 50 traj)
- 96 % des trajectoires ont un premier pas croissant (Δ_1 νD > 0) **avant t = 0.48**
  (avant la première fenêtre A1) ; 58 % ont leur pic νD dans [0.48, 1.98] (fenêtres A1) ;
  t_pic médian = **1.59 s**, min 0.0, max 2.0.
- Conséquence : la montée est souvent déjà entamée à l'entrée de la fenêtre ; 42 % des
  trajectoires ont leur pic hors de [0.48, 1.98]. L'analyse « en raidissement » ne porte
  que sur les trajectoires dont le pic est dans la fenêtre ; le nombre de pas en
  raidissement par trajectoire est rapporté (traj avec < 5 pas exclues de M1_raid, n rapporté).
- Pas croissants (k=1) sur les fenêtres : 66.2 % ; fenêtres avant t_pic : 64.4 %.
- Pics locaux : 29 traj à 1 pic, 21 traj à 0 pic local, **0 traj multi-pics** → t_pic propre.

### 1.3 — Définitions retenues
- **Raidissement (principale)** : t < t_pic(νD_vrai), t_pic = argmax global sur [0, 100].
  **Alternative (exploratoire)** : pas où Δ_1 νD_vrai > 0. La déf. (a) couvre 66 % des pas
  (peu discriminante, bruitée) ; la déf. (b) est robuste (0 multi-pics). Décroissance : t ≥ t_pic.
- **Horizons** k ∈ {1, 5, 10} pas (0.02 / 0.10 / 0.20 s). **Exclusion** |Δ_k νD| < 5 % de
  l'étendue intra-traj de νD. Fractions exclues mesurées @ν=0.01 : k=1 **98.2 %**,
  k=5 47.0 %, k=10 27.5 %. → **k=1 rétrogradé EXPLORATOIRE avec seuil 2 %** (garde 21 % des
  pas ; médiane |Δ_1 νD|/étendue = 1.1 %, p95 = 3.6 %) ; **k=5 et k=10 CONFIRMATOIRES** au
  seuil 5 %.
- **Lags** M5 ∈ [−10, +10] pas.

## 2. Runs et comparateurs

**CONFIRMATOIRE** — protocole heldout_nu, strate ν=0.01 (test) :
- aphys_mono0 × {s42, s7, s123} = `window_w25_heldout_nu_zenergy_centered_s{42,7,123}`
  (z_target_mode=energy_centered, monotonicity_weight=0.0) ;
- none × {s42, s7, s123} = `window_w25_heldout_nu` + `_s7` + `_s123` (z_target_mode=none) ;
- comparateurs, évalués avec EXACTEMENT les mêmes métriques :
  - PLAFOND centré : `nuD_est_t` = (E_{t−1}−E_{t+1})/2dt (déjà dans les npz) ;
  - causal_w25 : (E_{t−24}−E_t)/(24·dt) (vérité terrain) ;
  - HORLOGE : profil moyen LOO des autres trajectoires test à ν=0.01 au même t
    (référence, pas un modèle — accepté, avec l'artefact M2 = −1 documenté §4) ;
  - STATIQUE : z constant par trajectoire ;
  - BASELINE E_t (sans entraînement) : z_E = a·E_t + b, OLS sur le **train** heldout_nu
    (ν ∈ {0.05, 0.20}, t ∈ [24, 99]) → **a = 0.40016, b = 0.01923** (figés) ; variante
    affine(E_t, E0) en exploratoire ;
  - none_recal : z de window-A none recalibré affine (OLS z ~ νD_vrai sur train heldout_nu),
    appliqué au test.

**EXPLORATOIRE** (étiqueté) : asup = `window_w25_heldout_nu_zsupervised_s42` ;
aphys λ_mono=0.1 s42 (même npz que mono0 s42, tag différent) ; protocole standard à ν=0.01
et ν=0.05 (`window_w25_standard*`, `..._zenergy_centered_*`) ; définition alternative du
raidissement (Δ_1 νD > 0) ; w=50 (`window_w50_*`, t ∈ [49, 99]).

## 3. Métriques M1–M8 (défs figées) et agrégation
Fenêtres A1 : t_index ∈ [24, 99], z attribué à t (fin de fenêtre). Strate ν=0.01 held-out.
- **M1** : par trajectoire, corr(z − z̄_traj, νD − νD̄_traj), Pearson et Spearman, sur toute
  la trajectoire ET en raidissement seul (traj avec ≥ 5 pas en raidissement).
- **M2** : double centrage — retirer la moyenne par trajectoire, PUIS le profil horloge
  LOO au même t — corrélation des résidus poolés (Pearson + Spearman).
- **M3** : accord de signe Δ_k z / Δ_k νD, k ∈ {1, 5, 10}, en raidissement et en
  décroissance, avec exclusion §1.3 ; fraction exclue rapportée.
- **M4** : t_argmax(z) − t_argmax(νD_vrai), par trajectoire, en secondes.
- **M5** : lag de corrélation croisée intra-traj (z, νD centrés par traj), lag ∈ [−10, +10].
- **M6** : amplitude relative (max−min z)/(max−min νD_vrai) sur les pas en raidissement.
- **M7** : corrélation partielle de rang corr(z, νD | E_t) — résidus OLS sur rang(E_t) —
  poolée dans la strate ν=0.01, et à t fixé (t_index 25, 50, 75 = 0.5, 1.0, 1.5 s).
  Note Phase 0 : corr(E_t, νD_t) à t fixé = 0.49–0.91 @ν=0.01 → E_t et νD ne sont PAS
  indépendants à t fixé ; M7 reste définie (partialisation) mais l'interprétation
  « indépendance » du prompt est corrigée.
- **M8** : corr intra-trajectoire centrée de z avec E_t, en raidissement (contrôle raccourci).

**Agrégation** : médiane + distribution sur les 50 trajectoires ; IC95 bootstrap par
trajectoire (B ≥ 2000) au sein de chaque run ; moyenne ± IC inter-graines (Student, ddl=2)
pour aphys_mono0 et none (3 graines) ; comparaison appariée par graine aphys_mono0 vs none.

## 4. Seuils de décision FIGÉS (calibrés sur la vérité terrain, Phase 0)

Calibration mesurée @ν=0.01 (test, déf. raidissement = t < t_pic) :

| comparateur | M1_raid Pearson | M2 poolé | M3 k=5 raid | M4 médian | M6 | M8 raid |
|---|---|---|---|---|---|---|
| plafond centré | 1.000 | 1.000 | 1.00 | 0.00 s | 1.00 | −0.98 |
| causal_w25 | 0.972 | 0.674 | 1.00 | **0.24 s** | 0.76 | −1.00 |
| horloge LOO | −0.23 | **−1.0 (artefact)** | 0.28 | −0.86 s | 0.68 | 0.24 |
| baseline E_t | −0.98 | 0.34 | 0.00 | −1.19 s | 0.33 | 1.00 |

Corrections apportées aux seuils du prompt, justifiées par cette calibration :
1. **M4 : 0.2 s → 0.3 s** — le biais de lissage de causal_w25 (vérité terrain) est déjà
   0.24 s ; exiger ≤ 0.2 s exclurait un estimateur lissé réaliste. 0.3 s reste sous
   l'horloge (−0.86 s).
2. **« M2 > M2(horloge) » supprimé** — M2(horloge) = −1.0 par construction (artefact du
   double centrage LOO), le critère était vide. Remplacé par **M2 > 0 avec IC95 bootstrap
   excluant 0**.
3. **M3 k=1 exploratoire** (98 % exclus au seuil 5 % étendue) ; k=5, k=10 confirmatoires.
4. **M5 non seuillé** — pour causal_w25 et E_t le lag max tombe au bord de la plage (±10),
   M5 est rapporté sans seuil.

**TRACEUR DYNAMIQUE** si, pour aphys_mono0 (médiane sur les trajectoires, puis moyenne sur
les 3 graines), à ν=0.01 held-out :
M1_raid Pearson ≥ 0.7 **ET** M3 k=5 raid ≥ 0.8 **ET** M2 poolé ≥ 0.5 avec IC95 excluant 0
**ET** M1_raid > M1_raid(baseline E_t) avec IC de la différence excluant 0 **ET**
|M4| médian ≤ 0.3 s **ET** M6 ≥ 0.5.

**ÉTIQUETTE STATIQUE** si M1_raid < 0.3 **OU** M2 ≤ 0.2 **OU** M8_raid > M1_raid.

**PARTIEL** sinon (préciser : rang OK mais amplitude M6 faible, retard M4/M5, etc.).

## 5. Prédictions chiffrées (à figer AVANT le run) et falsification

| run / comparateur | M1_raid | M2 | M3 k=5 raid | M4 | M6 | verdict attendu |
|---|---|---|---|---|---|---|
| aphys_mono0 (3 graines) | 0.6–0.9 | 0.4–0.8 | ≥ 0.8 | −0.1…+0.3 s | 0.5–1.0 | TRACEUR ou PARTIEL |
| none (3 graines) | 0.2–0.5 | 0.0–0.3 | 0.3–0.6 | — | — | ÉTIQUETTE STATIQUE ou PARTIEL |
| horloge LOO | −0.35 (mesuré) | −1.0 (artefact) | 0.28 (mesuré) | −0.86 s | 0.68 | référence |
| baseline E_t | −0.98 (mesuré) | 0.34 (mesuré) | 0.00 (mesuré) | −1.19 s | 0.33 | ÉTIQUETTE (M8=1) |
| plafond centré | 1.0 (mesuré) | 1.0 (mesuré) | 1.0 (mesuré) | 0.0 s | 1.0 | plafond |

**Falsification** : si aphys_mono0 a M1_raid < 0.3 ou M2 ≤ 0.2 ou M8_raid > M1_raid →
l'hypothèse « traceur dynamique » est rejetée (étiquette statique). Si M1_raid ≥ 0.7 mais
M6 < 0.5 ou |M4| > 0.3 s → PARTIEL (rang temporel OK, amplitude/retard dégradés). Si none
atteint M1_raid ≥ 0.7, la conclusion A-Physics (préservation injectée) s'étend au non-supervisé.

## 6. Sorties prévues (Phase 2)
- `outputs/logs/a1_intra_trajectory.json` : agrégats, IC, verdict, hash de ce prereg,
  SHA-256 des entrées ;
- `outputs/logs/a1_intra_trajectory_per_traj.csv` : une ligne par run × trajectoire ;
- figures `outputs/figures/a1_*.png` : z(t), νD_vrai(t), E_t(t) normalisé superposés pour
  8 trajectoires tirées au hasard (rng seed = **42**) + 2 meilleures / 2 pires selon M1 ;
  histogrammes de M1, M2, M4. (matplotlib requis ; sinon le script exporte les séries et
  signale l'absence de figures.)

## 7. Biais d'échantillonnage documentés (Phase 0.8)
- 96 % des traj à ν=0.01 commencent leur raidissement avant la première fenêtre (0.48 s) ;
  42 % ont leur pic hors [0.48, 1.98] → les médianes « en raidissement » portent sur le
  sous-ensemble de traj avec ≥ 5 pas avant t_pic dans la fenêtre (n rapporté par traj) ;
  les trajectoires sans phase de montée dans la fenêtre ne faussent pas M1_raid (exclues),
  mais réduisent l'effectif — documenté, pas corrigé.
- Le chiffre « 22 % de fenêtres en raidissement » du prompt n'est pas retrouvé (mesures :
  64.4 % de fenêtres avant t_pic, 66.2 % de pas croissants k=1) — écart documenté.
- Variance inter-trajectoire de νD @ν=0.01 = 57 % (pas 95 % ; le 95 % du contexte concernait
  les métriques poolées de z incluant le profil temporel).
