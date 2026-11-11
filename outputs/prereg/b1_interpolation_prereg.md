# Pré-enregistrement B1 — Transfert de z en INTERPOLATION (ν = 0.05 held-out)

**Objectif** : le transfert de z à un ν non vu tient-il en interpolation (train/val
ν ∈ {0.01, 0.20}, test complet, évaluation à ν = 0.05), et pas seulement en extrapolation
vers ν = 0.01 (Step 6) ?

**Protocole** : `heldout_nu005` — train/val restreints à ν ∈ {0.01, 0.20}
(`ml_train_experiment_a_seq.py:180-181,205-206` : le filtre s'applique à train ET val ;
test complet). w=25, stride=1, fenêtres t ∈ [24, 99]. Standardiseurs fittés sur ce train.

**Bras** (6 runs, aucun ne réécrit les runs existants) :
- `aphys_mono0` : z_target_mode=energy_centered, λ_z (z_weight) = 1.0, λ_mono = 0.0,
  graines {42, 7, 123} → `window_w25_heldout_nu005_zenergy_centered_s{seed}` ;
- `none` : z_target_mode=none, graines {42, 7, 123} → `window_w25_heldout_nu005_s{seed}`.

**Hyperparamètres** : identiques au Step 6 (defaults de `train_and_evaluate_a_seq` :
batch 256, lr 1e-3, 200 époques, patience 10, latent_dim 8, conv (16,32,32,64,64) k=5 s=2,
trunk (256,), decoder (64,128), predictor (64,64), rec/pred weight 1.0, variance 0.01,
standardize_z_target=True fit sur le train du protocole).

## Phase 0 — mesures (vérité terrain, avant commit)

- **0.1 Standardisation** (train ν∈{0.01,0.20}, 30 400 fenêtres, cible nuD_est) :
  mean 0.1705, std 0.2060, kurtosis excès 6.61, max|z| 7.98, **top 1 % = 22.7 % de Σz²,
  top 5 % = 50.6 %**. Comparaison heldout_nu historique (ν∈{0.05,0.20}) : top 1 % = 21.7 %,
  top 5 % = 50.3 %, max|z| 9.09. → La MSE est dominée par ~5 % de fenêtres, **sans excès
  par rapport au protocole Step 6 déjà validé**. Verdict NUANCÉ : pas de variante Huber/log
  par défaut ; **règle de déclenchement pré-enregistrée** : si top 1 % > 30 % de Σz² sur le
  train heldout_nu005 → run complémentaire Huber (λ_z sur Huber) en EXPLORATOIRE.
  Mesuré 22.7 % → non déclenchée.
- **0.2 Validation** : ACCEPTÉ — code `ml_train_experiment_a_seq.py:180-181,205-206` :
  `nu_filter_train` restreint train ET val ; early stopping / sélection sur val
  ν ∈ {0.01, 0.20} uniquement (ν=0.05 absent de val).
- **0.3 Plage** : ACCEPTÉ — plage νD train heldout_nu005 = [0.00277, 1.8337] ⊇ test ν=0.05
  [0.0132, 1.3869] → **0 % de fenêtres hors plage**.
- **0.4 Baselines à ν=0.05 (test, vérité terrain)** : plafond centré Spearman 1.000 /
  R² 1.000 / pente 1.000 / rel_rmse 0.0005 ; causal_w25 0.992 / 0.917 / 1.028 / 0.340 ;
  ν-oracle (moyenne globale train filtré = 0.1708, ν=0.05 absent du train) R² −0.0007 /
  rel_rmse **1.179** ; baseline E_t affine (a=0.354, b=0.0219, fit train filtré) 0.766 /
  0.477 / 0.436 / 0.853.
- **0.5 A1 à ν=0.05** : 50 traj test ; pic intérieur [24,99] : 19 ; pic intérieur ET ≥ 5 pas
  avant le pic : **17** ; pic à t=2.0 : 9 ; pic avant 0.48 : 22 ; pas croissants 39.8 %.
  → métriques A1 à ν=0.05 **EXPLORATOIRES** (n=17), non seuillées.
- **0.6 Séparation νD/D** : à ν fixé, Spearman(z,νD) = Spearman(z,D) exactement (vérifié à
  ν=0.05 sur run standard aphys s42 : 0.99380 = 0.99380). Critères de séparation en unités
  physiques : **ratio médian z/D à ν=0.05 ∈ [0.025, 0.10]** (attendu ≈ 0.05 ; mesuré 0.0489
  sur le run standard) et pente physique ∈ [0.5, 2].
- **0.7 Références** : C2 standard (in-distribution) R² νD par strate : ν=0.01 → 0.9472,
  **ν=0.05 → 0.9087 (minimum)**, ν=0.20 → 0.9910 (`metrics_c3_window_standard.json`).
  Prédiction : interpolation au moins aussi bonne en rang que l'extrapolation ν=0.01
  (Spearman aphys heldout_nu = 0.965), mais R² possiblement la plus faible des strates.
  Comparaison aux runs heldout_nu des mêmes graines en référence.
- **0.8 Critique du prompt** : 0.1 la queue ν=0.01 n'est pas pire que le protocole validé ;
  0.5 l'effectif A1 à ν=0.05 (17 traj) rend A1 exploratoire ; 0.3 confirmé 0 %.

## Règle de décision (v2, seuils du prereg A-Physics §7 inchangés, appliqués à ν=0.05)

Machine-lisible dans `b1_interpolation_prereg.json`. IDENTIFIABLE si les 7 conditions
C1–C7 sont toutes vraies (bras aphys_mono0, moyenne des 3 graines, IC inter-graines
t ddl=2 rapportés) ; NOT IDENTIFIABLE si Spearman < 0.655 ; PARTIEL sinon.
C1 Spearman(z,νD)@ν=0.05 ≥ 0.8 ; C2 R² affine ≥ 0.5 ; C3 pente ∈ [0.5, 2] ;
C4 rel_rmse(z) < rel_rmse(ν-oracle)@ν=0.05 (oracle = 1.179, Phase 0) ; C5 rang effectif ≥ 3 ;
C6 rec RMSE ≤ 0.15 ; C7 médian z/D ∈ [0.025, 0.10].
**Complémentaire (hors verdict)** : appariement aphys vs none par graine (IC t ddl=2) ;
métriques A1 à ν=0.05 (exploratoire, n=17) ; variante Huber si déclenchée.

## Prédictions chiffrées et falsification

- aphys_mono0 @ν=0.05 : Spearman ∈ [0.90, 0.99], R² ∈ [0.6, 0.95], pente ∈ [0.8, 1.1],
  rel_rmse < 1.179, médian z/D ≈ 0.05 → IDENTIFIABLE attendu (C6 rec RMSE possiblement
  > 0.15 → PARTIAL comme au Step 6).
- none @ν=0.05 : Spearman < 0.655 → NOT IDENTIFIABLE attendu.
- Falsification : si aphys Spearman < 0.655 → le transfert en interpolation échoue ; si
  R² < 0.5 ou rec RMSE > 0.15 → PARTIAL ; si médian z/D hors [0.025, 0.10] → la séparation
  νD/D échoue (z suivrait D, pas νD).
