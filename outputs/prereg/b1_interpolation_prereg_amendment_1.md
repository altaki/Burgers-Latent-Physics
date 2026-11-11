# Amendement 1 au pré-enregistrement B1 (avant tout run de production)

Amende `b1_interpolation_prereg.md` / `b1_interpolation_prereg.json` (commit f431e66).
Points 1, 2, 4, 5, 6, 8 du retour de revue. Aucun entraînement effectué pour cet
amendement ; toutes les valeurs ci-dessous sont des mesures numpy sur les npz
`outputs/data/*_nu_sweep.npz` (lecture seule) ou des recalculs depuis des
`predictions_test.npz` commités.

## 1. ν-oracle : convention « strate held-out du TRAIN COMPLET » (bloquant)

- **Doc** : `README.md:342-344` documente que `nu_oracle` est calculé sur le FULL train
  (les 3 strates, y compris le ν held-out) — oracle qui connaît ν, critère conservateur.
- **Code au Step 6** : `baseline_z_vs_nuD` (`ml_train_experiment_a_seq.py:643-677`) était
  appelée avec le train FILTRÉ (`:969`) ; `baseline_predictions`
  (`ml_train_experiment_c.py:563-569`) fait un fallback moyenne globale quand la strate
  est absente (`:568`). **Vérification numérique** : la valeur publiée
  `baseline_nu_oracle_rel_rmse_nu001 = 1.4287638052743117` (metrics du run
  `window_w25_heldout_nu_zenergy_centered_s42`) correspond à la convention A (moyenne
  globale du train filtré {0.05,0.20} = 0.16934164, rel_rmse recalculé 1.4287638073,
  écart 2.0e-9 = float32), PAS à la convention B documentée (strate ν=0.01 du train
  complet = 0.16917955, rel_rmse 1.4287588935, écart 4.9e-6). **Écart doc/code constaté
  au Step 6**, sans impact : `beats_nu_oracle` compare rel_rmse(z) = 0.537 << 1.4288 dans
  les deux conventions → verdict inchangé.
- **Amendement** : à partir de maintenant, `nu_oracle` est calculé sur le TRAIN COMPLET
  (convention B, conforme à README:342), paramétré dans `verdict_a_seq` /
  `baseline_z_vs_nuD` (`nu_oracle_source="full_train"`). La moyenne globale du train filtré
  reste une baseline complémentaire renommée `global_mean`.
- **ν=0.05 (B1)** : oracle convention B = moyenne de νD sur la strate ν=0.05 du train
  complet, fenêtres t∈[24,99] = **0.16616840**, rel_rmse test ν=0.05 = **1.18029214**
  (convention A : 0.17084722, rel_rmse 1.17931282 — retenue comme `global_mean`).
  C4 utilise désormais **1.1803**.

## 2. Règle de décision : ajout de C8 (battre la baseline E_t avec IC excluant 0)

Règle f431e66 (citée mot pour mot dans le JSON) : C1 Spearman ≥ 0.8 ; C2 R² ≥ 0.5 ;
C3 pente ∈ [0.5, 2] ; C4 rel_rmse(z) < rel_rmse(ν-oracle) ; C5 rang ≥ 3 ; C6 rec RMSE ≤ 0.15 ;
C7 médian z/D ∈ [0.025, 0.10] ; F1 Spearman < 0.655 → NOT IDENTIFIABLE.
**Manque** : aucune condition ne force à battre la baseline E_t sans entraînement
(à ν=0.05 : Spearman 0.7658, R² 0.4769 — le seuil C2 R² ≥ 0.5 est à 0.023 près atteint par
ce proxy d'énergie). **Ajout C8** : à ν=0.05, pour le bras aphys_mono0,
ΔSpearman = Spearman(z,νD) − Spearman(E_t_affine,νD) et ΔR² = R²(z) − R²(E_t_affine),
avec IC95 bootstrap par trajectoire (B ≥ 2000, graine fixée) puis IC inter-graines
t ddl=2 ; **C8 vraie si les deux IC excluent 0** (borne inférieure > 0).
Baseline E_t : z_E = a·E_t + b, (a,b) = (0.3544, 0.0219) ajustés par moindres carrés sur
le train filtré {0.01,0.20}, fenêtres t∈[24,99] (recalculé dans l'évaluateur).
**causal_w25** (R² 0.9166, Spearman 0.9923 à ν=0.05) et le **plafond centré** (R² 1.000)
sont des **références rapportées, hors règle de décision** (estimateur lissé à 25 pas :
plafond réaliste, pas un critère).
Verdict : IDENTIFIABLE si C1–C8 toutes vraies ; NOT IDENTIFIABLE si F1 ; PARTIEL sinon.

## 4. Queue lourde : réconciliation revue pré-A-Physics vs mesure B1

Revue historique : kurtosis 23.9, top 1 % = 39 %, top 5 % = 68 %. Mesure B1 phase 0 :
top 1 % = 22.7 %, top 5 % = 50.6 % (train filtré {0.01,0.20}, fenêtres effectives
t∈[24,99], cible standardisée). **Réconciliation mesurée** : la revue mesurait sur
t∈[0,100] (toutes positions, y compris le transitoire initial t<24 exclu du training
A-seq) avec le kurtosis NON centré (m4 = kurt_excess + 3) : train filtré {0.01,0.2},
t∈[0,100] → m4 = 23.95 ≈ 23.9, top 1 % = 38.7 % ≈ 39 %, top 5 % = 64.6 % (train complet :
62.9 %). La queue lourde vient des premiers pas t<24, exclus des fenêtres
t∈[24,99]. Pas d'incohérence : définitions différentes (split × plage temporelle ×
kurtosis centré ou non). **Le seuil Huber « top 1 % > 30 % » a été fixé APRÈS la mesure
de 22.67 % sur la distribution effective d'entraînement** (train filtré, t∈[24,99]) ;
il ne se serait PAS déclenché sur le protocole heldout_nu validé (top 1 % = 21.7 %),
mais il AURAIT été déclenché sur la mesure historique t∈[0,100] (38.7 %) — d'où
l'écriture explicite de la définition (t∈[24,99], train filtré) dans X3.

## 5. Ratio z/D ∈ [0.025, 0.10] : justification + intercept

Bornes = moyennes géométriques avec les strates voisines : √(0.01·0.05) = 0.0224,
√(0.05·0.20) = 0.1000. Mesuré sur le run STANDARD aphys s42 (ν=0.05 VU à l'entraînement —
illustration uniquement, pas une prédiction held-out) : médian z/D = 0.04894,
pente de z ~ D = 0.04542, **intercept = 0.00894** (rapporté comme demandé).

## 6. Prédictions chiffrées par bras + falsification

- aphys_mono0 @ν=0.05 : Spearman ∈ [0.90, 0.99] (falsifié si < 0.655 → F1), R² ∈ [0.6, 0.95]
  (falsifié si < 0.5), pente ∈ [0.8, 1.1] (falsifié si hors [0.5, 2]), rel_rmse < 1.1803
  (C4), médian z/D ∈ [0.025, 0.10] ≈ 0.05 (falsifié si hors plage → z suivrait D),
  C8 : ΔSpearman et ΔR² vs E_t avec IC excluant 0. Verdict attendu IDENTIFIABLE ou
  PARTIAL (C6 rec RMSE possiblement > 0.15 comme au Step 6).
- none @ν=0.05 : Spearman < 0.655 → NOT IDENTIFIABLE attendu.
- **Note** : les R² C2 citées (ν=0.05 → 0.9087, minimum des strates) sont
  IN-DISTRIBUTION (run standard, ν=0.05 vu) ; il n'existe PAS de référence held-out à
  ν=0.05 avant B1.

## 8. A1 à ν=0.05 : portée

22/50 trajectoires test à ν=0.05 ont leur pic de νD avant 0.48 (début de la couverture
fenêtres) → les métriques A1 à ν=0.05 testent surtout la DÉCROISSANCE, pas le
raidissement ; n effectif = 17 trajectoires (pic intérieur + ≥ 5 pas) → A1 à ν=0.05
reste EXPLORATOIRE, non seuillé (X2).

## 7/9. Export + commandes

Chaque run de `main_train_b1.py` écrit automatiquement son `predictions_test.npz`
(même format que Step 6, nommage incluant protocole et graine, sans écraser les runs
existants) ; les SHA-256 des nouveaux npz seront consignés après les runs.
Commandes harmonisées en `python3`.
