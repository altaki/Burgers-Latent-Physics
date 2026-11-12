# Pré-enregistrement — Expérience « TD » : biais inductif de l'encodeur et écart au plafond analytique

**Commit** : ce fichier (+ `td_encoder_prereg.json`) est commité SEUL, avant tout code
d'expérience (Phase 1). Aucun run de production n'a été lancé avant ce commit.

## 0. Question

Pourquoi les encodeurs appris restent-ils sous l'estimateur à 2 snapshots
(backward_w25 : R² 0.9995 à ν=0.01 held-out, `outputs/logs/experiment_a_seq_analytic_baseline.json`)
alors que aphys_mono0 atteint R² 0.844 (Step 6, 3 graines) ? L'écart est-il
**REPRÉSENTATIONNEL** — l'encodeur Conv1D+ReLU+flatten représente mal la fonctionnelle
quadratique dE/dt ≈ ∫ u_t·(u_t − u_{t−1})/Δt dx, invariante par translation — ou dû à autre
chose (goulot latent scalaire, pertes auxiliaires, optimisation) ?

## 1. Bras (tous : L_z = energy_centered, λ_z=1, λ_mono=0, w=25, hyperparamètres identiques
à aphys_mono0 sauf l'encodeur)

| bras | encodeur | canaux d'entrée (in_channels) | pooling |
|---|---|---|---|
| TD0 | Conv1DEncoder actuel (référence) | w = 25 (fenêtre brute) | flatten |
| TD1 | TD0 + canaux de différences temporelles explicites | w + 5 = 30 : fenêtre brute + Δ_k u = (u_t − u_{t−k})/(kΔt), k ∈ {1,2,4,8,16} | flatten |
| TD2 | TD1 + invariance par translation | 30 (idem TD1) | **global average pooling** spatial (moyenne sur la longueur de sortie conv) à la place du flatten |
| TD3 | TD2 + canal multiplicatif u_t ⊙ Δ_1 u (DIAGNOSTIC, injection architecturale) | 31 | GAP |

- Δt = Δt_saved = 0.02. u_t = dernier canal temporel de la fenêtre (dim 0 = temps,
  `ml_datasets_experiment_c.py:585-600`). La transformée de canaux est faite **dans
  l'encodeur** (pas dans le dataset) : les npz restent en lecture seule.
- TD3 rend ∫u·∂_t u accessible linéairement après pooling moyen :
  mean_x(u_t ⊙ Δ_1 u) ≈ (1/Nx)·dE/dt = −νD/Nx (identité dE/dt = −νD, zero_mean).
  C'est une **INJECTION ARCHITECTURALE** de la structure de l'identité, étiquetée comme
  telle : TD3 ne teste pas une découverte, il calibre ce que le biais inductif peut
  apporter au maximum.
- Chaque bras TD0–TD3 est lancé en deux modes appariés : `zenergy_centered` (L_z active)
  et `none` (sans L_z, requis par C6'a).

## 2. Graines, protocoles, nommage

- **Graines (n=5, fixées a priori)** : {42, 7, 123, 2024, 999} — cohérent avec C6'
  (29c7c3e : {42, 7, 123, + 2 nouvelles}).
- **Protocole principal** : `heldout_nu` (train/val ν ∈ {0.05, 0.20}, test complet,
  évaluation à ν=0.01) — même protocole que le Step 6 où l'écart 0.844 vs 0.9995 a été mesuré.
- **Protocole secondaire** : `heldout_nu005` (train/val ν ∈ {0.01, 0.20}, évaluation à
  ν=0.05) — réplication indépendante (B1).
- Nommage des runs (jamais d'écrasement, abort si le dossier existe) :
  `td{bras}_{protocole}_zenergy_centered_s{seed}` et `td{bras}_{protocole}_s{seed}` (none),
  ex. `td2_heldout_nu_zenergy_centered_s42`. Chaque run écrit `predictions_test.npz`
  (format Step 6 existant) + `metrics_*.json` + `standardizers.json`.

## 3. Hyperparamètres figés (identiques aphys_mono0 / B1, prereg f431e66+fc1075b)

batch_size=256, lr=1e-3 (Adam), num_epochs=200, patience=10, weight_decay=0,
latent_dim=8, conv_channels=(16,32,32,64,64), kernel_size=5, conv_stride=2,
trunk_dims=(256,) (TD2/TD3 : entrée trunk = 64 = dernier canal conv, GAP),
decoder_hidden_dims=(64,128), predictor_hidden_dims=(64,64), scalar_head_hidden_dims=(),
activation=relu, dropout=0, λ_rec=λ_pred=λ_z=1.0, λ_var=0.01 (min_var=0.01), λ_mono=0,
standardize_z_target=True (stats fit sur le split TRAIN du protocole courant),
z_target_mode ∈ {energy_centered, none}, stride=1, t ∈ [w−1, n_times−stride−1] = [24, 99],
standardiseur d'entrée = GlobalStandardizer fit sur train complet raw (inchangé).

**Anti-sélection sur le test** : hyperparamètres figés dans ce prereg avant tout run ;
un seul jeu de runs par bras/mode/graine/protocole ; aucune itération sur le test ν=0.01 ;
la règle de décision (§6) ne porte que sur les comparaisons pré-enregistrées ;
toute analyse postérieure est étiquetée EXPLORATOIRE et hors verdict.

## 4. Métriques

**Primaires** (protocole principal, strate ν=0.01 held-out du test, z en unités physiques,
3 800 fenêtres / 50 trajectoires) : Spearman(z, νD), R² affine νD~z, pente de calibration,
rel_rmse ; **Δ apparié par graine** ΔR²_s = R²(TD_bras, s) − R²(TD0, s) ; écart au plafond
R²_bras − 0.9995 (backward_w25 ν=0.01). **M6** amplitude en raidissement (définition prereg
A1 `5623042` : raidissement = t < t_pic, t_pic = argmax global de νD sur [0,100],
M6 = (max−min z)/(max−min νD_vrai) sur les pas en raidissement).

**Secondaires** : C6'a / C6'b (repris de 29c7c3e, §7), rang effectif 95 % de h
(`latent_pca_stats`, ml_models.py:574-599), participation ratio, A1 complet (M1–M8) à ν=0.01,
protocole secondaire heldout_nu005 (métriques à ν=0.05).

## 5. Puissance (calculs exacts, noncentral t ; n=5, ddl=4, t(0.95,4)=2.1318)

- sd inter-graines connu (Step 6 aphys_mono0, R² 0.859/0.791/0.881) : **sd = 0.047** ;
  Spearman : sd = 0.0036.
- Comparaison principale ΔR² (TD2−TD0 apparié), n=5, seuil d'effet δ_R = 0.05 :
  puissance = 0.92 si sd(Δ)=0.03 et vrai Δ=0.05 ; 0.58 si sd(Δ)=0.05 ; 0.98 si vrai Δ=0.10
  et sd(Δ)=0.05. Probabilité de conclure NON REPRÉSENTATIONNEL si vrai Δ=0 :
  0.91 (sd 0.03) / 0.54 (sd 0.05).
- C6'a (hérité 29c7c3e, δ=0.10, sd ratio=0.085) : puissance exacte n=5 = **0.70**
  (le prereg 29c7c3e annonce 68 % par approximation centrale décalée ; n=3 : 0.39 exact
  vs 23 % approximé — n=5 retenu).

## 6. Règle de décision machine-lisible (comparaison principale, CONFIRMATOIRE)

Sur d_s = R²(TD2, s) − R²(TD0, s), s ∈ 5 graines, ν=0.01 held-out ; IC unilatéral 95 %,
t ddl=4 (t=2.1318) ; δ_R = 0.05 (≈ 1 sd inter-graines de R² au Step 6 = 0.047 ; un effet
inférieur à la dispersion inter-graines n'est pas actionnable).

- **REPRÉSENTATIONNEL** si mean(d) ≥ δ_R **ET** borne inférieure unilatérale 95 % de mean(d) > 0.
- **NON REPRÉSENTATIONNEL** si borne supérieure unilatérale 95 % de mean(d) < δ_R
  (on exclut un gain ≥ 0.05 : le pooling + différences temporelles n'expliquent pas l'écart).
- **INDÉTERMINÉ** sinon.

Comparaisons secondaires (même forme, appariées par graine, n=5) :
- TD1−TD0 (canaux de différences seuls, flatten) : CONFIRMATOIRE, même règle avec δ_R=0.05.
- TD3−TD2 (injection du canal produit) : **EXPLORATOIRE** (diagnostic d'injection, pas découverte).
- M6, rang effectif, A1 complet, protocole heldout_nu005 : EXPLORATOIRE (non seuillés).

## 7. C6' repris de 29c7c3e (CONFIRMATOIRE, par bras z vs none apparié, n=5)

- **C6'a** : ratios r_s = rec(bras_z, s)/rec(bras_none, s) ; vrai si borne supérieure
  unilatérale 95 % mean(r) + t(0.95,4)·sd(r)/√5 < 1+δ, **δ=0.10**.
- **C6'b** : vrai si rec(bras_z) ≤ plancher PCA k=8 = **0.188** (k fixe = latent_dim).
- Sensibilité B1 hors verdict (rappel) : C6'a 1.261 > 1.10 échoue, C6'b 0.196 > 0.188 échoue.

## 8. Prédictions chiffrées par bras (ν=0.01 held-out, protocole principal) et falsification

- TD0 (référence re-entraînée sur code déterministe) : R² ∈ [0.78, 0.90] (Step 6 : 0.844,
  sd 0.047), Spearman ∈ [0.95, 0.98]. Falsification du prereg : TD0 hors de ces plages →
  le code déterministe change le résultat de référence → STOP, investiguer avant toute
  interprétation.
- Si REPRÉSENTATIONNEL : R²(TD2) ∈ [0.90, 0.9995]. Si NON REPRÉSENTATIONNEL :
  R²(TD2) ∈ [0.78, 0.90]. Falsification croisée : R²(TD2) < R²(TD0) − 0.05 → les canaux
  de différences + pooling dégradent (effet négatif, à rapporter tel quel).
- TD3 (injection) : si le canal produit fonctionne comme prévu, R²(TD3) ≥ 0.99 attendu ;
  c'est le plafond de ce que le biais inductif peut apporter, étiqueté INJECTION.
- M6(TD0) attendu ≈ 0.81 (Step 6) ; toute amélioration de M6 avec TD2 est rapportée
  (EXPLORATOIRE).

## 9. Exigences Phase 0 intégrées (bloquantes avant toute run de production)

- **Déterminisme (0.2)** : `torch.use_deterministic_algorithms(True)`,
  `torch.set_num_threads` fixé (4), générateur `torch.Generator` explicite et seedé pour
  chaque DataLoader shufflé. Test automatique `main_determinism_check.py` : deux runs
  identiques (même graine, 2 époques, données tronquées, in-process) doivent diverger de
  **≤ 1e-6** sur toutes les métriques et prédictions (sinon exit 1). **Ce test doit
  passer sur l'environnement Windows avant toute run de production TD.** Tant qu'il
  n'existe pas/passe, aucune comparaison appariée par graine n'est interprétable au-delà
  du smoke. L'écart s42 historique (0.1775 vs 0.1663) reste en « Limites » (cause probable :
  working tree local non commité — le working tree EST actuellement modifié, voir Phase 0).
- **Traçabilité (0.3)** : chaque run écrit dans ses metrics `traceability` = {git_hash,
  working_tree_dirty, torch/numpy/python versions, start_time, end_time}. Si le working
  tree est modifié (fichiers suivis hors `__pycache__`), le run **refuse de démarrer**
  (sauf `--smoke`).
- **Coût (0.4)** : 6 runs B1 = 3–5 min chacun (mtimes 2026-11-11/12 : 3 min 06 s à
  4 min 37 s, early stopping 30–57 époques). Budget TD : 4 bras × 2 modes × 5 graines ×
  2 protocoles = **80 runs ≈ 4–7 h CPU** (TD2/TD3 moins chers, moins de paramètres).
  Aucune réduction nécessaire (< 3 jours). Ordre recommandé : TD0 d'abord (vérification
  des sorties vs plage §8), puis le reste.

## 10. Statut des analyses

CONFIRMATOIRE : comparaison principale TD2−TD0, TD1−TD0, C6'a/C6'b par bras, métriques
primaires à ν=0.01 (Spearman, R², pente, rel_rmse, ΔR², écart au plafond).
EXPLORATOIRE : TD3−TD2, M6, rang effectif, participation ratio, A1 complet,
protocole heldout_nu005, toute analyse post-hoc.

## 11. Commandes (python3, env torch, racine du repo)

```
python3 main_determinism_check.py                       # DOIT passer (exit 0) avant tout run
python3 main_train_td.py --encoder td0 --protocol heldout_nu --z-target-mode energy_centered --seed 42   # TD0 d'abord
# puis : pour bras in td0 td1 td2 td3, mode in energy_centered none, seed in 42 7 123 2024 999,
#        protocole in heldout_nu heldout_nu005 :
python3 main_train_td.py --encoder {bras} --protocol {protocole} --z-target-mode {mode} --seed {seed} 2>&1 | Tee-Object -FilePath outputs/logs/td_run_{bras}_{protocole}_z{mode}_s{seed}.log
python3 main_evaluate_td.py                             # lit ce prereg JSON, exit 1 si NaN ou conditions != prereg
```

Smoke (2 époques, données tronquées en tmp, aucune écriture dans outputs/) :
`python3 main_train_td.py --encoder td2 --protocol heldout_nu --z-target-mode energy_centered --seed 42 --smoke`
