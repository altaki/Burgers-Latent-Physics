# Amendment 1 au pré-enregistrement TD — BROUILLON POUR REVUE, NE PAS COMMITER

**Statut** : purement déclaratif. Aucune règle du pré-enregistrement
(`td_encoder_prereg.md` / `.json`, commit 515cf5b) n'est modifiée. Aucun verdict passé
n'est recalculé. Aucun entraînement TD1–TD3 n'a été lancé (preuve §7).

## 0. Définition du R² — clarification déclarative, fixée AVANT tout run TD1–TD3

**Citations mot pour mot du prereg** :
- md ligne 68 : « Spearman(z, νD), **R² affine νD~z**, pente de calibration, rel_rmse » ;
- JSON : `/metriques/primaires/liste[1] = 'r2_affine_nuD_z'` ;
  `/decision_rule/comparaison_principale/metric = 'r2_affine_nuD_z'` (idem D2 et D3) ;
- md §8 ligne 115 : « TD0 (référence re-entraînée sur code déterministe) : R² ∈ [0.78, 0.90]
  (Step 6 : 0.844, sd 0.047), Spearman ∈ [0.95, 0.98] ».

**Citations mot pour mot du code** (tous les chemins de calcul) :
- `ml_train_experiment_a_seq.py:593-595` : `ss_res = float(np.sum((pred - targ) ** 2))` ;
  `ss_tot = float(np.sum((targ - targ.mean()) ** 2))` ; `r2 = 1.0 - ss_res / ss_tot`
  (pred = z, targ = νD, **sans ajustement affine**) — le docstring ligne 781 dit
  « R² affine(nuD ~ z) » mais la formule est identité ;
- `main_evaluate_td.py:88` : `r2 = 1 - np.sum((z - y) ** 2) / np.sum((y - y.mean()) ** 2)`,
  stocké sous la clé `r2_affine_nuD_z` (ligne 89) ;
- `main_evaluate_b1.py:86` : même formule, même label de clé.

**Déclaration** : le texte du prereg est ambigu (il dit « affine »), le code est uniforme
(identité partout). La métrique R² retenue pour D1/D2/D3, §8, C6'a/C6'b et toutes les
comparaisons est la **R² identité** `R² = 1 − Σ(z−y)² / Σ(y−ȳ)²` à ν held-out, z en unités
physiques — celle que calcule tout le code (`ml_train_experiment_a_seq.py:595`,
`main_evaluate_td.py:88`, `main_evaluate_b1.py:86`). Le label « affine » dans les noms de
clés est un héritage de nommage trompeur, sans effet sur les valeurs. La calibration est
rapportée séparément via `calibration_slope` (régression de y sur z). La R² affine au sens
strict (lstsq pente+ordonnée) n'est rapportée qu'en sensibilité : TD0 heldout_nu = **0.8845**
affine vs **0.8747** identité.

**Sur quelle définition §8 [0.78, 0.90] a été calibrée** : sur la R² identité du Step 6 —
les runs `window_w25_heldout_nu_zenergy_centered_s{42,7,123}` donnent en identité
0.8589 / 0.7915 / 0.8811, moyenne **0.8438 ≈ 0.844** (prereg md:115), calculées par
`ml_train_experiment_a_seq.py:595`. **TD0 corrigé est dans la plage pour les deux
définitions** : identité 0.8747 ∈ [0.78, 0.90] et affine 0.8845 ∈ [0.78, 0.90] → §8 non
déclenché quelle que soit la définition retenue.

## 1. Bug de filtre `heldout_nu` — déclaration, preuve, commits

**Déclaration** : la clause §8 du prereg (« TD0 hors de [0.78, 0.90] → STOP, investiguer »)
s'est déclenchée sur les 10 premiers runs `td0_heldout_nu_*` (R² identité ν=0.01 ∈
[0.9258, 0.9614]). Cause identifiée : filtre inversé dans `main_train_td.py:48`
(`heldout_nu_values=(0.01,)` au lieu de `(0.05, 0.2)` ; prereg md:40, `config.py:417`).
Les 10 runs ont entraîné sur **ν=0.01 seul** (in-distribution).

**Preuve à 8 décimales** (reproduite depuis `outputs/data/train_data_nu_sweep.npz`,
réplication exacte de `ml_train_experiment_a_seq.py:997-1010` + `index_map`
`ml_datasets_experiment_c.py:575-580`, w=25, stride=1) :

| fit train | mean | std |
|---|---|---|
| complet {0.01,0.05,0.2} | 0.16903235 | 0.20018719 |
| {0.05,0.2} | 0.16934350 | 0.18047746 |
| {0.01,0.2} | 0.17047177 | 0.20604785 |
| {0.01} | 0.16841002 | 0.23469114 |

- runs archivés `td0_indist_nu001_zenergy_centered_*` : standardiseur enregistré
  0.168410/0.234691 = fit **{0.01}** → entraînés sur ν=0.01 seul (bug confirmé) ;
- runs corrigés `td0_heldout_nu_zenergy_centered_*` (commit 4f3b713) :
  0.169343/0.180477 = fit **{0.05,0.2}** → conformes au prereg ;
- runs `td0_heldout_nu005_z*` : 0.170472/0.206048 = fit **{0.01,0.2}** → conformes.

**Commits** : `36ba701` (correction de bug déclarée : module `td_protocols.py` source de
vérité unique des protocoles, importé par `main_train_td.py`, `main_validate_run.py`,
`main_determinism_check.py`, `main_evaluate_td.py` ; garde-fou `expected_train_nu` refusant
tout run dont l'ensemble ν effectif (lu sur `index_map`) diffère du prereg ; test
`python3 td_protocols.py --check`) ; `0ff901a` (troncature smoke stratifiée) ;
`d844a7a` (archivage `git mv` des 10 runs erronés → `td0_indist_nu001_*` + note
`outputs/prereg/td0_indist_nu001_archive_note.md`) ; `4f3b713` (garde-fou lit l'ensemble ν
sur `index_map`, pas sur `nu_values` brut) ; `1a93648` (artefacts des 10 runs corrigés,
SHA-256).

**Runs erronés lus avant correction** : oui — leurs métriques ont été lues et ont déclenché
la clause §8 avant que la cause ne soit identifiée. Le brouillon précédent de cet amendment
(cause « correctif de déterminisme », garde-fou R² ≥ 0.90 basé sur 0.9448 in-distribution,
puissance estimée avec sd ≈ 0.02) est **abandonné** : le saut 0.86 → 0.96 à ν=0.01
s'explique par le filtre inversé, pas par le déterminisme.

**Effet du code déterministe — formulation retenue** : aucun effet démontré, effet faible
non exclu ; runs historiques et runs TD ne sont jamais comparés entre eux. **Différences
par graine (non appariées au sens strict)** — une définition unique des deux côtés, la R²
identité définie en §0, calculée depuis les `predictions_test.npz` à ν held-out :
- TD0 − Step 6 (heldout_nu, ν=0.01, graines 42/7/123) : [+0.0363, +0.0530, −0.0018],
  moy **+0.0292**, sd 0.0281, IC95 t(ddl=2) **[−0.0406, +0.0989]** ; en R² affine :
  moy +0.0118, IC95 [−0.0207, +0.0443] ;
- B1 − TD0 (heldout_nu005, ν=0.05, graines 42/7/123) : [+0.0196, −0.0035, +0.0072],
  moy **+0.0078**, sd 0.0116, IC95 t(ddl=2) **[−0.0209, +0.0365]** ; en R² affine :
  moy +0.0056, IC95 [−0.0072, +0.0184].

Toutes les IC incluent 0. « Non appariées au sens strict » car le générateur du DataLoader
a changé entre le code historique et le code TD : la même graine ne produit pas la même
trajectoire d'optimisation, donc la graine n'apparie que nominalement dans ces comparaisons
historiques (à l'intérieur de l'expérience TD, même code + mêmes données + même graine =
même run déterministe, l'appariement par graine y est strict).

## 2. Référence TD0 corrigée — 20 runs (5 graines × 2 protocoles × 2 modes)

Définitions métriques = `main_evaluate_td.py` exactement, R² identité per §0
(`1−Σ(z−y)²/Σ(y−ȳ)²` à ν held-out), pente = régression de y sur z, rel_rmse =
RMSE(z,y)/mean(y), M6 prereg A1 5623042. Graines {42, 7, 123, 2024, 999}.

**heldout_nu (test ν=0.01), mode z** : R² = **0.8747 ± 0.0201**, IC95 t(ddl=4)
[0.8497, 0.8997] ; Spearman 0.9645 ± 0.0086 ; pente 1.009 ± 0.047 ; rel_rmse 0.5044 ;
biais z−νD +0.0205 ; ordonnée −0.0219 ; M6 0.7427 ± 0.0974 ; rec 0.1969 ± 0.0065 ;
écart au plafond 0.9995 − 0.8747 = **0.1248**. Mode none : rec 0.1620 ± 0.0054.

**heldout_nu005 (test ν=0.05), mode z** : R² = 0.9238 ± 0.0084 ; Spearman 0.9824 ;
pente **1.1637 ± 0.0232** (sous-dispersion réelle à ν=0.05, comme B1) ; rec 0.1899 ± 0.0047.
Mode none : rec 0.1775. En heldout_nu corrigé la pente est 1.009 (sous-dispersion absente) —
la sous-dispersion n'est pas qualifiée d'artefact en général.

**Clause §8 réévaluée sur les runs corrigés** : R² identité = 0.8747 ∈ [0.78, 0.90]
(affine 0.8845 aussi dans la plage) et Spearman = 0.9645 ∈ [0.95, 0.98] → **§8 NON
déclenché ; garde-fou d'origine conservé**.

## 3. Règle de décision D1/D2/D3 — citée mot pour mot, INCHANGÉE

Depuis `td_encoder_prereg.json` (`decision_rule`) :

- **D1** (CONFIRMATOIRE) : `bras_haut=td2, bras_bas=td0, metric=r2_affine_nuD_z, nu=0.01,
  appariement=par graine (meme seed), n=5, delta_R=0.05, ic=unilateral_95, distribution=t_ddl4,
  t_0.95_4=2.1318`. Verdicts : `REPRESENTATIONNEL si mean(dR2) >= 0.05 AND
  lower_unilateral_95(dR2) > 0` ; `NON REPRESENTATIONNEL si upper_unilateral_95(dR2) < 0.05` ;
  `INDETERMINE sinon`. (La métrique `r2_affine_nuD_z` s'entend désormais comme la R²
  identité définie en §0 — c'est ce que calcule l'évaluateur.)
- **D2** (CONFIRMATOIRE) : `bras_haut=td1, bras_bas=td0`, même métrique, ν=0.01, n=5,
  δ_R=0.05, IC unilatéral 95 % t ddl=4.
- **D3** (EXPLORATOIRE, injection architecturale) : `bras_haut=td3, bras_bas=td2`, ν=0.01, n=5.

Texte md §6 (lignes 90-104) inchangé, notamment : « REPRÉSENTATIONNEL si mean(d) ≥ δ_R ET
borne inférieure unilatérale 95 % de mean(d) > 0. NON REPRÉSENTATIONNEL si borne supérieure
unilatérale 95 % de mean(d) < δ_R (on exclut un gain ≥ 0.05 : le pooling + différences
temporelles n'expliquent pas l'écart). INDÉTERMINÉ sinon. »

**Puissance informative recalculée** (dispersion réelle des runs corrigés, R² identité) :
sd inter-graines R²(TD0, heldout_nu) = 0.0201 → sd_diff ≈ √2·0.0201 = 0.0284 sous hypothèse
d'indépendance entre bras (**conservatrice**) ; MDE à 80 % = 0.039 ; puissance δ=0.05 → 0.94,
δ=0.125 (écart réel au plafond) → 1.00 ; P(NON-REPR | Δ=0) = 0.93. Borne Step 6
(sd_diff 0.0665) : MDE 0.090, puissance δ=0.05 → 0.40. **n = 5 maintenu.**

## 4. C6'a / C6'b (hérités de 29c7c3e, inchangés) — rapportés sur TD0 corrigé

Plancher PCA k=8 recalculé par protocole (standardiseur scalaire = train complet comme le
modèle, `ml_train_experiment_a_seq.py:265-268` ; PCA fit sur le train filtré du protocole ;
test complet ; métrique standardisée = `reconstruction_rmse_test`, ligne 1102) :
complet **0.1875** (cohérent avec 0.188 du prereg) ; {0.05,0.2} **0.1869** ;
{0.01,0.2} **0.1897** ; {0.01} 0.1919. Variante avec standardiseur refit sur le train filtré
(plus défavorable) : {0.05,0.2} → 0.1985, {0.01,0.2} → 0.1927 — C6'b échoue dans les deux
variantes en heldout_nu.

- **heldout_nu** : C6'a ratios rec(z)/rec(none) par graine {1.198, 1.311, 1.179, 1.234,
  1.158}, moy **1.2161**, sd 0.0599, borne sup unilatérale 95 % = **1.2732 > 1.10 → C6'a
  ÉCHOUE**. C6'b : rec_z 0.1969 > 0.188 (plancher commité) et > 0.1869 (plancher protocole)
  → **C6'b ÉCHOUE**.
- **heldout_nu005** : C6'a moy 1.0715, borne sup unil. 95 % = 1.1225 > 1.10 → **ÉCHOUE**
  (marginal). C6'b : rec_z 0.1899 > 0.188 et > 0.1897 → **ÉCHOUE** (marginal).

## 5. Budget révisé et ordre des runs

Durées mesurées (runs corrigés, logs `outputs/logs/td_corrected/`) : heldout_nu z ≈ 12 min,
none ≈ 32 min ; nu005 z ≈ 17 min (s42 = 964 min, anomalie non reproductible), none ≈ 20 min.
**Budget total 3 bras × 2 modes × 5 graines × 2 protocoles ≈ 20 h CPU.**
Ordre : **heldout_nu TD2 → TD1 → TD3, puis heldout_nu005** (TD2 d'abord : comparaison
principale D1).

## 6. Vérification des runs `none` sans standardiseur de cible (audit §4 demande)

`window_w25_standard*`, `window_w50_standard*` : `heldout_nu_values = None` dans leurs
metrics → pas de filtre ν → train complet (45 600 paires à w=25, 30 600 à w=50, calcul depuis
`train_data_nu_sweep.npz` et `index_map`) → conformes au protocole `standard`.
`window_w25_heldout_nu*`, `window_w50_heldout_nu*` : `heldout_nu_values = [0.05, 0.2]` →
30 400 paires à w=25 → conformes.

## 7. Preuve qu'aucun run TD1–TD3 n'existe

`ls outputs/models/latent_dynamics_autoencoder_seq/ | grep -c "td[123]"` → **0** (vérifié le
jour de la rédaction de ce brouillon). L'évaluateur `main_evaluate_td.py:177-183` construit
les chemins par nommage exact `td{bras}_{protocole}[_z{mode}]_s{seed}` à partir des bras du
prereg — les dossiers archivés `td0_indist_nu001_*` ne sont jamais lus (aucun glob).
