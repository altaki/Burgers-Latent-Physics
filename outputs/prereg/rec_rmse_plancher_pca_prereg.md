# Pré-enregistrement — C6' : seuil de reconstruction relatif (expériences FUTURES uniquement)

**Portée** : aucun verdict passé n'est recalculé (Step 6, A1, B1 conservent C6 = rec RMSE ≤ 0.15
tel que pré-enregistré). C6' s'applique aux expériences futures uniquement. Une sensibilité de B1
sous C6' est rapportée ci-dessous, **étiquetée hors verdict**.

## Problème constaté (B1, Step 7)

C6 = rec RMSE ≤ 0.15 échoue dans tous les protocoles (Step 6 standard 0.164–0.170 artefacts
committés ; heldout_nu 0.193–0.206 ; B1 heldout_nu005 aphys 0.190–0.205, none 0.169–0.181) et aussi
sans L_z. Le seuil 0.15 a été calibré sur le dataset legacy Nx=128 (README:76,167,204) ; sur
nu_sweep_v1 (Nx=1024) le plancher PCA linéaire vaut 0.188 à k=8 et 0.103 à k=16, donc 0.15 est
inatteignable en linéaire sous le rang ~16.

## Définitions

- rec RMSE = `reconstruction_rmse_test = sqrt(mean((rec−u)²))`, test complet, états standardisés
  (`ml_train_experiment_a_seq.py:960`).
- Plancher PCA : PCA linéaire des états, **standardiseur identique à celui du modèle**
  (`input_standardizer` fit sur le train complet raw, `ml_train_experiment_a_seq.py:188-192` ;
  GlobalStandardizer mean/std scalaires, `ml_datasets.py:154-165`), composantes fit sur le train
  complet, RMSE sur le test complet. Mesuré sur nu_sweep_v1 : k=4 → 0.359, k=5 → 0.298,
  k=6 → 0.225, **k=8 → 0.188**, k=16 → 0.103.

## C6'a — non-infériorité de L_z (critère principal)

- Ratios appariés par graine : r_s = rec(aphys, s) / rec(none, s), s ∈ graines.
- **Test de non-infériorité UNILATÉRAL** : H0 : ratio moyen ≥ 1+δ ; C6'a vrai si la **borne
  supérieure unilatérale à 95 %** mean(r) + t(0.95, n−1)·sd(r)/√n < 1+δ.
- **δ = 0.10**, marge conventionnelle de 10 % fixée a priori (indépendante des données B1).
  Cohérence notée a posteriori : CV de rec(none) en B1 = 3.6 % (0.169/0.181/0.178), 3×CV ≈ 11 %.
- **n = 5 graines pré-enregistrées** {42, 7, 123, + 2 nouvelles}. Calcul de puissance avec
  sd(r) ≈ 0.085 (mesuré B1), δ = 0.10, sous vrai ratio = 1 :
  puissance = P(t_{n−1} < δ√n/sd − t(0.95,n−1)) → n=3 : 23 % (inatteignable en pratique,
  demi-largeur bilatérale t(ddl=2) ≈ 0.21 > δ) ; **n=5 : 68 %** ; n=8 : 90 %.
  n=5 retenu (proposition revue), puissance 68 % notée comme limite.

## C6'b — sanité plancher (k fixe)

- C6'b vrai si rec(aphys) ≤ plancher PCA à **k = latent_dim = 8** = 0.188 (k fixe, NON indexé sur
  le rang 95 % du run, qui récompenserait la contraction de rang).

## Sensibilité B1 sous C6' (HORS VERDICT, n=3)

- C6'a : ratios 1.211/1.046/1.096, mean 1.118, sd 0.085, borne sup unilatérale 95 %
  (t(0.95,2)=2.920) = 1.118 + 2.920×0.085/√3 = **1.261 > 1.10 → C6'a échoue**.
- C6'b : rec aphys B1 moy **0.196 > 0.188 → C6'b échoue** (none 0.169–0.181 ≤ 0.188 passerait).

## Falsification

C6'a échoue si la borne sup unilatérale du ratio dépasse 1+δ (surcoût significatif de L_z) ;
C6'b échoue si rec(aphys) > plancher k=8 (le modèle ne bat pas une PCA linéaire de même
dimension). Si les deux passent dans une expérience future, le coût de reconstruction est
imputable au rang/dimension, pas à L_z.
