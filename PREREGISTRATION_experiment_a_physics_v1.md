# Pré-enregistrement — Experiment A-Physics (v1)

- **Date** : 2026-11-11T12:53:31Z (UTC)
- **Version** : v1 — figé et commité **avant tout run de production** A-Physics.
- **Dataset** : `outputs/data/{train,val,test}_data_nu_sweep.npz` (nu_sweep_v1, lecture seule).
  Faits mesurés sur le npz : Nx=1024, 101 snapshots, dt_saved=0.02 (T=2.0),
  max|mass0| = 3.49e-15 (zero_mean sur TOUTES les familles), max|u| = 1.79,
  ν ∈ {0.01, 0.05, 0.20} stratifié 200/50/50 (train) et 50/50/50 (val/test).
- **Protocole d'évaluation** : heldout_nu (train/val restreints à ν ∈ {0.05, 0.20}) ;
  métrique principale sur la strate **ν = 0.01 du test** (50 trajectoires, ~3 800 fenêtres
  de paires w=25) : Spearman(z, νD), Pearson, R² affine, rel_rmse, pente de calibration.
  z évalué en **unités physiques** (inverse de la standardisation de cible).

## 1. Faits mesurés avant enregistrement (npz, lecture seule)

1. Standardiseur de la cible z, population exacte des paires train heldout
   (n=30 400, t∈[24,99], ν∈{0.05,0.2}) : **mean = 0.169343, std = 0.180477**.
   (Les chiffres 0.1734/0.1732 cités en smoke étaient ceux d'un run tronqué à 1 200 paires.)
2. νD n'est PAS monotone à bas ν (test) : fraction de pas νD(t+1)>νD(t) = **68.8 % à ν=0.01**
   (100 % des trajectoires, pic médian t=1.58), 38.5 % à ν=0.05, 0 % à ν=0.2.
3. Hinge numériquement inerte à λ_mono=0.1 : |Δz| par pas (z standardisé, train heldout)
   médiane 0.0069, p95 0.092 → contribution hinge² moyenne 1.5e-4 → **≈ 1.5e-5 à λ=0.1**
   contre z_loss O(0.01–1).
4. Queue lourde (même population paires, z standardisé) : kurtosis = **8.8**,
   top 1 % des échantillons = **21.7 %** de Σz², top 5 % = 50.3 %.
5. Hors plage : max νD train heldout (paires) = **1.8121** ; max νD test ν=0.01 = **1.4509**
   → **0.00 % des fenêtres test ν=0.01 hors plage** (paires et all-t). 5.2 % des fenêtres
   test ν=0.01 ont |z| > 3 (max +7.05).
6. Cible centrée νD_est = (E_{t−1}−E_{t+1})/(2·dt) vs νD stocké (test) : rel err médiane 1.3e-4,
   p95 3.5e-3, max 0.145 **au bord t=0 uniquement** (one-sided ; hors bords max = 0.0182, top-10
   dans la phase de choc à ν=0.01, max|u_x| 55–70). Spearman = 0.999999.
7. L_z s'applique **à z_t uniquement** (ml_losses.py:500-501), jamais à z_tp ; t+1 ≤ 100
   toujours valide (index_map t ∈ [w−1, n_times−stride−1]) ; branche t==0 morte pour w ≥ 2.

## 2. Références (w=25, ν=0.01 held-out, test)

| Référence | Spearman | R² | pente | rel_rmse | Source |
|---|---|---|---|---|---|
| Plafond analytique (centrée, sans réseau) | 1.0000 | 0.9999 | 0.993 | 0.012 | `experiment_a_seq_analytic_baseline.json` |
| **C2 supervisé c3_window w25 held-out (référence principale)** | **0.973** | **0.690** | 0.90 | 0.795 | réexporté du checkpoint, `bootstrap_references.json` ; IC boot-95 % [0.922, 0.986] |
| Sonde linéaire sur h (window-A) | 0.655 | — | — | — | `h_linear_probe_window_a.json` |
| window-A none (s42) | 0.249 | −0.37 | ~0 | 1.68 | réexporté, `bootstrap_references.json` |

## 3. Plan de runs (design apparié : mêmes graines pour toutes les variantes)

| Run | z-target | λ_z | λ_mono | graines |
|---|---|---|---|---|
| `aphys_mono0` (principal) | energy_centered | 1.0 | **0** | 42, 7, 123 |
| `aphys` (ablation hinge) | energy_centered | 1.0 | 0.1 | 42 |
| `asup` (contrôle bruit FD) | supervised | 1.0 | 0.1 | 42 |
| `wina` (window-A apparié) | none | 0 | 1.0 (défaut) | 7, 123 (s42 existe) |

w=25, protocole heldout_nu. Chaque run écrit `predictions_test.npz` (tous modes).

## 4. Prédictions et falsification (ν=0.01 held-out, w=25)

- Si L_z fonctionne : **Spearman(z,νD) ∈ [0.85, 1.0]**, R² affine ∈ [0.6, 0.9999],
  pente de calibration ∈ [0.8, 1.2].
- **Falsification** : Spearman < 0.655 (sonde linéaire sur h) → L_z sélectionne moins bien
  que ce qui est déjà dans h : problème de sélection/optimisation, pas d'information.
- **4 issues pré-enregistrées** : (i) Spearman ≥ 0.95 ≈ C2 → z préserve νD comme la
  supervision directe ; (ii) 0.655–0.95 → préservation partielle ; (iii) < 0.655 → échec
  de la sélection ; (iv) > 0.973 (C2) → L_z dépasse la supervision directe (prudence :
  cibles identiques à 0.35 % près).

## 5. Règle de décision statistique

- Comparaisons **appariées par graine** (mêmes graines entre variantes) : différences
  appariées, moyenne ± écart-type, **IC à 95 % de la différence appariée** (t de Student,
  ddl = n−1 ; n=3 → t=4.303). Un effet n'est déclaré que si l'IC exclut 0 **et** dépasse la
  distance aux références fixes du §2 (les seuils absolus 0.05 / ±0.03 sont abandonnés :
  non défendables sans estimation du bruit inter-graines).
- **Deux IC rapportés séparément** par run (`main_bootstrap_a_seq.py`) : IC bootstrap par
  trajectoire (B=2000 rééchantillonnages des 50 trajectoires ν=0.01) et IC inter-graines
  (métriques ponctuelles des 3 graines).
- asup vs aphys : cibles identiques à 0.35 % (p95) → tout écart observé = bruit
  d'optimisation ; asup n'est pas un test physique.

## 6. Conditions pré-enregistrées d'évolution du protocole

- Cible en **log(νD)** (ou perte de Huber sur z standardisé) **seulement si** la pente de
  calibration à ν=0.01 sort de [0.8, 1.2] (queue lourde mesurée §1.4–1.5).
- Variante « tête z sur encodeur gelé » **retirée** : avec une tête linéaire, c'est
  exactement la sonde linéaire déjà mesurée (0.655) ; une tête MLP non linéaire n'ajouterait
  qu'une transformation monotone, sans effet sur Spearman.

## 7. Critères de verdict (v2, inchangés)

`verdict_a_seq` v2_aphysis : r_holdout ≥ 0.8, R²_affine ≥ 0.5, pente ∈ [0.5, 2],
beats ν-oracle, structural = rang ≥ 3 + rec RMSE ≤ 0.15, **monotone_check =
not_applicable** (z_target_mode ≠ none) ; `verdict_version` + seuils snapshottés dans le JSON.

## 8. Interprétation (garde-fou de circularité)

energy_centered supervise z par νD **via** l'identité dE/dt = −νD. Ces runs testent la
**préservation d'une observable injectée**, pas sa découverte.
