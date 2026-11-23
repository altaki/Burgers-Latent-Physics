# Burgers-Latent-Physics

Can neural networks discover physically meaningful latent scalar observables from the
dynamics of the viscous 1D Burgers equation?

    u_t + u u_x = nu * u_xx      on [0, L] periodic

## Synthèse (v1.0-preservation)

**Résultat central : préservation d'une observable injectée, pas découverte.** Sans
supervision, z ne s'aligne pas sur νD (window-A : Spearman 0.25–0.44 @ν=0.01 held-out) ;
injectée via l'identité exacte dE/dt = −νD, νD est **préservée** (Step 6 A1 « traceur
dynamique » ; B1 ν=0.05 held-out R² 0.927, Spearman 0.985 ; TD3 R² 0.977).

1. **Plafond analytique** (estimateur 2 snapshots, sans réseau) : R² identité 0.9995 @ν=0.01
   (`outputs/logs/experiment_a_seq_analytic_baseline.json`).
2. **Campagne TD** (80 runs, prereg `515cf5b` + amendment `182c8fc`, artefacts `91199bc`) :
   verdicts machine D1 (TD2−TD0) et D2 (TD1−TD0) **INDÉTERMINÉS**, D3 (TD3−TD2, exploratoire,
   injection) REPRÉSENTATIONNEL ; écart au plafond 0.125 (TD0) → 0.022 (TD3) fermé presque
   entièrement par l'injection de ∫u·∂ₜu ; aucun biais générique testé ne le ferme.
3. **Résultats négatifs et limites** : C6/C6' jamais satisfaits en heldout_nu (coût de
   reconstruction de la perte z ≈ +22 % relatif : rec_z 0.197 vs rec_none 0.162, et
   rec_z > plancher PCA k=8 = 0.187) ; 3–5 graines par bras ; dataset unique (1D, T=2,
   3 valeurs de ν) ; la strate test ν=0.01 a été consultée de nombreuses fois depuis le
   Step 5 ; puissances post-hoc à Δ=δ_R ≤ 0.5 (MDE80 0.06–0.08 à n=5).
4. **Portée du transfert** : en ν uniquement (extrapolation ν=0.01, interpolation ν=0.05),
   magnitude dans la plage d'entraînement (0 % des fenêtres test hors plage :
   max νD test 1.4509 < max train 1.8121) ; aucune claim au-delà.
5. **Méthodologie** : preregs commités avant les runs ; bug de filtre prouvé à 8 décimales,
   corrigé, 10 runs erronés archivés ; déterminisme bit à bit validé inter-session ; règle :
   **le contenu d'un commit se vérifie par `git show --stat`, jamais par son message**
   (anomalie `182c8fc` : le message annonçait une note README absente du commit — impact
   nul, note ajoutée en `e49a957`). Inventaire des définitions de R² en Step 9 :
   évaluation = identité, analyses de latents = Pearson² (affine in-sample), sondes
   linéaires = affine ajusté sur train évalué sur test.

Voir « Inventaire de reproductibilité » (fin du README) : SHA-256 des données, lock
d'environnement, commandes canoniques, mapping résultat → artefact.

## Repository layout

Flat prototype (single commit; the `ml/`, `pde/`, `utils/` package dirs were moved to
root-level files — loaders try both layouts). No tests, no requirements file.

```
config.py                     # all hyperparameters (dataclasses) + get_default_config()
pde_burgers_solver.py         # periodic FD + RK4/Euler solver for Burgers
pde_initial_conditions.py     # 3 IC families: fourier / gaussian_bumps / random_smooth
pde_observables.py            # E(u)=1/2 ∫u²dx, D(u)=∫|u_x|²dx, TV, max|u_x|, monotonicity metrics,
                              # compute_experiment_c_targets (5 supervised targets for Experiment C)
pde_dataset_builder.py        # self-contained data generation -> .npz splits
ml_models.py                  # MLPEncoder/Decoder, LatentPredictor, ScalarHead,
                              # EnergyRegressor, LatentAutoencoder, LatentDynamicsModel,
                              # LatentDynamicsAutoencoder (Architecture A) + latent_pca_stats
ml_models_experiment_c.py     # Experiment C models: C1 MLP / C2 CNN / C3 window-CNN, shared encoder + 5 heads
ml_losses.py                  # reconstruction / prediction / monotonicity (squared hinge) /
                              # variance losses (incl. unused latent_variance_regularization) +
                              # latent_dynamics_autoencoder_total_loss (per-dim variance floor)
ml_datasets.py                # BurgersSnapshot/Pair/TrajectoryDataset + GlobalStandardizer
ml_datasets_experiment_c.py   # nu_sweep_v1 snapshot/window datasets + TargetStandardizer (Experiment C)
ml_train_supervised.py        # EnergyRegressor training
ml_train_experiment_c.py      # Experiment C training/eval engine (multi-task MSE, held-out-nu, metrics)
ml_train_experiment_a_seq.py  # window-A (Architecture A-Sequence) training/eval engine (loss A, held-out-nu z-vs-nuD)
ml_train_latent.py            # LatentAutoencoder + LatentDynamicsModel + LatentDynamicsAutoencoder training
ml_evaluate.py                # extraction, regression/correlation/monotonicity metrics +
                              # latent PCA/rank evaluation for Architecture A
evaluate + identify + compare + visualize entry points:
main_generate_data.py, main_train_supervised.py, main_train_latent.py, main_evaluate.py,
main_identify_latent_variable.py, main_compare_z_and_dissipation.py,
main_visualize_results.py, main_visualize_latent_results.py, main_experiment_c.py,
main_experiment_c2.py        # Experiment C2: long temporal windows (c3_window, w=5/25/50)
main_train_latent_seq.py     # window-A: latent dynamics autoencoder on w-snapshot windows (Step 5)
main_analyze_window_a_latents.py  # window-A latent analysis: per-dim h vs nu*D + linear probe (no retraining)
main_analytic_nuD_baseline.py     # analytic nu*D baseline (no network): centered/backward/causal estimators vs stored nuD
physics_candidates.py         # ~37 candidate observables + ranking vs a latent scalar
utils_plotting.py             # figures
outputs/                      # data/ models/ figures/ logs/ (all results committed)
```

## Workflow

```
python main_generate_data.py        # train/val/test .npz + reports
python main_train_supervised.py     # energy regressor
python main_train_latent.py         # latent models (mode hardcoded; incl. 'latent_dynamics_autoencoder')
python main_evaluate.py             # test-split reports for all checkpoints (incl. Architecture A)
python main_identify_latent_variable.py   # candidate rankings vs z and vs each h_j (Architecture A)
python main_compare_z_and_dissipation.py  # z vs nu*∫|u_x|² + residual analysis
python main_visualize_results.py / main_visualize_latent_results.py
python main_experiment_c.py [--variants c1_mlp c2_cnn c3_window] [--protocols standard heldout_nu]
                              # Experiment C: supervised separability baseline on nu_sweep_v1
                              #   options: --window-sizes W [W...] (c3_window), --tag TAG
python main_experiment_c2.py  # Experiment C2: c3_window with w = 5 25 50, tag "c2"
python main_train_latent_seq.py [--protocols standard heldout_nu] [--window-size 25] [--tag aseq]
                              # window-A (Step 5): latent dynamics autoencoder on w=25 windows,
                              #   verdict z-vs-nuD in outputs/logs/experiment_a_seq_summary_{tag}.json
python main_analyze_window_a_latents.py [--window-sizes 25 50] [--protocols standard heldout_nu]
                              # analyse des latents h des checkpoints window-A (sans ré-entraînement)
python main_train_latent_seq.py --tag aphysics --z-target energy_centered --z-weight 1.0 --mono-weight 0.1
                              # Experiment A-Physics (Step 6): z ancré sur nuD estimé par -dE/dt
python main_train_latent_seq.py --tag asup --z-target supervised --z-weight 1.0 --mono-weight 0.1
                              # contrôle supervisé: z ancré sur le nuD stocké
python main_analytic_nuD_baseline.py   # baseline analytique nuD_est vs nuD stocké (sans réseau)
```

## Data

L = 2π, Nx = 128, ν = 0.05, T = 2.0, RK4 with dt = 2e-4, snapshot every 100 steps
→ 101 snapshots at Δt = 0.02. Splits: 500/100/100 trajectories, `states` (n_traj, 101, 128).
ICs drawn uniformly from {fourier (k≤3, amp 0.5), gaussian_bumps (2 bumps), random_smooth
(spectral decay 3.0)}, each L2-normalized to 1 → **E(0) = 0.5 exactly for every trajectory**.
Energy is non-increasing on 100% of trajectories; mean energy decays 0.5 → ~0.30–0.33.

### nu_sweep_v1 dataset (Step 3, generated + audited)

`outputs/data/{train,val,test}_data_nu_sweep.npz` (legacy `*_data.npz` untouched):
600/150/150 trajectories, 101 snapshots, **Nx = 1024, dt = 1e-4, save_every = 200**, RK4,
L = 2π, T = 2.0 (Δt_saved = 0.02). ν ∈ {0.01, 0.05, 0.20} stratified 200/50/50 per split
(contiguous blocks), IC families round-robin within each ν stratum (67/67/66 per ν in train,
17/17/16 in val/test), target L2 norm log-uniform(0.5, 2.0) → E(0) = ½·norm² ∈ [0.125, 2.0],
**zero_mean=True for ALL families** (frozen spec: mass0 ≈ 0 everywhere, ≤ 3.5e-15 — this
overrides the Step-3 plan line that kept gaussian_bumps mass). New npz keys:
`dissipation_physical` (= ν·D), `target_l2_norm`, `energy0`, `mass0`, `trajectory_seeds`,
`initial_cell_peclet`, `dataset_version="nu_sweep_v1"`, `protocol`.

## Models (all plain MLPs, ReLU, latent_dim = 8, Adam lr 1e-3, batch 64, 100 epochs, CPU, seed 42, global standardization)

| Model | Params | Objective | Test result |
|---|---|---|---|
| EnergyRegressor (supervised) | 25,297 | MSE(energy_hat, E) | RMSE 0.0134, Pearson 0.992, Spearman 0.976 |
| LatentAutoencoder | 50,705 | rec MSE + z variance floor | rec RMSE 0.0925; z vs E Pearson −0.02; monotone fraction 0.40 |
| LatentDynamicsModel | 30,553 | latent 1-step MSE + mono hinge on z + z variance floor | latent MSE 4.5e-9; z monotone fraction 0.989; z vs E Pearson −0.136 |
| LatentDynamicsAutoencoder (Architecture A) | 55,961 | rec MSE + latent pred MSE + mono hinge on z + per-dim variance floor on h | rec RMSE 0.0874; latent pred RMSE 0.0092; effective rank 7; monotone 0.66 (max viol 4.3e-7); z vs D Spearman −0.13 |

## Findings (from outputs/logs)

1. **Supervised energy regression works** (RMSE 0.013 on E ∈ [0.30, 0.5]).
2. **The autoencoder's scalar z is not physical** (no correlation with anything, not monotone).
3. **The dynamics model's z is nearly monotone** (98.9% of steps non-increasing) **but is not energy** (Pearson −0.14, affine R² 0.01).
4. **z aligns most with viscous dissipation** ν∫|u_x|²dx: top candidate globally (|Spearman| 0.426, affine R² 0.144), wins on 41/100 trajectories per-trajectory. Residuals of the dissipation fit still correlate with max|u_x| (Pearson 0.73, Spearman 0.76).

## Audit of the latent space (recomputed from checkpoints, see session analysis)

- **h ∈ R⁸ is exactly rank-1**: R²(h_j ~ h_0) = 1 for all j, off-line residual 3e-16; PCA eigenvalues [0.00148, 0, …]. In the encoder, **only 1 of 64 layer-2 units is alive** (dead-ReLU ratchet) → h = W₃[:,27]·a₂₇ + b₃. The AE encoder has 60/64 alive (rank ≈ 7).
- **z is an exact affine function of h** (R² = 1): the scalar head loses nothing; the collapse is in the encoder.
- The predictor stays on the same line; the induced 1-D map is near-deterministic: R²(s_{t+1} ~ s_t) = 0.99998.
- **Why one scalar suffices:** pooled self-maps R²(x_{t+1} ~ x_t) ≥ 0.9994 for E, D, TV, ‖u_x‖, ‖u‖∞, spectral centroid — at fixed ν and normalized ICs the one-step dynamics nearly close in any single decay scalar. But D(t) does **not** determine u(t+1) (R² of Fourier modes at t+1 given D(t): 0.04–0.87) — the collapse loses real state information, because the prediction loss is computed in latent space, not state space.
- **z is mostly a static per-trajectory label**: 95% of Var(z) is between trajectories; cross-sectional corr(z, D) at fixed t = **+0.38 at every one of the 101 time bins** (not a time artifact); corr(z, E)|t ≈ −0.32. Partial corr(z, D | max|u_x|) = 0.18 — the alignment is with the whole gradient/shock complex, inseparable at fixed ν.
- PCA of states: rank-1 reconstruction floor RMSE 0.84 vs rank-8 MLP AE 0.0925; **95.7% of state variance is between-trajectory (IC identity)**.
- dE/dt = −νD exactly in this data (R² = 1.0).

## Interpretation (reviewed and stress-tested)

z behaves as a **noisy, mostly-static dissipation/roughness label of the trajectory plus a
small monotone decay enforced by the loss** — not a classical observable, not energy, not a
clock, and not specifically a shock-event indicator (no shock-timing signature in z). The
per-trajectory "energy wins with |Spearman| = 1" rows in the aggregated CSVs are a
monotonicity artifact (both z and E decrease in time). The rank-1 collapse is **not a bug**:
it is the exact, absorbing global optimum of an objective that (a) evaluates prediction error
in latent space, (b) constrains only one readout direction of h, and (c) never asks h to
retain information about u.

## Limitations

Fixed ν (dissipation ν∫|u_x|², enstrophy ∫|u_x|², max|u_x| are inseparable); E(0) constant by
construction (L2-normalized ICs) — **both addressed by the nu_sweep_v1 dataset (Step 3)**;
T = 2 only; MLP encoder (no translation equivariance);
`main_train_latent.py` has the training mode hardcoded; package files moved to root but not
committed; no README/tests/requirements (this file is the first).

## Step 2 — decided plan: Architecture A (joint reconstruction + latent prediction + monotone head)

New model `LatentDynamicsAutoencoder` = MLPEncoder + MLPDecoder + LatentPredictor + ScalarHead
(all existing classes reused unchanged). New loss (pure composition of existing functions):

    L = λ_rec·MSE(û_t, u_t) + λ_pred·MSE(h_{t+1}^pred, h_{t+1})
      + λ_mono·mean(relu(z_{t+1} − z_t)²) + λ_var·Σ_j relu(min_var − Var(h_t[:, j]))

Patch plan (additive only; existing checkpoints/eval stay valid):

1. `ml_models.py`: add `LatentDynamicsAutoencoder` (no existing class modified).
2. `ml_losses.py`: add `latent_dynamics_autoencoder_total_loss` (reuses reconstruction/prediction/monotonicity losses; **per-dim sum** variance floor — do not reuse the mean-based `latent_variance_regularization` as-is).
3. `config.py`: unchanged (reuse LossConfig fields).
4. `ml_train_latent.py`: add epoch/eval/train functions + `_from_config`; reuse `build_dynamics_dataloaders`.
5. `main_train_latent.py`: add `'latent_dynamics_autoencoder'` mode, output `outputs/models/latent_dynamics_autoencoder/`.
6. `ml_evaluate.py` + `main_evaluate.py`: extractor + evaluator incl. **latent rank** (PCA of h) → `evaluation_latent_dynamics_autoencoder_test.json`.
7. `main_identify_latent_variable.py`: candidate rankings vs z **and vs each h_j**.
8. `main_visualize_latent_results.py`: figure block.

Starting hyperparameters: λ_rec = λ_pred = λ_mono = 1.0, λ_var = 0.01, **min_var = 0.01**
(per-dim floor; healthy AE dims have Var 0.45–7). Fallbacks: pred/rec term ratio > 10 → λ_rec = 10;
rank < 3 at epoch 30 → min_var = 0.05 or λ_rec = 10; monotone < 0.9 → λ_mono = 10;
alive layer-2 units < 8 → switch activation to leaky_relu.

Monitoring (per epoch, one val batch): effective rank, per-dim variances, alive-unit count,
four loss terms, val monotone fraction, corr(z, D).

Success criteria: not exactly rank-1 and effective rank ≥ 3 with ≥ 5 dims above min_var;
reconstruction RMSE ≤ 0.15 (within 2× of the PCA floor for the achieved rank); monotone
fraction ≥ 0.95; z top candidate in the dissipation family (|Spearman| ≥ 0.4, corr(z,D)|t ≥ 0.3);
**≥ 1 non-z dim with |Spearman| ≥ 0.5 vs a distinct candidate** (shock_location,
dominant_mode_index, fourier_amp_k*, ic_type), reported as static vs dynamic; final
pred/rec loss ratio within [0.01, 100].

**Run A0 first** (autoencoder + monotonicity on pairs, no predictor — half a day) to de-risk:
if A0 gives rank ≥ 3 and a monotone z, full A is low-risk.

Follow-up (not in this patch): ν-sweep dataset (ν ∈ {0.01, 0.05, 0.2}) and non-normalized ICs
to make the physical identity of z identifiable (separate ν∫|u_x|² from ∫|u_x|² and max|u_x|,
and let E(0) vary) → decided as **Step 3 (nu_sweep_v1)** below.

## Step 2 — results: Architecture A (implemented, trained, evaluated)

Implementation status: patch items 1–7 done (additive only; `config.py` untouched as planned);
item 8 (figure block in `main_visualize_latent_results.py`) still pending; the A0 de-risking run
was skipped. Artifacts: `outputs/models/latent_dynamics_autoencoder/` (best/last checkpoints,
history JSON, training summary) and `outputs/logs/evaluation_latent_dynamics_autoencoder_test.json`.

Training: best epoch 66 / 100, best val loss 6.54e-3
(λ_rec = λ_pred = λ_mono = 1.0, λ_var = 0.01, min_var = 0.01, squared hinge, MSE prediction).

Test results (`evaluation_latent_dynamics_autoencoder_test.json`):

| metric | value |
|---|---|
| reconstruction RMSE / MAE | 0.0874 / 0.0588 |
| latent prediction RMSE (MSE) | 0.0092 (8.4e-5) |
| PCA eigenvalues of h | [0.927, 0.733, 0.640, 0.433, 0.419, 0.335, 0.279, 0.129] |
| effective_rank (95%) / max_offline_residual | 7 / 3.69 |
| dims above min_var / alive layer-2 units | 8 / 49 |
| monotone_fraction (pairwise) | 0.6597, but max positive violation 4.3e-7 |
| corr(z, E) Pearson / Spearman | +0.169 / +0.212 |
| corr(z, D) Pearson / Spearman | −0.162 / −0.135 |

Success-criteria scorecard: rank ≥ 3 with ≥ 5 dims above min_var **PASS** (7, 8/8);
rec RMSE ≤ 0.15 **PASS** (0.0874); monotone ≥ 0.95 **FAIL as counted** (0.66) but all violations
≤ 4.3e-7 → the count criterion is a metric artifact (squared hinge controls magnitude, not count);
use a tolerance-based count. z in dissipation family (|Spearman| ≥ 0.4) **FAIL** (−0.135);
≥ 1 non-z dim with |Spearman| ≥ 0.5 **PASS** (h2, below); pred/rec MSE ratio 0.011 **PASS**.

**Revised interpretation:** the rank-1 collapse is gone (same blocks — it was objective-driven),
but the dissipation-like z of the old LatentDynamicsModel did **not** survive: with the collapse
removed, corr(z, D) drops from +0.426 to −0.135. The old alignment was induced by the collapse +
hinge — at fixed ν any monotone decay scalar correlates with D (alignment by necessity, not
discovery). In Architecture A the variance floor sits on h, not z, and the squared hinge is
satisfied for free by any non-increasing z, so z drifted to a weak static mix.

## Per-dimension identification of h (Architecture A)

`outputs/logs/latent_dynamics_autoencoder_candidate_rankings_per_dim.csv` +
`latent_dynamics_autoencoder_per_dim_summary.json` (37 candidates × 8 dims):

| dim | best candidate | abs_spearman | pearson | affine R² | between_frac | classification |
|---|---|---|---|---|---|---|
| h0 | negative_mass | 0.40 | +0.53 | 0.28 | 0.95 | static_label |
| h1 | shock_location | 0.44 | +0.44 | 0.19 | 0.93 | static_label |
| h2 | mean_u | 0.57 | −0.85 | 0.72 | 0.96 | static_label, interpretable |
| h3 | negative_mass | 0.24 | +0.20 | 0.04 | 0.88 | static_label |
| h4 | fourier_amp_k2 | 0.26 | +0.27 | 0.07 | 0.91 | static_label |
| h5 | spectral_centroid | 0.20 | +0.17 | 0.03 | 0.89 | static_label |
| h6 | shock_location | 0.33 | −0.30 | 0.09 | 0.93 | static_label |
| h7 | shock_location | 0.37 | −0.36 | 0.13 | 0.85 | static_label |

All dims are static trajectory labels (between_frac 0.85–0.96, consistent with the 95.7%
between-trajectory state variance). The only interpretable dim, h2 ≈ mean_u, is the conserved
mass — and since `zero_mean=True` is the IC default for the spectral families, mean_u is mostly
a gaussian_bumps indicator (see audit below). No dim is dissipation-like.

## Identifiability audit of the current dataset (measured on test_data.npz)

- ν fixed at 0.05 → **Spearman(ν∫u_x², ∫u_x²) = 1.0000 exactly** (constant factor ν);
  dissipation vs max|u_x| 0.89, vs TV 0.90, E vs max|u_x| −0.78 → dissipation / enstrophy /
  shock complex are NOT separable.
- E(0) = 0.5 for every trajectory (L2-normalized ICs) → the only energy variation is temporal
  decay, collinear with every decay scalar (dE/dt = −νD exactly, R² = 1).
- `zero_mean=True` is already the default in `random_fourier_ic` / `random_smooth_ic` (config does
  not override it): mass0 = 0 for all fourier / random_smooth trajectories (77% of the test split),
  only gaussian_bumps carry mass ∈ [−2.45, 2.35]; mass is conserved to 3.6e-15 along trajectories.
- Consequence: on this dataset "z correlates with dissipation" is unfalsifiable — it is
  indistinguishable from "z correlates with any decay scalar".

## Step 3 — nu_sweep_v1 dataset protocol (IMPLEMENTED, generated, audited: GO)

Goal: make physical observables identifiable by breaking the ν and E(0) degeneracies.

- ν per trajectory, stratified uniform over {0.01, 0.05, 0.20}
- target L2 norm ~ log-uniform(0.5, 2.0) per trajectory → E(0) = ½·norm² ∈ [0.125, 2.0]
- `zero_mean=True` applied to **ALL** IC families — measured on the npz: max|mass0| = 3.49e-15
  including gaussian_bumps (the legacy "conserved-mass control" subpopulation no longer exists)
- IC families unchanged: fourier (max_mode=3), gaussian_bumps (n_bumps=2), random_smooth
  (spectral_decay=3.0)
- solver: L = 2π, RK4, dt = 2e-4, T = 2.0, save_every = 100 → 101 snapshots, Δt = 0.02,
  **Nx = 1024** (measured `states.shape[2]`; the legacy spec said Nx=128 / save_every=200 —
  the nu_sweep_v1 npz is the ground truth)
- splits 600/150/150, stratified 200/50/50 per ν, IC families dealt round-robin within each ν stratum
- PILOT GATE for ν = 0.01: 20 trajectories per ν at Nx = 128; pass if ≥ 95% of ν = 0.01
  trajectories have energy_monotone_fraction ≥ 0.99, zero NaN, max|u| ≤ 5, |dE/dt + νD| rel. err
  ≤ 1e-2, and mean E_final ordering E(0.20) < E(0.05) < E(0.01); else regenerate ALL splits at Nx = 256
- per-trajectory seeds recorded; new npz keys: `target_l2_norm`, `mass0`, `dissipation_physical`
  (= nu_values · dissipation), `trajectory_seeds`, `dataset_version`, `protocol`
- new filenames `outputs/data/*_data_nu_sweep.npz` (old npz files untouched)
- additive-only changes to: `config.py`, `pde_dataset_builder.py`, `main_generate_data.py`
  (new `main_nu_sweep()` + argv dispatch; no-arg run unchanged), `physics_candidates.py`
  (per-ν candidate function)
- evaluation protocol: per-ν-stratum and fixed-t correlations of z and each h_j vs
  {E, ν∫u_x², ∫u_x², max|u_x|, TV, mass, ic_type}; held-out-ν transfer (train on ν ∈ {0.05, 0.20},
  test alignment at ν = 0.01 — a true dissipation coordinate keeps corr with ν∫u_x² high there while
  corr with ∫u_x² drops); level-vs-rate separation via the E(0) variation

### Step 3 — audit results (read-only numerical / statistical / physics audit; verdict GO)

Reports: `outputs/logs/nu_sweep_pilot_report.json`, `nu_sweep_generation_summary.json`,
`data_generation_report_nu_sweep_{train,val,test}.json`, `config_snapshot_nu_sweep.json`.

- **Numerical**: all pilot + per-split gates passed. Monotone fraction = 1.0 on 100% of the 900
  trajectories (atol = 1e-8·E(0)); max|u| = 1.85 (gate 5); median |dE/dt + νD|/(νD) = 5.5e-3
  forward / 1.2e-4 central differences (gate 1e-2); max initial cell Peclet 1.14 (gate 2.0);
  diffusive number ν·dt/dx² ≤ 0.531; advective CFL ≤ 0.030. Stored E / D / νD reproduce
  recomputation from `states` to 0.0. Seeds reproduce exactly from
  `default_rng(42+offset).integers(...)`; 900 unique seeds, no overlap between splits.
- **Statistical**: stratification exact (200/50/50 per ν), IC round-robin verified from
  `ic_type_ids` (67/67/66, 17/17/16), target_l2_norm log-uniform (KS 0.038–0.095 vs uniform),
  energy0 = ½·target² to 9e-16, mass0 ≤ 3.5e-15 for all families, no hidden index bias
  (only corr(index, ν) = 0.89 by design and corr(index, Peclet) = −0.74, mechanical).
- **Physics**: Spearman(νD, D) = **0.642 globally** (exactly 1.0 on the legacy dataset) but still
  1.0 within each ν stratum — the identifiability gain is **between-strata only**. At fixed E(0)
  quartile, corr(νD, D) drops to −0.03…0.39. A linear D~νD probe trained on ν∈{0.05, 0.20}
  errs by 91% median at ν=0.01 (held-out-ν signal). νD vs max|u_x| = 0.53 and vs TV = 0.67
  globally (0.89 / 0.90 legacy); D vs max|u_x| = 0.96 and vs TV = 0.95 — the shock complex
  (D, max|u_x|, TV) remains ONE confounded factor. E_final ordering
  E(0.20) < E(0.05) < E(0.01) holds in every split; shocks form at ν=0.01
  (mean max|u_x| 1.36 → 15.1, max 91 by T=2).
- **Verdict: GO** for training the next latent generation on nu_sweep_v1, with
  between-strata / held-out-ν evaluation protocols (within a stratum νD ≡ D by construction).

## Step 4 — Experiment C: supervised separability baseline on nu_sweep_v1 (implemented, validated, production runs done — results below)

Goal: before any latent model, measure how well each observable is predictable from the
states, and whether νD is separable from D **in model space** (not just in correlation tables).

- Targets (5): E, νD = ν∫u_x², D = ∫u_x², TV = ∫|u_x|, max|u_x| (TV / max|u_x| computed from
  `states` by periodic central differences — same discretization as the stored D).
- Models (shared encoder + 5 scalar heads, multi-task standardized MSE, Adam, early stopping,
  best checkpoint): **C1** MLP on a single snapshot (Nx=1024), **C2** 1D-CNN on a single
  snapshot, **C3** 1D-CNN on a 5-snapshot window (tests whether ν is inferable from temporal
  context, since dE/dt = −νD).
- Protocols: `standard` (train on all 3 ν strata) and `heldout_nu` (train on ν∈{0.05, 0.20}
  only, evaluate on test ν=0.01). Evaluation subsets: global, per-ν, per-t (0/1/2), per-E(0)
  quartile, per-trajectory aggregated. Metrics: RMSE, rel RMSE (mean/std), MAE, max abs err,
  Pearson, Spearman, R², calibration slope/intercept. Baselines: global-mean and
  ν-oracle-mean predictors.
- Verdict criteria for νD identifiability (per variant, from `heldout_nu`): r_holdout =
  Spearman(νD̂, νD) at ν=0.01; ρ = rel_rmse_holdout / max in-distribution rel_rmse over
  ν∈{0.05,0.20}; equiv = R²(D̂ ~ νD̂) globally. **IDENTIFIABLE** if r_holdout ≥ 0.8 ∧ ρ ≤ 2 ∧
  equiv < 0.99 ∧ beats ν-oracle; **NOT IDENTIFIABLE** if r_holdout < 0.5 or
  rel_rmse_holdout > 0.5; else **PARTIAL**.
- Files (additive only): `pde_observables.py` (+TV, max|u_x|, `compute_experiment_c_targets`),
  `ml_datasets_experiment_c.py`, `ml_models_experiment_c.py`, `ml_train_experiment_c.py`,
  `main_experiment_c.py`, `config.py` (+`ExperimentCConfig` with all thresholds).
  Validated without torch in the sandbox (py_compile + stub-torch smoke tests on the real
  npz, read-only): recomputed E/D/νD match the stored keys exactly (err 0.0), dataset build
  (snapshot/window lengths, `nu_filter` train+val vs full test, standardizers fit on train
  only, window↔snapshot target alignment, item shapes), metrics/baselines/separability
  functions, verdict logic (IDENTIFIABLE / NOT IDENTIFIABLE / PARTIAL / beats_nu_oracle),
  CLI `--help`, and the Conv1d length arithmetic (Nx=1024 → 512→256→128→64→32,
  flatten_dim=2048). Real training runs need the torch environment; quick synthetic
  self-test there: `python ml_train_experiment_c.py` (12 traj, 2 epochs, no data touched).
- Run:

      python main_experiment_c.py --variants c1_mlp --protocols standard heldout_nu   # first production run
      python main_experiment_c.py --variants c2_cnn c3_window                          # then C2, C3
      python main_experiment_c.py                                                      # or everything at once

  Note on the `nu_oracle` baseline: it is computed from the FULL train split (all 3 ν
  strata, including ν=0.01) even in the `heldout_nu` protocol — an oracle that knows ν and
  all train statistics. This makes `beats_nu_oracle` a conservative (harder) criterion for
  the model; the r_holdout / ρ criteria do not involve the baseline.

- Outputs: `outputs/models/experiment_c/{variant}_{protocol}/` (best_model.pt, metrics JSON,
  metrics_long CSV, standardizers.json) and `outputs/logs/experiment_c_summary.json`
  (sanity + identifiability verdict per variant) + `experiment_c_config_snapshot.json`.
- Interpretation guide: if C3 ≫ C2 ≈ C1, ν is only inferable from temporal context → latent
  models must be evaluated at trajectory level with sequence input; if all variants fail
  held-out-ν, νD remains confounded with D in model space and the next architecture must
  inject ν explicitly (ν-conditioned input, or a physics head predicting dE/dt = −νD).

### Step 4 — results (production runs, seed 42)

6 runs (3 variants × 2 protocols) completed; early stopping at 12–18 epochs (best_epoch 2–8;
train loss keeps dropping while val loss plateaus → mild overfit, not undertraining).
Artifacts: `outputs/models/experiment_c/{variant}_{protocol}/` (best_model.pt, metrics JSON +
long CSV, standardizers.json). Caveat: `outputs/logs/experiment_c_summary.json` reflects only
the LAST invocation (c2_cnn c3_window) — the c1_mlp verdict below was recomputed from its
per-run metrics files with `main_experiment_c.evaluate_identifiability`.

| variant | R² E (std) | R² νD (std) | Spearman νD (std) | r_holdout ν=0.01 | rel_rmse holdout | beats ν-oracle | verdict νD |
|---|---|---|---|---|---|---|---|
| c1_mlp | 0.980 | 0.346 | 0.670 | 0.783 | 2.13 | no | **NOT IDENTIFIABLE** |
| c2_cnn | 0.989 | 0.403 | 0.662 | 0.746 | 2.14 | no | **NOT IDENTIFIABLE** |
| c3_window | 0.994 | 0.638 | 0.806 | 0.874 | 1.97 | no | **NOT IDENTIFIABLE** |

- Sanity: R²_E ≥ 0.98 for all variants (pipeline OK); the "other targets ≥ 0.95" gate fails on
  νD (R² 0.30–0.64) — νD is the hard target by design. CNN variants predict D / TV / max|u_x|
  well (R² 0.94–0.96, Spearman ≥ 0.96).
- In-distribution, **C3 ≫ C2 > C1 on νD** (R² 0.64 / 0.40 / 0.35) → the ν signal lives in the
  temporal window (consistent with dE/dt = −νD). νD is most predictable late (t=2: R² 0.88,
  c3 standard) and per-trajectory aggregated (R² 0.76, Spearman 0.90).
- Held-out ν=0.01: R² collapses (negative) for all variants but **Spearman stays 0.75–0.87**
  (c3 best, 0.874); the per-ν median ratio pred_νD/pred_D is flat (~0.07–0.13) instead of
  tracking ν → at unseen ν the νD head behaves as ≈ c·D head with c ≈ 0.1: **νD remains
  confounded with D in model space**; a 5-snapshot window (0.08 s) does not let the model
  infer ν.
- In the standard protocol, C3's ratio does track ν ({0.017, 0.054, 0.152} vs
  {0.01, 0.05, 0.2}) — the two heads are partially separated in-distribution only.
- Metric caveat: at ν=0.01, νD is heavy-tailed (shocks, max|u_x| up to 91) so even the
  ν-oracle mean predictor has rel_rmse_mean = 1.54 there; the rel_rmse_fail = 0.5 threshold
  is unachievable at ν=0.01 for ANY predictor. The informative held-out criteria are
  r_holdout (0.75–0.87, above the 0.5 fail line) and beats_nu_oracle (failed:
  1.97–2.14 vs 1.54–1.56).
- **Conclusion: νD is NOT identifiable from states by these models at held-out ν**
  (NOT IDENTIFIABLE ×3). The next architecture must inject ν explicitly (ν-conditioned
  input) or use a physics head over a longer temporal context (predict dE/dt = −νD along a
  trajectory, not over a 0.08 s window).

### Step 4 — Experiment C2: long temporal windows (implemented, RUN — results below)

Hypothesis (from the C results): the information about ν in the data is **primarily
temporal** — dE/dt = −νD is exact in nu_sweep_v1, so a window measures νD directly,
ν-agnostically, via −ΔE/Δt. C2 tests whether ν becomes identifiable when the temporal
context grows: **w = 5 (0.08 s, C3 baseline), w = 25 (0.48 s), w = 50 (0.98 s = T/2)**,
same dataset, same 5 targets, same protocols (standard + heldout_nu), same metrics.

- Implementation (minimal, reuses all of Experiment C — `window_size` was already wired
  end-to-end: window dataset → c3_window model → metrics → verdict):
  - `config.py`: +`ExperimentCConfig.window_sizes` (None = `[window_size]`, historical
    behavior unchanged).
  - `main_experiment_c.py`: `run_experiment_c(..., window_sizes=None, tag="")`; runs are
    labeled `c3_window_w{w}` when several window sizes are given (historical `c3_window_*`
    runs are never overwritten; with the default single size the label stays `c3_window`);
    `--window-sizes W [W...]` and `--tag TAG` CLI options (tag suffixes the global reports,
    e.g. `experiment_c_summary_c2.json`).
  - `main_experiment_c2.py` (new, thin wrapper): `python main_experiment_c2.py`
    = c3_window × w ∈ {5, 25, 50} × {standard, heldout_nu}, tag "c2".
  - No change to `ml_datasets_experiment_c.py` / `ml_models_experiment_c.py` /
    `ml_train_experiment_c.py`.
- Outputs: `outputs/models/experiment_c/c3_window_w{w}_{protocol}/` and
  `outputs/logs/experiment_c_summary_c2.json` (verdict per w) + config snapshot.
- Validated without torch: py_compile + stub-torch smoke tests (run labels, summary
  structure, verdicts, tag filenames, backward-compatible labels, CLI) + window datasets
  on the real npz for w = 5/25/50 (lengths 97/77/52 windows per trajectory, shapes
  (w, 1024), window covers [t−w+1, t]).
- Run (torch env):

      python main_experiment_c2.py                          # w = 5 25 50, both protocols
      python main_experiment_c2.py --window-sizes 25 50     # subset

- Expected behavior (predictions to falsify): R² νD standard ↑ with w (0.64 → ~0.85–0.92);
  r_holdout stays high (0.87 → ~0.90–0.95); **R² νD and rel_rmse at held-out ν=0.01 should
  improve with w** (from −0.65 / 1.97 toward positive / ~1.0–1.3, vs ν-oracle 1.54); the
  per-ν ratio pred(νD)/pred(D) at ν=0.01 should approach 0.01. Caveats: secant bias grows
  with w (E(t) convex when shocks form; targets are at the window's final time), fewer
  windows per trajectory (97 → 52), no early-time coverage for large w (first window ends
  at t = (w−1)·0.02; t=0 subset empty for all window variants, t=1/t=2 common to all w).
- Decision rule: if held-out R² νD at ν=0.01 clearly improves with w and beats ν-oracle →
  νD is identifiable from dynamics alone → next: trajectory-level models (sequence over the
  101 snapshots) and latent models with sequence input. If w=50 ≈ w=5 held-out → the temporal
  hypothesis is falsified; the blocker is the dataset (3 ν strata, held-out at the extreme)
  → next: finer ν sweep (6–8 log-spaced values in [0.005, 0.3]) before any new architecture.
  Companion control (cheap, same day): ν-conditioned c2/c3 (ν as input) = upper bound —
  with ν given and D predicted at R² 0.96, νD should be near-perfect even held-out.

### Step 4 — Experiment C2 results (production runs, seed 42)

6 runs (w ∈ {5, 25, 50} × {standard, heldout_nu}). Note: `experiment_c_summary_c2.json`
reflects the LAST invocation (w = 25, 50) — the w=5 verdicts below were recomputed from the
`c3_window_w5_*` metrics files. **Early stopping on all 6 runs (12–38 epochs, patience=10)**:
val loss stops improving for 10 consecutive epochs while train loss keeps dropping
(overfit / val noise) — the best checkpoint (argmin val) is kept, so this is the mechanism
working as designed, not an error. Larger w → later stopping and much lower val loss
(w=5: 0.066/0.102 → w=25: 0.018/0.015 → w=50: 0.014/0.006, standard/heldout).

| w | R² νD std | Spearman νD std | holdout ν=0.01: R² / Spearman / rel_rmse | r_holdout | beats ν-oracle | ratio pred νD/pred D @ν=0.01 holdout | formal verdict |
|---|---|---|---|---|---|---|---|
| 5 | 0.638 | 0.806 | −0.65 / 0.874 / 1.97 | 0.874 | no | 0.109 | NOT IDENTIFIABLE |
| 25 | 0.944 | 0.984 | **+0.69 / 0.973 / 0.79** | 0.973 | **yes** | 0.048 | NOT IDENTIFIABLE (threshold artifact) |
| 50 | 0.944 | 0.989 | **+0.69 / 0.978 / 0.68** | 0.978 | **yes** | 0.051 | NOT IDENTIFIABLE (threshold artifact) |

- **Temporal hypothesis CONFIRMED**: w=25 ≈ w=50 ≫ w=5 (diminishing returns beyond w=25:
  secant bias + fewer windows per trajectory). In-distribution per-ν R² νD at w=50:
  {ν=0.01: 0.949, ν=0.05: 0.913, ν=0.2: 0.993}; the ratio pred νD/pred D tracks ν almost
  exactly in standard ({0.0105, 0.0458, 0.1995} vs {0.01, 0.05, 0.2}) — the two heads are
  separated in-distribution.
- Held-out ν=0.01 at w ≥ 25: R² +0.69, Spearman 0.97, calibration slope 0.86–0.89 (≈1), and the
  model **beats the ν-oracle** (rel_rmse 0.68–0.79 vs 1.54). Strikingly, D and max|u_x|
  collapse held-out (R² 0.26–0.28, OOD shocks) while νD survives (0.69) → the temporal channel
  −ΔE/Δt (dE/dt = −νD exact) carries νD in a ν-agnostic way even when the spatial shock
  features fail.
- The formal verdict stays NOT IDENTIFIABLE only through threshold artifacts at ν=0.01:
  rel_rmse_fail = 0.5 is unreachable there (ν-oracle = 1.54, heavy tail) and
  ρ = rel_rmse_holdout / rel_rmse_indist = 2.1–2.2 > 2 only because in-distribution rel_rmse
  dropped to 0.26–0.42. **Substantively: νD is largely identifiable from dynamics alone at
  held-out ν once w ≥ 25** (residual gap: median ratio at ν=0.01 still ~5× the true 0.01;
  rel_rmse 0.68–0.79 vs in-distribution 0.26–0.42).
- Sanity: R²_E ≥ 0.99 everywhere; sanity_ok=False only because νD R² = 0.944 sits just under
  the 0.95 gate.
- **Next**: (1) trajectory-level sequence model over the full 101 snapshots (w=50 covers only
  t ≥ 0.98; per-trajectory aggregation already reaches R² 0.82–0.84 held-out); (2) cheap
  companion controls: ν-conditioned model (ceiling) and recalibrated verdict thresholds
  (fail line relative to the ν-oracle instead of the absolute 0.5); (3) only if a residual
  held-out gap remains: finer ν sweep near 0.01.

### Note de réinterprétation des Steps 4 et C2 (ajoutée après le Step 6 — résultats historiques ci-dessus inchangés)

Mesures de référence (Step 6 pre-run, `main_analytic_nuD_baseline.py`, test nu_sweep_v1,
lecture seule) :

1. **Le canal qui fait gagner C2 est l'identité d'énergie, et il est quasi sans bruit.**
   La différence centrée de E, (E_{t−1}−E_{t+1})/(2·dt), estime le νD stocké avec
   Spearman 1.0000, R² 0.9999, pente 1.007, rel_rmse 0.012 à ν=0.01 (w=25). « Les longues
   fenêtres rendent νD identifiable » se lit donc plus précisément : *une fenêtre permet de
   mesurer −ΔE/Δt, qui égale νD par identité — aucune inférence séparée de ν n'est
   nécessaire*. La performance held-out de C2 (Spearman 0.97, R² 0.69) se situe **sous** ce
   plafond analytique (1.0 / 0.9999) : le CNN capture le classement mais pas toute la magnitude
   à ν=0.01.
2. **L'échec de Step 4 (w=5) n'est pas informationnel.** La pente causale intra-fenêtre à
   w=5, (E_{t−4}−E_t)/(4·dt), estime déjà νD avec une erreur relative médiane de 2.2 % et
   Spearman 0.9994 (0.9995 à ν=0.01) — la fenêtre de 5 snapshots *contient* νD. Le modèle
   entraîné ne l'a pas extrait (tête νD ≈ 0.1·tête D, R² holdout −0.65) parce que l'encodeur
   partagé privilégie le complexe de choc D (R² 0.96). Cela nuance la conclusion historique
   « une fenêtre de 0.08 s ne permet pas d'inférer ν » : elle permet de *mesurer* νD ; c'est le
   modèle entraîné qui ne l'a pas fait. Cohérent avec le Step 5 : ce qui compte est ce que
   l'objectif sélectionne, pas le contenu informatif brut.
3. **Biais de sécante** : l'estimateur moyenne-fenêtre se dégrade quand w croît (erreur médiane
   2.2 % à w=5 → 14 % à w=25 → 27 % à w=50, à cause du raidissement), alors que le modèle
   s'améliore de w=5 à w=25 — le modèle utilise donc des différences *locales* dans la
   fenêtre, pas la moyenne ; c'est pourquoi la cible centrée (et non la moyenne) a été
   retenue pour A-Physics.
4. **Les verdicts formels « NOT IDENTIFIABLE » de C2** bloquaient sur rel_rmse_fail = 0.5 absolu,
   un seuil calibré dans un autre régime (à ν=0.01 même l'oracle ν vaut 1.54). Ce seuil n'est
   **pas** inatteignable en principe : la baseline backward à 2 points atteint R² = 0.9995 et
   rel_rmse = 0.033 à ν=0.01 (mesuré, `main_analytic_nuD_baseline.py`). L'échec à w=5 est donc
   un problème d'**apprenabilité** (le modèle entraîné n'extrait pas le canal pente d'énergie,
   item 2), pas un déficit d'information ni un artefact de seuil. Le traitement des seuils
   reste documenté au Step 6 (verdict v2 : critère de monotonie non applicable car νD n'est
   pas monotone à faible ν). Substantiellement, νD est identifiable à ν held-out dès w ≥ 25
   via le canal énergie.
5. **Pont Steps 4/C2 → 5 → 6** : window-A a échoué (r_holdout 0.249) précisément parce que la
   loss A ne sélectionne pas le canal pente d'énergie (sonde linéaire sur h : 0.655, donc
   partiellement présent) ; A-Physics l'injecte explicitement (L_z = MSE(z, νD_est)) avec un
   plafond analytique Spearman ≈ 1.0. L'arc est cohérent : l'information est la pente
   d'énergie ; la question ouverte est de savoir si un objectif *latent* peut la sélectionner.

## Step 5 — Architecture A-Sequence (implemented — results: NOT IDENTIFIABLE)

Motivation (from C2): w=25 ≈ w=50 ≫ w=5, and at w ≥ 25 νD is largely identifiable from
dynamics alone at held-out ν (R² 0.69, Spearman 0.97, calibration slope 0.87, beats ν-oracle).
Question: can a **latent** model discover an interpretable z aligned with νD, tested at
held-out ν? Input: **window = 25 snapshots** (0.48 s), the C2 sweet spot.

1. **Reuse of Architecture A (additive only)** — keep `LatentDynamicsAutoencoder`
   (`ml_models.py`), `latent_dynamics_autoencoder_total_loss` (`ml_losses.py`), the training
   loop (`ml_train_latent.py`), and the evaluation stack (`ml_evaluate.py` extractor + latent
   PCA rank, `main_identify_latent_variable.py` candidate rankings). Implemented as:
   - pluggable encoder: `LatentDynamicsAutoencoder(..., encoder=None, output_dim=None)`
     (`ml_models.py`) — None → `MLPEncoder` as today; otherwise use the given module
     (forward signature `forward(u_t, u_tp)` and output keys
     `h_t / h_tp_pred / reconstruction / h_tp / z_t / z_tp` unchanged); `output_dim`
     defaults to `input_dim`, else the decoder reconstructs `output_dim` values;
   - new `NuSweepWindowPairDataset` (in `ml_datasets_experiment_c.py`): items = pairs of
     consecutive stride-1 windows (window ending at t, window ending at t+1) + snapshots
     u_t / u_tp (u_t = reconstruction target, final snapshot of the first window) +
     metadata incl. `nu_dissipation_t`; `nu_filter` for heldout_nu; 76 pairs per trajectory
     at w=25 → ~45.6k pairs in train;
   - factory `build_window_dynamics_autoencoder(window_size, nx, ...)`
     (`ml_models_experiment_c.py`): `Conv1DEncoder(in_channels=w)` + shared trunk →
     `LatentDynamicsAutoencoder(encoder=..., output_dim=nx)`; decoder MLP latent → Nx=1024;
   - new training engine `ml_train_experiment_a_seq.py` (loss A unchanged via
     `latent_dynamics_autoencoder_total_loss`, early stopping + best checkpoint, z-vs-νD
     evaluation reusing the Experiment C subset machinery + ν-oracle baseline, verdict
     `verdict_a_seq`) and new entry point `main_train_latent_seq.py` (CLI
     `--protocols/--window-size/--tag`) → models in
     `outputs/models/latent_dynamics_autoencoder_seq/window_w{w}_{protocol}/`, summary in
     `outputs/logs/experiment_a_seq_summary_{tag}.json`; config `ExperimentASeqConfig`
     (`config.py`).
2. **Losses kept unchanged**: L = λ_rec·MSE(û_t, u_t) + λ_pred·MSE(h_{t+1}^pred, h_{t+1})
   + λ_mono·mean(relu(z_{t+1}−z_t)²) + λ_var·Σ_j relu(min_var − Var(h_j)), same weights as A
   (λ_rec=λ_pred=λ_mono=1, λ_var=0.01, min_var=0.01). Latent prediction step = 0.02 s
   (stride-1 windows: window ending at t → window ending at t+1).
3. **Temporal window encoding**: reuse `Conv1DEncoder` (`ml_models_experiment_c.py`) with
   `in_channels=25`, `nx=1024`, conv_channels (16,32,32,64,64), k=5, s=2 → spatial length
   1024→32, flatten 2048 → trunk → h (latent_dim=8). Time enters as channels (proven in C2);
   a 2D conv (time × space kernels) is the ablation if channel-as-time underperforms. Windows
   normalized with the train `GlobalStandardizer` (same as C/C2).
4. **Held-out-ν evaluation of z**: protocols `standard` + `heldout_nu` (train ν∈{0.05,0.2});
   z computed per window (attributed to its final time t), evaluated vs νD with the Experiment
   C subset machinery (`build_eval_subsets`): per-ν (esp. ν=0.01), per-t, per-E(0) quartile,
   per-trajectory; candidate rankings vs z and each h_j on nu_sweep_v1; latent PCA rank
   (collapse check); monotone fraction of z (tolerance-based).
5. **Success criteria** (held-out ν=0.01, z vs νD; supervised C2 w=25 references in parentheses):
   - r_holdout = Spearman(z, νD) @ν=0.01 ≥ **0.8** (C2: 0.973);
   - R² held-out = affine R²(νD ~ z) @ν=0.01 ≥ **0.5** (C2: 0.69);
   - calibration slope a of νD ≈ a·z + b @ν=0.01 in **[0.5, 2.0]**, ideally ≈ 1 (C2: 0.886);
   - plus: beats ν-oracle rel_rmse; structural (from A): effective rank ≥ 3, rec RMSE ≤ 0.15,
     monotone fraction ≥ 0.95 (tolerance), pred/rec loss ratio ∈ [0.01, 100].
   - Verdict: **IDENTIFIABLE** if all three held-out criteria + structural pass; **PARTIAL** if
     r_holdout ≥ 0.8 but R² or slope fails; **NOT IDENTIFIABLE** if r_holdout < 0.5.
6. **snapshot-A vs window-A vs trajectory-A** (predictions from C/C2):
   - *snapshot-A* (A as-is on nu_sweep_v1, MLP encoder on one snapshot): control. C1/C2 say a
     single snapshot cannot separate νD from D → expect z ≈ shock complex: r_holdout moderate
     (0.7–0.85, via D ranking) but R² held-out ≤ 0 and slope off (magnitude wrong).
   - *window-A* (w=25, proposed): expect r_holdout ≥ 0.9, R² held-out 0.5–0.7, slope 0.7–1.3;
     minimal changes (reuses `Conv1DEncoder`), C2 sweet spot.
   - *trajectory-A* (full 101-snapshot sequence, window encoder + temporal conv/GRU): upper
     bound; gains mainly early-time coverage (w=50 only covers t ≥ 0.98) and temporal
     smoothing — C2 says w=50 ≈ w=25, so held-out νD gain over window-A should be small;
     higher cost (params, data, train time).
   - **Recommendation: implement window-A (w=25) first**, snapshot-A as cheap control,
     trajectory-A only if window-A leaves an early-time gap.

Caveats: on legacy fixed-ν data, A's z was a weak static mix (corr(z,D) = −0.135) and all h
dims were static IC labels; on nu_sweep_v1 mass0 ≈ 0 removes the mass label. The multi-ν
dataset + held-out protocol is what makes z's identity testable. Risk: the mono hinge +
predictor may again yield a generic decay scalar (E-like) rather than νD — the held-out νD
criteria (rank + magnitude + calibration) are exactly what discriminates.

Run (torch env):

```
python main_train_latent_seq.py                          # w=25, protocols standard + heldout_nu, tag "aseq"
python main_train_latent_seq.py --window-size 50         # ablation w=50
python main_train_latent_seq.py --protocols standard     # standard only (no cross-protocol verdict)
python main_train_latent_seq.py --tag aseq_w50 --window-size 50
```

Read `outputs/logs/experiment_a_seq_summary_aseq.json` → `verdict.window_a`:
`r_holdout`, `r2_holdout_nu001`, `calibration_slope_nu001`, `beats_nu_oracle`,
`effective_rank`, `reconstruction_rmse_test`, `monotone_fraction_test`, `structural_ok`,
`verdict` (IDENTIFIABLE / PARTIAL / NOT IDENTIFIABLE), plus per-run metrics JSON/CSV under
`outputs/logs/` and checkpoints under `outputs/models/latent_dynamics_autoencoder_seq/`.

### Results (runs tag "aseq", w=25 and w=50, both protocols)

Per-run metrics: `outputs/models/latent_dynamics_autoencoder_seq/window_w{w}_{protocol}/metrics_window_a_*.json`.
Cross-protocol verdict recomputed post-hoc with the real `verdict_a_seq` in
`outputs/logs/experiment_a_seq_verdict_computed.json` (the summary file only reflects the
last single-protocol CLI run, which overwrote it):

| run (heldout_nu, strate ν=0.01) | r_holdout Spearman(z,νD) | Pearson | rel_rmse (ν-oracle) | verdict |
|---|---|---|---|---|
| window-A w=25 | **0.249** | 0.302 | 1.675 (1.429) | **NOT IDENTIFIABLE** |
| window-A w=50 | **−0.035** | −0.013 | 2.971 (1.219) | **NOT IDENTIFIABLE** |

Structural (protocole standard): effective rank 6–7 (≥ 3 OK, no collapse), rec RMSE
0.153–0.169 (w=50 above the 0.15 threshold), monotone fraction 0.80–0.90 (< 0.95) →
`structural_ok = False`. Reference: supervised C2 on the same windows reaches Spearman
0.973, R² 0.69 @ν=0.01 held-out. w=50 *standard* reaches Spearman 0.61 @ν=0.01
in-distribution but collapses at held-out ν.

**Interpretation**: the νD information IS present in the w=25 window (C2 proves it), but
loss A (rec + pred + mono hinge + variance floor) does not select νD as z — z behaves as a
generic monotone decay scalar (E-like; monotone fraction 0.8–0.9) and its ranking of νD
collapses at held-out ν. No latent collapse (rank 6–7), so this is an identification/loss
failure, not a collapse artifact. Metric caveat: raw `r2`/`calibration_slope` are in z's
arbitrary latent units; the calibrated R² is Pearson² (0.09 @ν=0.01, w=25 held-out) — the
verdict is driven by scale-invariant Spearman anyway.

### Latent analysis of the existing checkpoints (no retraining)

`main_analyze_window_a_latents.py` → `outputs/logs/h_dims_window_a_analysis.{json,csv}`
(per-dim h0..h7/z vs νD: Spearman, Pearson, R² affine, slope, rel_rmse on global/ν strata)
and `outputs/logs/h_linear_probe_window_a.json` (linear readout νD ~ [h,1] / [z,1] fit on
train, evaluated on test).

Key numbers (test, strate ν=0.01 held-out):

| run | best single dim (Spearman @ν=0.01) | z (Spearman) | linear probe on h (Spearman / R² @ν=0.01) | probe on z |
|---|---|---|---|---|
| w=25 heldout_nu | h0: **−0.539** (R²_aff 0.34) | 0.249 | **0.655 / 0.09** | 0.249 / 0.04 |
| w=50 standard | h3: 0.565, h5: 0.385 | 0.614 | **0.777 / 0.40** | 0.614 / 0.33 |
| w=50 heldout_nu | h4: 0.415 | −0.035 | **0.548 / 0.19** | −0.035 / −0.08 |

Dim identification (Spearman vs D / vs E, test global): h0–h2 are anti-correlated with both
D and E (≈ −0.5) → energy/decay-like dims; h3/h4/h7 positively correlated with D → shock
complex. No single dim carries νD at held-out ν (|ρ| ≤ 0.57).

**Answers**:
- *Does νD exist in h but fail to be selected by z?* **Partially yes**: a linear readout on
  h roughly triples the held-out Spearman vs z (0.25→0.66 at w=25; −0.04→0.55 at w=50) and
  yields positive R² at ν=0.01 — νD is present in h in **distributed** form, but the scalar
  z head + mono hinge select a different (E-like decay) scalar. Still far from supervised C2
  (0.97 / R² 0.69): the encoder retains only part of the ν-specific information.
- *Minimal z-objective modification to recover νD*: add a **self-supervised energy-balance
  term** L_z = MSE(z_t, (E_{t−w} − E_t)/(w·dt)) — by the exact identity dE/dt = −νD this
  target IS νD, and E is computable from the window endpoints alone (no labels). Keep the
  mono hinge (νD decays). Ceiling variant: supervised z-head on the stored
  `nu_dissipation_t`.

### Next experiments (ranked)

1. ~~**Per-dim h evaluation of the existing checkpoints**~~ **Done** (see above): νD is in h
   only in distributed form (linear probe 0.55–0.78 @ν=0.01 held-out vs 0.25/−0.04 for z);
   no single h_j carries it.
2. **Physics-informed z (self-supervised)**: add λ_z·MSE(z, −ΔE/Δt) with −ΔE/Δt estimated
   from the window itself, (E(t)−E(t−w))/(w·dt) = νD exactly by the energy identity —
   anchors z to νD without labels, reuses all window-A infra. **Main next run.**
3. **Supervised z-head (ceiling)**: λ_z·MSE(z, νD standardized) on top of loss A — tests
   whether the bottleneck preserves the νD information that C2 reads.
4. **snapshot-A control** (A as-is on nu_sweep_v1): cheap, confirms single-snapshot
   insufficiency (expected from C1/C2).
5. **ν-conditioned models / finer ν sweep near 0.01**: dataset-level options if a residual
   held-out gap remains after 2–3.

- **Écart au pré-enregistrement (définition du R²)** : le critère Step 5 (ligne 569) dit
  « R² held-out = **affine** R²(νD ~ z) @ν=0.01 ≥ 0.5 (C2: 0.69) » ; le code calcule une R²
  **identité** (`verdict_a_seq`, `ml_train_experiment_a_seq.py:801` ← `_subset_metrics:594-595`
  ; la référence C2 0.69 est aussi une identité : `physics_candidates.py:97` via
  `ml_train_experiment_c.py:421`). Identité ≤ affine in-sample ⇒ le critère « ≥ seuil »
  était évalué **plus strictement** que le texte. Verdict inchangé : NOT IDENTIFIABLE est
  décidé par Spearman < 0.5, le critère R² n'est pas décisif ici.

## Step 6 — Experiment A-Physics (implemented, pré-enregistré `212db0f` — voir `PREREGISTRATION_experiment_a_physics_v1.md`, runs done — results below)

Motivation (Step 5 audit): νD is present in h (linear probe 0.55–0.78 @ν=0.01 held-out)
but the scalar z does not select it — the problem is the z-objective. A-Physics anchors z
on the exact energy identity **dE/dt = −νD** via an auxiliary loss
**L_z = λ_z·MSE(z_t, νD_est_t)**, self-supervised (ν-agnostic → held-out ν by construction).

- **νD_est(t) = (E_{t−1} − E_{t+1})/(2·dt)** — centered difference of the stored raw energy
  (E_{t+1} = last snapshot of window_tp, available in the pair dataset). Validated on the
  npz: Pearson/Spearman vs stored νD = 1.000 (finite-difference error ~1e-4 relative).
  The window-mean variant (E_{t−w}−E_t)/(w·dt) lags by w·dt/2 (Pearson 0.93) — rejected.
- **Complete loss**: L = λ_rec·MSE(û_t,u_t) + λ_pred·MSE(h_pred_{t+1},h_{t+1})
  + λ_var·Σ relu(min_var−Var h_j) + λ_mono·mean(relu(z_{t+1}−z_t)²) + λ_z·MSE(z_t, νD_est_t),
  with **λ_rec=λ_pred=λ_z=1.0, λ_var=0.01 (min_var=0.01), λ_mono=0.1**. The z target is
  **standardized** (mean/std fit on the train split of the current protocol only — heldout_nu
  → ν∈{0.05,0.2}); z is inverse-transformed to **physical units** for all evaluation metrics,
  so the calibration slope stays ≈ 1 (the verdict slope window [0.5, 2] is not scale-free).
  Raw physical-unit MSE would be dominated by large-νD samples and underweight ν=0.01.
- **Monotonicity**: measured on test nu_sweep_v1, νD is **NOT monotone at low ν** — at ν=0.01,
  68.8% of steps have νD(t+1)>νD(t) (100% of trajectories affected, median peak at t=1.58);
  ν=0.05: 38.5% (74% of trajectories); ν=0.2: monotone from t=0. A perfect z=νD would have
  monotone_fraction ≈ 0.64 (strict) / 0.84 (rtol 1%) — far below the 0.95 threshold. Hence:
  hinge demoted to smoothing (λ_mono=0.1), and **verdict v2** (`z_target_mode ≠ none`):
  monotone criterion `not_applicable` in `structural_ok` (rank + rec RMSE only), with
  `verdict_version` + thresholds snapshot recorded in the output JSON.
- **z stays scalar** (interpretability goal); z∈R² is only a fallback ablation. Note: with
  `scalar_head_hidden_dims=[]` the z head is linear in h, so "train z-head on frozen encoder"
  ≡ the linear probe already done (0.55–0.78) — no extra run needed.
- **Interpretation guard (circularity)**: energy_centered supervises z with νD *via* the
  identity dE/dt=−νD. These runs test the **preservation of an injected observable**, not its
  discovery. Measured: centered estimate vs stored νD → rel err median 1.3e-4, p95 3.5e-3,
  Spearman 0.999999 — so `energy_centered` and `supervised` targets are numerically almost
  identical; `asup` is kept only as a 1-seed control, not as an independent experiment.
- **Temporal alignment**: z_t is computed from window [t−w+1, t] only. The target uses E_{t+1},
  which comes from the **second window of the pair** (`window_tp`, last snapshot) — fed to the
  model only as a prediction *target*, never as encoder input, so no leakage at inference.
  **L_z applies to z_t only, never to z_tp** (`ml_losses.py:500-501`), hence no border case at
  t+1 = 100: `index_map` gives t ∈ [w−1, n_times−stride−1], so t+1 ≤ 100 is always valid (and the
  t==0 fallback branch of `nuD_est_t` is dead code for w ≥ 2).
  The causal in-window alternative (E_{t−w+1}−E_t)/((w−1)dt) is far worse: rel err median
  0.14 (w=25) / 0.27 (w=50) vs 1.3e-4 centered; R² @ν=0.01: 0.78 / 0.50 vs 0.9999. Centered kept.
- **Analytic baseline (no network)**, `main_analytic_nuD_baseline.py` →
  `outputs/logs/experiment_a_seq_analytic_baseline.{json,csv}`: centered estimator vs stored
  νD on test pairs — **Spearman 1.0000, R² 0.9999, slope 1.007, rel_rmse 0.012 @ν=0.01**
  (w=25; w=50: 0.9999/0.009). This is the ceiling of what L_z can achieve: if z perfectly
  learned νD_est, Spearman(z,νD) ≈ 1.0. Backward local: R² 0.9995. Causal w=25: Spearman
  0.987, R² 0.78. Target validation (rel err per ν / per t / during steepening dD/dt>0) is in
  the same JSON.
- **Implementation (additive, backward compatible)**: `nuD_est_t` metadata in
  `NuSweepWindowPairDataset`; optional `z_target`/`z_weight` in
  `latent_dynamics_autoencoder_total_loss` (inactive by default, `z_loss=0`);
  `z_target_mode` ∈ {none, energy_centered, supervised} + `standardize_z_target` +
  `monotone_rtol` threaded through `ml_train_experiment_a_seq.py`; verdict v2 versioned;
  CLI `--z-target/--z-weight/--mono-weight/--seed/--monotone-rtol/--no-standardize-z-target`;
  run dirs `window_w{w}_{protocole}[_z{mode}]_s{seed}/`; summaries
  `experiment_a_seq_summary_{tag}_s{seed}.json` (one per tag+seed — no more overwrite);
  `main_analyze_window_a_latents.py` auto-discovers all `window_w*` runs.
- **Runs (plan revu pré-lancement, pré-enregistré)** : run principal **`aphys_mono0`**
  (energy_centered, λ_z=1, **λ_mono=0**) sur **3 graines** (42, 7, 123) — le hinge est
  physiquement faux (νD non monotone : 68.8 % de pas montants à ν=0.01) et numériquement
  inerte à λ=0.1 (|Δz| médian 0.0069 en unités standardisées → contribution ≈ 1.5e-5 contre
  une z_loss O(0.01–1)) ; ablation **`aphys`** (λ_mono=0.1) sur 1 graine (42) ; contrôle
  **`asup`** (supervised) sur 1 graine (42), interprété comme sensibilité au bruit FD
  (cibles identiques à 0.35 % près), pas comme test physique ; **`window-A` (`--z-target none`)
  relancé sur les graines 7 et 123** pour une comparaison appariée par graine. w=25 only
  (C2: w=50 ≈ w=25).
- **Références pré-enregistrées** (w=25, ν=0.01 held-out, test) : plafond analytique
  (Spearman 1.0000, R² 0.9999, rel_rmse 0.012) ; **référence principale = C2 supervisé
  c3_window w25 held-out** (même encodeur, même fenêtre, supervision νD : Spearman 0.973,
  R² 0.690, pente 0.90, rel_rmse 0.795, IC bootstrap-95 % [0.922, 0.986] — reproduit depuis le
  checkpoint via `main_export_predictions.py` + `main_bootstrap_a_seq.py`,
  `outputs/logs/bootstrap_references.json`) ; sonde linéaire sur h = 0.655 ; window-A none =
  0.249. **4 issues pré-enregistrées** : (i) aphys ≈ C2 (Spearman ≥ 0.95) → z préserve νD
  aussi bien que la supervision directe ; (ii) entre 0.655 et C2 → préservation partielle ;
  (iii) < 0.655 → L_z sélectionne moins bien que la sonde linéaire sur h (problème de
  sélection/optimisation, pas d'information) ; (iv) > C2 → L_z dépasse la supervision directe
  (à interpréter avec prudence : cibles quasi identiques).
- **Prédictions chiffrées** (w=25, ν=0.01 held-out) : si L_z fonctionne, Spearman(z,νD) ∈
  [0.85, 1.0], R² affine ∈ [0.6, 0.9999]. Falsification : Spearman < 0.655 (sonde h).
- **Sauvegarde des prédictions** : chaque run écrit `predictions_test.npz` (z en unités
  physiques, h, ic_type_id, E, D, max|u_x|, nu, time_t, trajectory_index, time_index_t,
  energy0, nu_dissipation_t, nuD_est_t, split, protocol, seed — **tous modes, y compris
  `none`**) → bootstrap par trajectoire et IC inter-graines via `main_bootstrap_a_seq.py`,
  rapportés **séparément**. Les checkpoints historiques (window-A w25/w50, C2 c3 w25/w50) ont
  été réexportés au même format (`main_export_predictions.py`, env torch).
- **Queue lourde** (mesurée sur la population exacte des paires train heldout, n=30 400,
  t∈[24,99], ν∈{0.05,0.2}) : standardiseur mean=0.169343, std=0.180477 ; kurtosis de z
  standardisé = 8.8, top 1 % = 21.7 % de Σz², top 5 % = 50.3 %. Côté test ν=0.01 : **0 % des
  fenêtres hors plage** (max νD test = 1.4509 < max train = 1.8121), mais 5.2 % des fenêtres
  à |z| > 3 (max +7.05) → la queue vient de la forme de la distribution, pas d'une
  extrapolation hors plage. Condition pré-enregistrée : cible en log(νD) (ou perte de Huber
  sur z standardisé) **seulement si** la pente de calibration à ν=0.01 sort de [0.8, 1.2].
- **Success criteria**: verdict v2 (`verdict_a_seq`): r_holdout ≥ 0.8, R²_affine ≥ 0.5,
  calibration slope ∈ [0.5, 2], beats ν-oracle, structural = rank ≥ 3 + rec RMSE ≤ 0.15
  (monotone not_applicable); IDENTIFIABLE / PARTIAL / NOT IDENTIFIABLE.

Run (torch env):

```
python main_analytic_nuD_baseline.py                       # baseline analytique (sans réseau)
for s in 42 7 123; do
  python main_train_latent_seq.py --tag aphys_mono0 --seed $s --z-target energy_centered --z-weight 1.0 --mono-weight 0
done
python main_train_latent_seq.py --tag aphys --seed 42 --z-target energy_centered --z-weight 1.0 --mono-weight 0.1
python main_train_latent_seq.py --tag asup --seed 42 --z-target supervised --z-weight 1.0 --mono-weight 0.1
for s in 7 123; do
  python main_train_latent_seq.py --tag wina --seed $s --z-target none   # window-A apparié par graine
done
python main_export_predictions.py --which all          # (env torch) réexport des checkpoints -> predictions_test.npz
python main_bootstrap_a_seq.py --glob "outputs/models/latent_dynamics_autoencoder_seq/*/predictions_test.npz" \
    --out outputs/logs/bootstrap_a_seq.json           # IC bootstrap par trajectoire + IC inter-graines
python main_analyze_window_a_latents.py        # inclut les runs _z*_s* automatiquement
```

Read `outputs/logs/experiment_a_seq_summary_aphys_s{seed}.json` → `verdict.window_a`
(`verdict_version: v2_aphysis`, `monotone_check: not_applicable`).

### Step 6 — results (production runs, pré-enregistrement commit `212db0f`)

Runs : `aphys_mono0` (energy_centered, λ_z=1, λ_mono=0) graines 42/7/123 ; `aphys` (λ_mono=0.1) s42 ;
`asup` (supervised) s42 ; `wina` (none) s7/s123 (+ s42 historique). w=25, heldout_nu.
Métriques ν=0.01 held-out (test : 3 800 fenêtres / 50 trajectoires), z en **unités physiques** :

| run | Spearman | R² | pente | rel_rmse | beats ν-oracle | rec RMSE | verdict v2 |
|---|---|---|---|---|---|---|---|
| aphys_mono0 s42 | 0.9655 | 0.859 | 0.962 | 0.537 | ✓ | 0.1775 | PARTIAL |
| aphys_mono0 s7 | 0.9683 | 0.791 | 0.962 | 0.652 | ✓ | 0.1702 | PARTIAL |
| aphys_mono0 s123 | 0.9611 | 0.881 | 0.867 | 0.493 | ✓ | 0.1642 | PARTIAL |
| aphys (λ_mono=0.1) s42 | 0.9655 | 0.859 | 0.926 | 0.537 | ✓ | 0.1663 | PARTIAL |
| asup s42 | 0.9602 | 0.824 | 1.011 | 0.600 | ✓ | 0.1700 | PARTIAL |
| wina none s42/s7/s123 | 0.249 / 0.438 / 0.327 | < 0 | ~0 | 1.4–2.4 | ✗ | 0.1556 / 0.1488 / 0.1484 | NOT IDENTIFIABLE |

- **IC inter-graines** (3 graines appariées) : Spearman 0.9650 ± 0.0036, **IC95 [0.956, 0.974]** ;
  R² 0.844, IC95 [0.728, 0.960] ; pente 0.930, IC95 [0.794, 1.067] → condition log/Huber non
  déclenchée. **IC bootstrap par trajectoire** (B=2000, 50 traj) : Spearman IC95 ≈ [0.89, 0.98]
  par run (`outputs/logs/bootstrap_a_seq.json`). **C2 supervisé (Spearman 0.973, 1 seule graine)
  tombe dans l'IC bootstrap-95 % des 3 runs aphys** (vérifié numériquement) — on ne dit pas
  « Δ dans le bruit » : C2 = 1 graine, aphys = 3 graines.
- **Issue pré-enregistrée (i)** : aphys ≈ C2 sur le rang, R² au-dessus (0.84 vs 0.69). Écart au
  **plafond analytique** : R² 0.84 vs 0.9995 (perte d'optimisation / goulot scalaire).
- **νD vs D dans la strate ν=0.01** : νD = 0.01·D, donc Spearman(z,νD) = Spearman(z,D)
  *exactement* (vérifié : 0.9655 = 0.9655) — le rang ne distingue pas νD de D. La preuve que z
  suit νD et pas D est en unités physiques : **pente z~νD = 0.87–0.96** (si z suivait D, la pente
  serait ~100) et **médian z/D @ν=0.01 = 0.0130–0.0142 ≈ ν** (runs none : 0.003 / 0.030 / −0.024,
  aucune échelle fixe).
- **Contrôles** : λ_mono 0 vs 0.1 → aucun écart (hinge inerte, ~1.5e-5 prédit) ; asup = aphys
  (cibles identiques à 0.35 %). **Coût reconstruction apparié** (artefacts commités, protocole
  standard) : rec RMSE aphys 0.1663/0.1702/0.1642 vs none 0.1557/0.1488/0.1484 → surcoût
  **+0.011 à +0.021** (moy +0.016) — c'est ce coût qui fait échouer le seuil 0.15 (**seuil
  inchangé**), d'où PARTIAL et non IDENTIFIABLE. (La valeur publiée +0.016 à +0.022 utilisait
  0.1775 pour s42, exécution antérieure non archivée — voir note de provenance ci-dessous.)
  Ces rec RMSE sont celles du protocole **standard** (`verdict_a_seq`,
  `ml_train_experiment_a_seq.py:727` `s = standard_results`). Note provenance s42 : le summary
  (0.1775, mtime 2026-11-11 14:30, non commité — seul `experiment_a_seq_summary_aseq.json` est
  suivi) et les artefacts commités (16:40, `97c6357`, rec 0.1663) sont deux exécutions du **même
  commit `212db0f`** (13:57) à la même graine 42 → **non-déterminisme à graine fixée constaté**
  (0.1775 vs 0.1663) ; cause non tranchée (working tree local non commité entre les deux, ou
  source de hasard non seedée — `set_global_seed` fixe random/np/torch/cuda,
  `ml_train_experiment_a_seq.py:94-99`, device CPU par défaut `:829`, num_workers=0).
  Vérification par 2 smoke runs même graine (faits le 2026-11-12, résultats dans Limites
  Step 7 : pipeline déterministe) → cause la plus probable : working tree local non commité
  entre les deux exécutions. Plage sur artefacts commités : 0.164–0.170. s7/s123 : summaries = metrics commités.
  L'analyse A1 exploratoire standard s42 (`main_analyze_a1_intra_trajectory.py:46`) utilise le
  npz commité (16:40).
- **Stratifications depuis `predictions_test.npz`** (sans réentraînement,
  `outputs/logs/aphysis_stratifications.json`) : à ν=0.01, **0 % des fenêtres hors plage**
  (max νD test 1.4509 < max train 1.8121) ; par phase (dD/dt > 0 = raidissement ; **22 % affiché =
  2514/11400, dénominateur poolé sur les 3 strates — fraction intra-strate ν=0.01 = 66.2 %**,
  les rel_rmse « raidissement » portent bien sur les n=2514 fenêtres à ν=0.01) aphys tient dans
  les deux phases (Spearman 0.93–0.96 en raidissement, 0.95–0.97 en
  décroissance ; rel_rmse plus élevé en raidissement : 0.80–1.02 vs 0.31–0.43). **Rang effectif
  de h** (participation ratio sur le test) : none 3.7–5.7 vs aphys **1.6–2.5** — L_z concentre h
  sur ~2 directions (z + 1). Analyse per-dim (`h_dims_window_a_analysis.json`) : plusieurs dims
  portent νD à |sp| 0.74–0.94 (contre ≤ 0.56 diffus en none), sonde h linéaire @ν=0.01 = 0.96–0.97.
- **Canal énergie vs complexe de choc** : *hypothèse non tranchée*. Les dims porteuses de νD
  corrèlent avec E (0.65–0.80) et D (0.6–0.74), mais les **corrélations partielles
  corr(h_j, νD | E, D) à t fixé restent 0.5–0.91** (t=0.5 : h1 +0.91, h4 −0.88, h5 −0.82 ;
  t=1.0 : h1 +0.86, h4 −0.81) → le signal νD dans h n'est pas réductible aux seuls niveaux E, D.
- **Transfert** : « transférable » ne vaut que pour l'**extrapolation ν = 0.01** (held-out) ;
  aucune claim au-delà.
- **Écart au pré-enregistrement (définition du R²)** : le prereg
  (`PREREGISTRATION_experiment_a_physics_v1.md:89`) dit « R²_affine ≥ 0.5 » ; le code calcule
  une R² **identité** (`verdict_a_seq`, `ml_train_experiment_a_seq.py:801,829-833` ←
  `_subset_metrics:594-595`). aphys : identité 0.844, Pearson² in-sample 0.841–0.890
  (`main_analyze_window_a_latents.py:80`) — les deux ≥ 0.5 → `held_ok` inchangé ; le verdict
  PARTIAL est dû au rec RMSE structurel (> 0.15), pas au R². Identité ≤ affine in-sample ⇒
  critère « ≥ seuil » évalué plus strictement.
- **Interprétation** (garde-fou du prereg) : L_z injecte νD via dE/dt = −νD → ces runs montrent
  la **préservation** d'une observable injectée (z scalaire quasi physique, pente ≈ 0.93), pas
  sa découverte. C2, supervisé sur νD, montre l'**apprenabilité** du canal énergie par un CNN —
  pas une découverte non plus. La découverte sans injection reste ouverte (Open questions).

### Step 6 — A1: intra-trajectory dynamics of z (prereg `5623042`, data `97c6357`, verdict `ccb139a`, fix `57c9bb7`)

Question : z est-il un **traceur dynamique** de νD au sein de chaque trajectoire, ou une
**étiquette statique** bien calibrée ? Analyse en lecture seule (`main_analyze_a1_intra_trajectory.py`,
prereg `outputs/prereg/a1_intra_trajectory_prereg.md`), ν=0.01 held-out, w=25, t ∈ [0.48, 1.98],
50 trajectoires test. Règle de décision du prereg §4 (6 conditions) appliquée telle quelle →
**verdict : TRACEUR DYNAMIQUE** au sens des 6 critères pré-enregistrés : z suit la dynamique de
la dissipation au sein des trajectoires, à ν=0.01 held-out, w=25, t ∈ [0.48, 1.98].

- aphys_mono0 (3 graines) : M1_raid = **0.974**, M2 (double centrage, profil horloge LOO retiré)
  = **0.913** (IC95 inter-graines [0.858, 0.969]), M3(k=5) raid = 0.877, |M4| médian = **1 pas
  (0.02 s, limite de résolution)** sur les 29 trajectoires à pic intérieur (19 pics à t=2.0,
  2 avant 0.48), M6 = 0.81, M8 = −0.94, M5 = −2 pas (none : −9). Baseline E_t sans entraînement :
  M1_raid −0.976, diff vs aphys = 1.950, IC95 [1.940, 1.961] (condition 4 du §4, exclut 0).
  Nulls horloge : profil **médian** M2 = 0.009, profil **affine par traj** M2 = 0.529 (le −1 du
  LOO moyen est un artefact de construction).
- none_recal (3 graines, fit affine sur strates test 0.05/0.20 — prédictions train non exportées) :
  per-seed M1_raid = 0.523 / −0.742 / −0.756, **moyenne 3 graines = −0.325 = none brut** → le
  recalibrage affine n'apporte **aucune information dynamique** (α > 0 pour les 3 graines).
- Portée : à ν fixé νD = 0.01·D, donc M1–M5/M7/M8 sont invariants par changement νD→D ; la
  séparation νD/D est portée par **M6** (0.81 en unités νD vs 0.0081 rapporté à D) et la pente
  de calibration (≈ 0.93).
- Amplitude sous-estimée en raidissement (M6 = 0.81 < 1), cohérente avec l'écart au plafond
  analytique (R² 0.84 vs 0.9995).
- Analyse complémentaire (§3, pas un critère du §4) : appariement aphys vs none par graine,
  dM1_raid = +0.46/+1.72/+1.73, moyenne +1.30, **IC95 t ddl=2 = [−0.52, 3.12] non concluante à
  n=3** ; la largeur de l'IC vient de l'hétérogénéité de none (s42 M1_raid +0.52 vs s7/s123 ≈ −0.75).
- Limites : M1_raid teste surtout le signe de la tendance ; 21/50 trajectoires ont leur pic au
  bord de la fenêtre ; none_recal ajusté sur les strates test.
- **Écarts au pré-enregistrement** : le verdict a d'abord été rapporté PARTIEL à cause d'une
  implémentation non conforme au §4 (comparaison appariée vs none substituée à la condition 4
  vs baseline E_t) ; corrigé en `ccb139a` après relecture du texte commité `5623042`, dans le
  sens du texte. Verdict avant : PARTIEL → après : TRACEUR DYNAMIQUE.
- Définition du R² : les critères M1–M8 sont des corrélations de **Pearson** (aucun seuil de
  R²) → verdict inchangé ; `none_recal` et la baseline E_t sont des recalibrations **affine
  ajustées sur le train, évaluées sur le test** (catégorie 3 :
  `main_analyze_a1_intra_trajectory.py:381-395, 546-555`), pas des Pearson² ; l'horloge
  affine par trajectoire est un fit in-sample (Pearson² par traj, `:350-358`).
- Note Step 6 : `frac_dDdt_gt0` = 22.05 % = 2514/11400 (dénominateur poolé sur les 3 strates) ;
  fraction intra-strate ν=0.01 = 2514/3800 = **66.2 %** (ν=0.05 : 39.8 %, ν=0.20 : 0 %).
  Reproduit exactement par `main_reproduce_phase_strat.py` (`outputs/logs/phase_strat_reproduction.json`).

Dataset facts measured on the npz (nu_sweep_v1): Nx=1024, 101 snapshots, dt_saved=0.02
(T=2.0), max|mass0| = 3.49e-15 (zero_mean applied to ALL IC families, incl. gaussian_bumps),
max|u| = 1.79. Note: this differs from the legacy spec (Nx=128, solver dt=1e-4,
save_every=200) — the nu_sweep_v1 npz is the ground truth.

## Step 7 — B1: held-out INTERPOLATION (ν = 0.05) — prereg `f431e66` + amendment `fc1075b`, artefacts `a6cb108`

### Protocole
`heldout_nu005` : train/val ν ∈ {0.01, 0.20} (`main_train_b1.py:33,76`), test complet, w=25,
6 runs (aphys_mono0 = energy_centered λ_z=1 λ_mono=0, et none ; graines 42/7/123), tous sous
Windows (logs UTF-16, chemins `C:\`), early stopping 30–57 époques, aucun NaN. Évaluation :
`python3 main_evaluate_b1.py` (code `5e6f24e`), `test_egalite=True`, verdict PARTIAL.
SHA-256 des 6 `predictions_test.npz` + metrics + logs consignés dans le commit `a6cb108`.

### Verdict (règle fc1075b appliquée mot pour mot, bras aphys_mono0, moyenne 3 graines, IC95 t ddl=2)
C1 Spearman 0.985 [0.968, 1.002] ≥ 0.8 ✅ (note : la borne haute 1.002 dépasse 1, borne de
Spearman — artefact de l'IC t sur n=3 ; la moyenne 0.985 est la valeur de référence) ;
C2 R² 0.927 [0.883, 0.971] ≥ 0.5 ✅ (comparable à causal_w25 0.917, dont l'IC contient 0.917 —
pas de battement significatif) ; C3 pente 1.153 [1.069, 1.238] ∈ [0.5, 2] ✅ (prédiction
prereg [0.8, 1.1] formulée en plage ponctuelle, sans IC → **falsifiée** en ponctuel) ;
C4 rel_rmse 0.317 [0.219, 0.415] < ν-oracle 1.1803 ✅ (oracle = strate ν=0.05 du TRAIN COMPLET,
README:342) ; C5 rang effectif 5.667 ≥ 3 ✅ ; **C6 rec RMSE 0.196 [0.177, 0.215] > 0.15 ❌ (seul
échec)** ; C7 médian z/D 0.0470 [0.0458, 0.0482] ∈ [0.025, 0.10] ✅ ; C8 bat E_t (0.766/0.477)
avec IC excluant 0 des deux côtés ✅ (ΔSpearman 0.219 [0.202, 0.236], ΔR² 0.450 [0.406, 0.494]) ;
F1 non déclenché. **Verdict : PARTIAL.** none : NOT IDENTIFIABLE (Spearman −0.290/−0.046/−0.420,
moy −0.245 ; R² calibré Pearson² 0.011–0.250 ; R² brut −12 à −29 = z non calibré).
Appariement aphys−none (COMPLÉMENTAIRE, hors verdict) : ΔSpearman +1.28/+1.03/+1.40 par graine.

### Analyse C6
C6 échoue dans les deux cas (Step 6 : run standard ; B1 : run held-out) et aussi sans L_z ;
surcoût de L_z +0.020 [−0.014, +0.055], non significatif à n=3 (rec RMSE appariée aphys−none :
+0.036/+0.008/+0.017 ; none 0.169–0.181 > 0.15 aussi). Définition identique partout : `reconstruction_rmse_test = sqrt(mean((rec−u)²))`,
test complet, états standardisés (`ml_train_experiment_a_seq.py:960`). Le verdict Step 6 lit le
run **standard** (`verdict_a_seq`, `:727` `s = standard_results`) : summaries 0.1775/0.1702/0.1642
(s7/s123 = metrics standard exacts ; s42 : summary 0.1775 vs metrics standard 0.1663, exécution
antérieure) — les 0.193–0.206 sont les metrics du run **heldout_nu**, runs différents, pas
d'incohérence. Plancher PCA linéaire de nu_sweep_v1 (population = test complet 150×101 snapshots,
standardisation = GlobalStandardizer fit sur train complet, `ml_datasets.py:154-165`, unités =
états standardisés, identiques à la rec RMSE) : rang 4 → 0.359, 5 → 0.298, 6 → 0.225, 8 → 0.188,
16 → 0.103. **Au rang effectif 5–6 (plancher 0.225–0.298), aphys B1 (0.190–0.205) bat le
plancher ; à k = latent_dim = 8 (plancher 0.188), aphys B1 (moy 0.196) ne le bat pas, none
(0.169–0.181) oui.** Le seuil 0.15 a été calibré sur le dataset legacy Nx=128 (README:76,167,204 :
rec RMSE 0.0874 PASS sur legacy) et est inatteignable en linéaire sur nu_sweep_v1 (Nx=1024,
README:261) en dessous du rang ~16.

### Exploratoire (étiqueté, non seuillé)
A1 à ν=0.05 (50 traj ; pics νD sur [0,100] : 19 intérieurs [24,99], 17 avec ≥5 pas avant le pic,
9 à t=2.0, 22 avant 0.48 → teste surtout la décroissance), **définition principale du prereg
`5623042` (raidissement = t < t_pic, t_pic = argmax global [0,100])** : M1_all médian
0.970–0.980 ; M1_raid 0.982–0.989 (**n=26** traj avec ≥5 pas en raidissement, ≤ 28 attendu car
les 22 pics < 0.48 n'ont aucun pas de raidissement dans la fenêtre) ; M2 0.927–0.939 ;
**M4 médian +6/+7/+3 pas (n=19 pics intérieurs)** = retard systématique 0.06–0.14 s sur les
3 graines ; M6_raid 1.25–1.82 (n=26) mais **M6 trajectoire complète 0.97–1.21** et
corr(M6_raid, nb pas raid) = −0.41 à −0.50 ; M8 −0.15 à −0.27. **Hypothèse** (non testée au-delà
de ces contrôles) : la pente 1.15 (>1) = z sous-dispersé globalement (variance inter-trajectoire
compressée) tandis que M6_raid > 1 = sur-dispersion locale en raidissement — M6_full ≈ 1 et la
corrélation négative M6_raid/npas suggèrent un artefact de normalisation plutôt qu'une
amplification physique. Audit M4 à ν=0.01 (Step 6) : t_pic = argmax global [0,100]
(`main_analyze_a1_intra_trajectory.py:168`) et pour les 29 traj à pic intérieur argmax fenêtré
== t_pic global (vérifié numériquement) → M4 conforme, verdict `5623042` inchangé.
Interpolation vs extrapolation (mêmes graines, apparié) :
ΔSpearman B1−heldout_nu +0.020 [+0.010, +0.030], ΔR² +0.083 [−0.012, +0.178], Δrel_rmse −0.244
[−0.413, −0.074], Δrec RMSE −0.002 [−0.036, +0.032]. Strates 0.01/0.20 in-distribution en B1.

### Écarts au pré-enregistrement
1. Nommage des runs `w25_...` (code) vs `window_w25_...` (prereg) → résolu côté évaluateur
   (`2fa9b98`), prereg intact. 2. NaN → exit 1 ajouté à l'évaluateur (`2fa9b98`) avant
   l'évaluation de production. 3. `calibration_slope` inversée (pente z~y au lieu de
   νD~a·z+b, définition Step 6 `fit_affine_calibration`) → corrigée en `166c1ff`. 4. C5 :
   texte fc1075b mot pour mot `{"id": "C5_rang_effectif", "metric": "effective_rank_h", "op": ">=",
   "threshold": 3, "statut": "CONFIRMATOIRE"}` ; définition opérationnelle héritée du Step 6 =
   rang 95 % (`latent_pca_stats`, `ml_models.py:574-599`). L'évaluateur calculait d'abord un
   participation ratio → corrigé en `5e6f24e` (rang 95 % = 5.667). **Cette correction est
   favorable au résultat** : avec le participation ratio, s42 = 2.757 < 3 aurait fait échouer C5 ;
   le rang 95 % (5/6/6, moy 5.667) la fait passer. C5 passait **aussi** sous le participation
   ratio (moyenne 3.37 ≥ 3, seuil 3) — la correction ne change donc pas le résultat du critère,
   seule la valeur rapportée (3.37 → 5.667). Les trois corrections ont été faites après une
   première lecture des résultats, dans le sens du texte commité, contre l'intérêt apparent du
   résultat ; le verdict est inchangé (PARTIAL, C6 seul échec) dans les trois versions. Sorties
   archivées : `outputs/logs/b1_eval_archive/` (pre-`166c1ff`, pre-`5e6f24e`, HORODATAGES.md).
   pip freeze Windows non consignable depuis le sandbox : à produire côté env VS Code
   (`pip freeze > outputs/logs/b1_eval_archive/pip_freeze_windows.txt`).
  5. Définition du R² : C2 (« R² affine ≥ 0.5 », `b1_interpolation_prereg.md:61`,
  `b1_interpolation_prereg.json:147-148`) est calculé en **identité**
  (`main_evaluate_b1.py:86-87`, label `:94`) : 0.927 ≥ 0.5 ; Pearson² ≥ identité in-sample ⇒
  C2 passe sous les deux définitions → verdict inchangé. La baseline `E_t_affine`
  (a=0.354, b=0.0219, OLS sur train filtré, 0.766 @ν=0.05 test) est une catégorie 3
  (affine train→test).

### Limites
C6 échoue avec IC [0.177, 0.215] excluant 0.15 — structurel (même constat au Step 6, tous
protocoles, y compris none). Non-déterminisme à graine fixée constaté au Step 6 (run standard
s42 : 0.1775 vs 0.1663, même commit `212db0f`, même graine — voir note de provenance Step 6) :
impact sur l'appariement par graine (la variance inter-graines inclut cette composante) ;
**Vérification déterminisme** (2026-11-12) : 2 smoke runs `python3 main_train_b1.py
--z-target-mode none --seed 42 --smoke` → logs **identiques hors chemins /tmp**
(`outputs/logs/determinism_smoke_run{1,2}.log` : train_loss 9.620083e-01, val_loss 9.639770e-01,
rec RMSE 9.4553e-01, rang 5.0, monotone 0.5770) → pipeline déterministe pour ce test ; la
différence s42 (0.1775 vs 0.1663) n'est donc PAS expliquée par le pipeline → hypothèse retenue :
working tree local modifié (non committé) ou args différents entre les exécutions de 14:30 et
16:35. Environnement Windows consigné : `outputs/logs/b1_eval_archive/pip_freeze_windows.txt`
(torch 2.12.0+cpu, numpy 2.4.1). C5 : deux définitions ont coexisté (PR vs rang 95 %), corrigé ;
les deux ≥ 3 sauf PR s42. A1 à ν=0.05 exploratoire (n=19/26), M4 retard systématique non seuillé.
Appariement vs none à n=3, complémentaire. Environnement homogène (6/6 Windows, versions
consignées dans pip_freeze_windows.txt). ν-oracle Step 6 : écart doc/code (moyenne globale du
train filtré 1.4287638073 utilisée vs train complet 1.4287588915 annoncé README:342, écart
4.9e-6, impact nul sur beats_nu_oracle) ; corrigé à partir de B1 (`train_dataset_full`).

## Open questions

- ~~Does z track dissipation or the whole gradient/shock complex?~~ **Answered (Architecture A +
  nu_sweep_v1 audit): the old dissipation-like z was collapse-induced; on nu_sweep_v1 the νD/D
  degeneracy is broken globally (Spearman 0.642 vs 1.0 legacy) but NOT within a ν stratum, and
  D / max|u_x| / TV remain one shock complex (0.95–0.99).**
- Is νD separable from D by a *trained model* at held-out ν? → **Answered (Experiment C +
  C2): NO at w=5 (all variants NOT IDENTIFIABLE; νD head ≈ c·D head, c ≈ 0.1), but YES largely
  once the temporal window is long enough — C2: w=25/50 give R² νD = +0.69 and Spearman 0.97
  at held-out ν=0.01, beating the ν-oracle, via the ν-agnostic channel dE/dt = −νD.**
- Can a *latent* z (Architecture A loss, **no νD supervision**) align with νD at held-out ν?
  → **Answered NO** (window-A, Step 5 + Step 6 `wina`, 3 graines s42/s7/s123 : Spearman @ν=0.01
  = 0.249 / 0.438 / 0.327, NOT IDENTIFIABLE) — the νD info is in h (linear probe 0.53–0.66) but
  loss A does not select it.
- Can an **injected** νD observable be *preserved* in the scalar z at held-out ν? → **Answered
  YES** (A-Physics, Step 6 : L_z = MSE(z_t, νD_est), 3 graines — Spearman 0.965 ± 0.004,
  IC95 [0.956, 0.974], pente physique 0.87–0.96, médian z/D ≈ 0.013 ≈ ν, beats ν-oracle) —
  and **A1 (Step 6)**: preserved in rank, calibration AND intra-trajectory dynamics
  (verdict TRACEUR DYNAMIQUE, 6/6 prereg criteria, prereg `5623042`) — and **in interpolation**
  (B1, Step 7 : ν=0.05 held-out, Spearman 0.985, beats ν-oracle and E_t, médian z/D 0.047 ≈ ν,
  verdict PARTIAL — C6 rec RMSE seul échec, échoue aussi sans L_z) — preservation of an
  injected observable, not discovery.
- Can νD be **discovered without injection** (an unsupervised objective selecting the
  energy-slope channel on its own)? → **Open.**
- Why do the encoders stay below the 2-snapshot estimator (R² 0.84 vs 0.9995)? → **Open**
  (consistent with A1: amplitude under-estimated during sharpening, M6 = 0.81 < 1).
- Which loss term creates the structured z — predictor or monotonicity hinge? (A0 ablation, not run)
- ~~Do residual h dims encode shock position / Fourier content once collapse is fixed?~~ **Partially:
  with the collapse fixed, all h dims are static IC labels (shock_location, Fourier amps, spectral
  centroid, mass); no dynamic dim emerged at fixed ν. Note: on nu_sweep_v1 mass0 ≈ 0 everywhere
  (zero_mean for all families), so the mass label is gone.**
- Does the alignment persist at larger T? (nu_sweep_v1 covers ν ∈ {0.01, 0.05, 0.20};
  larger T not covered)


## Step 8 — TD (encodeur) : bug de filtre `heldout_nu`, correction et archivage

- **Bug** : `main_train_td.py:48` passait `heldout_nu_values=(0.01,)` au lieu de
  `(0.05, 0.2)` (prereg `td_encoder_prereg.md:40`) → 10 runs TD0 heldout_nu entraînés
  sur ν=0.01 seul (preuve : `z_target_standardizer` = 0.168410/0.234691 = fit ν=0.01,
  reproduit à 8 décimales). Corrigé dans `36ba701` (module `td_protocols.py`, source
  unique ; garde-fou `expected_train_nu` ; `python3 td_protocols.py --check`), archivage
  `d844a7a` (`td0_indist_nu001_*`, contrôle in-distribution, hors verdict), runs corrigés
  relancés sous les noms d'origine (`4f3b713`, artefacts `1a93648`).
- **Aucune affirmation d'effet du code déterministe sur la performance** : différences
  par graine (non appariées au sens strict — le générateur du DataLoader a changé, donc
  même graine ≠ même trajectoire), R² identité des deux côtés, depuis les
  `predictions_test.npz` : TD0−Step6 (ν=0.01) moy +0.0292, IC95 t(ddl=2) [−0.0406, +0.0989] ;
  B1−TD0 (ν=0.05) moy +0.0078, IC95 [−0.0209, +0.0365]. Toutes les IC incluent 0 →
  aucun effet démontré, effet faible non exclu ; runs historiques et runs TD ne sont
  jamais comparés entre eux. La limite « résultats historiques non reproductibles
  bit à bit » reste.

## Step 9 — Résultats TD (encodeur) — prereg `515cf5b` + amendment `182c8fc`, artefacts `91199bc`

### Protocole
- 80 runs = 4 bras (td0/td1/td2/td3) × 2 protocoles (heldout_nu, heldout_nu005) × 2 modes
  (energy_centered, none) × 5 graines {42, 7, 123, 2024, 999}. Les 60 runs TD1–TD3 ont
  tourné sur git=`182c8fc8`, working tree propre (dirty=False), torch 2.12.0+cpu /
  numpy 2.4.1 / 4 threads ; SHA-256 des 300 fichiers dans
  `outputs/logs/td_corrected/td_artifacts_sha256.txt` (commit `91199bc`). Évaluateur :
  `main_evaluate_td.py` → `outputs/logs/td_encoder_results.json`.
- R² d'évaluation = identité `1−Σ(z−y)²/Σ(y−ȳ)²` à ν held-out (définition fixée par
  l'amendement §0 ; voir « Inventaire des définitions de R² » ci-dessous).

### Verdicts (règle machine du prereg §6, rapportés tels quels)
| règle | mean(d) | IC unilatéral 95 % (t ddl=4) | verdict |
|---|---|---|---|
| D1 td2−td0 (CONFIRMATOIRE, principale) | −0.0034 | [−0.0606, +0.0537] | **INDÉTERMINÉ** |
| D2 td1−td0 (CONFIRMATOIRE) | +0.0310 | [+0.0034, +0.0585] | **INDÉTERMINÉ** |
| D3 td3−td2 (EXPLORATOIRE, injection) | +0.1060 | [+0.0433, +0.1687] | REPRÉSENTATIONNEL |

- **D1** : compatible avec un effet nul (mean −0.003), n'exclut pas jusqu'à +0.054 (borne
  sup unil. +0.0537) ; même sous effet nul, le verdict NON-REPR n'était obtenu qu'avec une
  probabilité ≈ 0.46 ; P(REPR | Δ=δ_R) ≈ 0.38 (post-hoc, sd observé 0.0599, diagnostic de
  conception).
- **D3** : mesure l'injection sur la branche z **plus** un effet indirect sur la
  reconstruction (rec td3 = 0.62 < rec td2 = 0.75) ; EXPLORATOIRE, étiqueté INJECTION.

C6' (29c7c3e, heldout_nu) : C6'a ÉCHEC td0 (1.273), td1 (1.157), td2 (1.139) ; td3 « OK »
(0.986) sans objet car rec collapsé. C6'b ÉCHEC pour les 4 bras (rec_z > 0.188).
nu005 : C6'a OK pour td1/td2/td3 ; C6'b OK seulement pour td1 (0.1830 ≤ 0.188).

### Résultats par bras (mode z, R² identité, moy ± sd, n=5)
ν=0.01 : R² = td0 0.8747±0.0201 · td1 0.9057±0.0241 · td2 0.8713±0.0614 · td3 0.9773±0.0053 ;
Spearman 0.9645/0.9839/0.9840/0.9872 ; pente 1.009/0.929/0.844/1.017 ; M6 0.743/0.911/
1.028/0.933 ; rec 0.197/0.190/0.746/0.623.
ν=0.05 : R² = 0.9238/0.9476/0.9765/0.9883 ; pente 1.164/1.123/1.050/1.027 ; rec 0.190/0.183/0.574/0.556.

### Bilan des prédictions pré-enregistrées (prereg §8, toutes)
- TD0 : R² ∈ [0.78, 0.90] → 0.8747±0.0201 : **CONFIRMÉ**. Spearman ∈ [0.95, 0.98] → 0.9645 :
  **CONFIRMÉ**. Falsification (TD0 hors plages → STOP) : **non déclenchée**.
- TD2 si REPR : R² ∈ [0.90, 0.9995] ; si NON-REPR : ∈ [0.78, 0.90] → 0.8713±0.0614 ∈ plage
  NON-REPR mais D1 = INDÉTERMINÉ (borne sup +0.0537 > δ_R). Falsification croisée
  (R²(TD2) < R²(TD0)−0.05) : −0.0034 > −0.05, **non déclenchée**.
- TD3 (injection, EXPLORATOIRE) : R² ≥ 0.99 attendu → 0.9773±0.0053 : **FALSIFIÉ de 0.013**.
- M6(TD0) ≈ 0.81 attendu (EXPLORATOIRE) → 0.7427 mesuré.
- TD1/D2 : **aucune prédiction chiffrée** dans §8 — seules les règles de décision §6
  (D2 = TD1−TD0 CONFIRMATOIRE, δ_R=0.05 ; D1 = TD2−TD0) ; bilan : D2 INDÉTERMINÉ.

### Écart au plafond analytique
Écart au plafond (0.9995) : TD0 0.1248 · TD1 0.0938 · TD2 0.1282 · TD3 0.0222. Les biais
génériques testés (canaux de différences temporelles, pooling global) ne ferment pas
l'écart ; l'injection de ∫u·∂ₜu (∝ dE/dt = −νD) le ramène à 0.022, presque entièrement.
Aucune comparaison confirmatoire n'atteint δ_R = 0.05 : TD1 donne un gain cohérent sur
5/5 graines (+0.0310, borne inf. unil. +0.0034), inférieur au seuil pré-enregistré →
descriptif, hors verdict.

### Inventaire des définitions de R² — trois catégories (relecture complète du code)
1. **Identité** `1−ss_res/ss_tot` sur prédictions brutes (z ou prédiction du modèle, sans
   ajustement) : évaluations Step 5 / Step 6 (`ml_train_experiment_a_seq.py:594-595`, lu par
   `verdict_a_seq:801` ; plafond analytique `main_analytic_nuD_baseline.py:157` →
   `mt.evaluate_predictions`), B1 (`main_evaluate_b1.py:86-87`, label `:94`), TD
   (`main_evaluate_td.py:88-89`, label `:94`), C2 supervisé (`physics_candidates.py:97` via
   `ml_train_experiment_c.py:421`).
2. **Affine in-sample = Pearson²** (fit univarié y ~ a·x + b sur le **même** échantillon
   évalué ; métriques sur valeurs calibrées) : `fit_affine_calibration`
   (`physics_candidates.py:106-119`), per-dim h/z vs νD (`_affine_metrics`,
   `main_analyze_window_a_latents.py:72-84`, r²aff 0.890/0.841/0.889), classements par
   dimension (`physics_candidates.py:336+`, `main_identify_latent_variable.py:368`),
   horloge affine par trajectoire A1 (fit in-sample, `main_analyze_a1_intra_trajectory.py:350-358`) ;
   README:107-108, 221, 632 (« calibrated R² = Pearson² 0.09 »), 928 (B1 calibré 0.011–0.250).
3. **Affine ajusté sur le TRAIN, évalué sur le TEST** (≠ Pearson², peut être négatif ;
   la métrique est une identité sur les prédictions du fit) : sondes linéaires h/z
   (`main_analyze_window_a_latents.py:145-153` + `_plain_metrics:86-95` →
   `outputs/logs/h_linear_probe_window_a.json`, 0.55–0.78 @ν=0.01), baseline `E_t_affine` de
   B1 (OLS train filtré, 0.766 @ν=0.05 test, `b1_interpolation_prereg.md:39`), A1
   `none_recal` et baseline E_t (`main_analyze_a1_intra_trajectory.py:381-395, 546-555`) ;
   README:882.

Endroits du README où deux définitions coexistent : ligne 569 (seuil Step 6 libellé
« affine R² ≥ 0.5 (C2: 0.69) » mais calculé en identité des deux côtés — label erroné),
ligne 632 (« calibrated R² = Pearson² 0.09 » vs R² brut identité), ligne 928 (B1 :
calibré Pearson² 0.011–0.250 vs brut −12 à −29), et la comparaison Step 9 : 0.844
(identité, évaluation) vs r²aff ≈ 0.89 (Pearson², analyse de latents) — deux définitions,
non comparables directement. Le label de clé `r2_affine_nuD_z` est erroné (la valeur est
identité) ; une clé `r2_identity_nuD_z` en doublon a été ajoutée dans le code
d'évaluation (commit séparé), l'ancienne clé est conservée.

### Limites
- Puissances **post-hoc, sd observé, diagnostic de conception** (aucune puissance
  déclarée a priori n'est modifiée) : verdict REPR complet (mean ≥ δ_R ET borne inf. > 0),
  Monte-Carlo exact (n=5, t_c=2.1318, sd échantillon) : à Δ=δ_R=0.05, P(REPR) = 0.50 (D2,
  sd 0.0289) et 0.38 (D1, sd 0.0599) — **≤ 0.5, tend vers 0.5 quand n augmente** ; MDE80 % =
  0.062 (D2), 0.082 (D1, contrainte active = borne inf. car t_c·se = 0.057 > δ_R).
  P(NON-REPR|Δ=0) = 0.93 (D2), 0.46 (D1) — valeur exacte (sd échantillon) ; par symétrie
  P(NON-REPR|Δ=0) = P(borne inf. > 0 | Δ=δ_R), ce qui explique que 0.46 apparaissait déjà
  en v1 comme « puissance à δ=0.05 » (coïncidence de symétrie, pas d'inversion ; la v1
  0.40 utilisait sd fixe, approximation).
- Les valeurs de l'amendement §3 (« MDE 0.039 / puissance 0.93–0.94 à δ=0.05 ») correspondent
  à la seule condition borne inf. > 0 — voir « Écarts au pré-enregistrement ».
- D1 est confondu : le pooling global sur tout l'encodeur rend la reconstruction impossible
  (rec 0.55–0.75, y compris en mode none ; test_losses : rec = 0.45/0.49 de la loss totale
  pour td2 vs 0.037/0.110 pour td0), change l'équilibre des pertes et l'arrêt précoce
  (best_val 0.206 vs 0.040, époque 36 vs 21) → explique probablement la variance de TD2
  (sd R² = 0.061). Défaut de conception du prereg : le pooling aurait dû être limité à la
  branche z.
- La strate test ν=0.01 a été consultée de nombreuses fois depuis le Step 5 (hors
  anti-sélection prereg §4 : aucune itération sur le test, mais lectures multiples).
- Run `td0_heldout_nu005_zenergy_centered_s42` : durée murale anormale (964 min vs ~17 min) ;
  résultats validés bit à bit (`main_validate_run.py`, écart 0.0).
- Exploratoire, sans verdict : hétérogénéité du pooling selon le protocole — TD2−TD0
  apparié par graine : ν=0.01 mean −0.0034 IC95 [−0.0778, +0.0710] ; ν=0.05 mean +0.0527
  IC95 [+0.0386, +0.0668], 5/5 graines positives.

### Écarts au pré-enregistrement
- Bug de filtre `heldout_nu` inversé (`main_train_td.py:48`), prouvé à 8 décimales,
  corrigé (`36ba701`, `0ff901a`, `4f3b713`), 10 runs archivés (`d844a7a`), runs corrigés
  relancés (`1a93648`).
- Amendement 1 rédigé **après lecture** des runs erronés (déclenchement §8) — déclaré.
- Définition du R² d'évaluation fixée par l'amendement §0 (identité) ; label de clé erroné
  documenté ci-dessus ; aucune valeur rétrospective recalculée.
- **Puissance mal déclarée** dans l'amendement §3 : « MDE 0.039 / puissance 0.93–0.94 à
  δ=0.05 » = condition borne inf. > 0 seule. Verdict REPR complet : P(REPR|Δ=δ_R) ≤ 0.5
  (0.50 D2 / 0.38 D1), MDE80 % = 0.062 (D2) / 0.082 (D1). **Aucun verdict modifié.**
- **Métrique primaire changée par l'amendement `182c8fc` (après lecture de TD0 seul,
  avant TD1–TD3)** : le prereg `515cf5b` dit « R² affine νD~z » (md:68) et
  `metric='r2_affine_nuD_z'` pour D1/D2/D3 (json) ; le code calculait une R² identité
  partout (`ml_train_experiment_a_seq.py:593-595`, `main_evaluate_td.py:88`,
  `main_evaluate_b1.py:86`). L'amendement §0 a fixé la **R² identité** pour tous les
  verdicts. **Selon le texte original (R² affine), D1 serait REPR de justesse.**
  Sensibilité recalculée depuis les `predictions_test.npz` (appariement strict par
  graine, IC unil. 95 % t ddl=4, δ_R=0.05) :
  | variante | D1 td2−td0 | D2 td1−td0 | D3 td3−td2 |
  |---|---|---|---|
  | R² identité (publiée ; définition amendée `182c8fc`) | −0.0034 [−0.0606,+0.0537] INDÉTERMINÉ | +0.0310 [+0.0034,+0.0585] INDÉTERMINÉ | +0.1060 [+0.0433,+0.1687] REPRÉSENTATIONNEL |
  | Pearson² in-sample (réajustée sur les labels de la strate évaluée, optimiste) | +0.0503 [+0.0366,+0.0641] REPRÉSENTATIONNEL de justesse (marge +0.0003 sur δ_R ; 5/5 graines > 0) | +0.0443 [+0.0282,+0.0604] INDÉTERMINÉ | +0.0447 (< δ_R) INDÉTERMINÉ |
  | calibration hors strate (proxy : strates test 0.05/0.2, prédictions val non exportées, `ml_train_experiment_a_seq.py:1182-1183`), évaluée à ν=0.01 | +0.0092 [−0.0476,+0.0660] INDÉTERMINÉ | +0.0399 [+0.0112,+0.0687] INDÉTERMINÉ | +0.1012 [+0.0386,+0.1638] REPRÉSENTATIONNEL |

  **D1 INDÉTERMINÉ sous 2 variantes sur 3, D3 REPR sous 2 sur 3 ; seule la variante
  réajustée sur les labels de la strate évaluée (Pearson² in-sample, optimiste) inverse
  les verdicts.** Interprétation : le pooling de td2 améliore la décodabilité linéaire
  (sonde h @ν=0.01 : 0.882 vs 0.861 pour td0 ; Pearson²(z) 0.935 vs 0.885) mais dégrade la
  calibration (pente 0.844, gap Pearson²−identité +0.064, le plus grand des 4 bras) —
  c'est cette recalibration in-sample qui fait basculer D1. Le gain TD3 persiste sous
  calibration hors strate (+0.1012, REPRÉSENTATIONNEL) et est ramené à +0.0447 < δ_R sous
  Pearson² in-sample : sous recalibration sur la strate évaluée, il est surtout de la
  calibration. **Verdicts publiés inchangés** (définition amendée = identité).
- Erratum : l'amendement commité dans `182c8fc` porte encore l'en-tête « BROUILLON POUR
  REVUE, NE PAS COMMITER » — anomalie de contenu signalée sans modification du fichier
  commité ; erratum séparé : `outputs/prereg/td_encoder_prereg_amendment_1_erratum.md`.
- Run 964 min (voir Limites).

### Open questions
- Pourquoi TD3 plafonne à 0.977 et non ≥ 0.99 (prereg falsifié de 0.013) ?
- Le gain TD1 (+0.031, 5/5 graines) confirmé à n plus grand (MDE80 0.062 à n=5) ?
- Un pooling restreint à la branche z (TD2') lève-t-il la confusion rec/z de TD2 ?

---

## Bilan global du projet (5 points)

1. **Données** : nu_sweep_v1 (L=2π, Nx=1024, RK4, T=2, ν∈{0.01,0.05,0.20}, 600/150/150)
   avec identité exacte dE/dt = −νD ; plafond analytique 2 snapshots R² 0.9995 à ν=0.01.
2. **Découverte vs préservation** : sans supervision z ne s'aligne pas sur νD (window-A) ;
   une observable νD injectée est préservée (Step 6 A1, B1) — préservation, pas découverte.
3. **TD** : aucun biais générique ne ferme l'écart au plafond (D1/D2 INDÉTERMINÉS ; TD1
   +0.031 < δ_R, 5/5 graines ; puissances post-hoc à Δ=δ_R ≤ 0.5, MDE80 0.06–0.08 à n=5) ;
   l'injection de ∫u·∂ₜu le ferme presque (TD3 0.977, résidu 0.022) ; le pooling global
   détruit la reconstruction (défaut de conception du prereg).
4. **Méthodologie** : bug de filtre prouvé à 8 décimales et corrigé, runs erronés archivés ;
   80 runs tracés (git, dirty, versions, horaires), déterminisme bit à bit validé
   inter-session ; aucun effet du code déterministe démontré (IC à 0) ; inventaire des
   définitions de R² en 3 catégories (évaluation = identité ; analyses de latents = Pearson²
   affine in-sample ; sondes = affine train→test) ; règle « le contenu d'un commit se vérifie
   par `git show --stat`, jamais par son message » (anomalie `182c8fc`, impact nul).
5. **C6'** : la perte z a un coût en reconstruction — C6'a/C6'b échouent pour TD0/TD1 en
   heldout_nu (rec_z > rec_none apparié, rec_z > plancher PCA k=8 = 0.188) ; seul TD1 passe
   C6'b en nu005 (0.1830).

## Proposition TD2' révisée (NON pré-enregistrée, NON implémentée)

- **Architecture** : tronc partagé = encodeur TD1 (fenêtre + canaux Δ_k u, k∈{1,2,4,8,16}) ;
  deux têtes — flatten spatial → décodeur/prédiction latente ; global average pooling →
  **seule** la tête z (z reste une lecture du tronc partagé, pas un encodeur séparé type C2).
- **Comparaisons** : D1'' = TD2'−TD1 isole le pooling (branche z seule) ; D2'' = TD2'−TD0 =
  effet total. Règle identique (R² identité ν=0.01, appariement strict par graine, IC
  unilatéral 95 % t ddl=4, δ_R=0.05, 3 verdicts).
- **Statut** : conception déclarée **post-lecture** → EXPLORATOIRE si lancée sans nouveau
  pré-enregistrement ; prereg commité AVANT les runs pour du CONFIRMATOIRE. Note : la règle
  « mean ≥ δ_R » plafonne P(REPR|Δ=δ_R) à 0.5 quel que soit n — à remplacer a priori si
  pré-enregistré.
- **Puissance (post-hoc, sd supposé 0.029, diagnostic de conception)** : MDE80 % ≈ 0.062 à
  n=5 ; ≈ 0.058 à n=10 (t(0.95,9)=1.8331, t(0.80,9)=0.8834, se=0.00917).
- **Budget** : n=5 → 20 runs ≈ 9–10 h CPU ; n=10 → 40 runs ≈ 18–20 h. Ordre : heldout_nu
  d'abord ; TD0/TD1 servent de référence (aucun rerun).

---

## Inventaire de reproductibilité (v1.0-preservation)

### Données (lecture seule)
- Régénération documentée (**ne jamais lancer `main_generate_data.py` sans feu vert
  explicite** — les npz de `outputs/data/` sont la vérité terrain, chargés avec
  `allow_pickle=True`, jamais modifiés) : `python3 main_generate_data.py`
  (builder `pde_dataset_builder.py`).
- Vérification d'intégrité (PowerShell) :
  `Get-FileHash outputs\data\train_data_nu_sweep.npz -Algorithm SHA256` (idem val/test).
  SHA-256 constatés :
  | fichier | SHA-256 |
  |---|---|
  | outputs/data/train_data_nu_sweep.npz | dab467cfbb3daccba6b355b353adb07225dbc56004aeee0018483a2ff494768f |
  | outputs/data/val_data_nu_sweep.npz | 07b67e8b6ca9d84d7efbe792eaa03bba8ae1204fcab40c8eb5a24d417e2edf51 |
  | outputs/data/test_data_nu_sweep.npz | e501f91f5bb73258eb63957bb7f54a091b1a4dd0a0db53a8469283cd881060bc |
  | outputs/data/train_data.npz (legacy) | db2db07a98bfc13af98fc83627d837a7c16eb276083cee967a4c2355622e8e71 |
  | outputs/data/val_data.npz (legacy) | 86e55d74b81e84cbb928dd0ed85e448480e679a52fb90d9926dc48570607fbef |
  | outputs/data/test_data.npz (legacy) | 9454dc8ba035b50d4bcdda335e13c3be6a1694d42621e1ce5846c2fe4911b8f3 |

### Environnement
- `requirements-lock.txt` (racine, **UTF-8**) : freeze exact — torch==2.12.0+cpu
  (`torch.__version__` ; le freeze pip brut affiche `torch==2.12.0`), numpy==2.4.1,
  pandas==2.3.3, scipy==1.17.1, scikit-learn==1.8.0. Freeze brut historique (UTF-16) :
  `outputs/logs/b1_eval_archive/pip_freeze_windows.txt`.
- 4 threads ; déterminisme bit à bit à code+graine fixés, validé inter-session :
  `outputs/logs/td_corrected/td_determinism_check_postfix.log` — **commité dans `1a93648`**
  (vérifié : `git log --oneline -1 -- outputs/logs/td_corrected/td_determinism_check_postfix.log`)
  ; écart max 0.0 sur 405 métriques + 20 tableaux npz, exit 0.

### Encodage des logs
- Les logs écrits par `Tee-Object` (PowerShell) sont en **UTF-16** : lecture Linux
  `iconv -f UTF-16LE -t UTF-8 <log>` ; PowerShell `Get-Content -Encoding Unicode <log>`.

### Commandes canoniques (Windows, PowerShell, python3)
- Pré-checks avant toute boucle (règle 0.3 + garde-fou filtre) :
  `git status --porcelain --untracked-files=no` (doit être vide) ;
  `python3 td_protocols.py --check` (exit 0) ;
  `python3 main_determinism_check.py` (exit 0).
- Run unitaire :
  `python3 -u main_train_td.py --encoder td0 --protocol heldout_nu --z-target-mode energy_centered --seed 42 2>&1 | Tee-Object -FilePath "outputs/logs/td_corrected/td_run_td0_heldout_nu_zenergy_centered_s42.log"; if ($LASTEXITCODE -ne 0) { throw "run KO" }`
  (logs dans `outputs/logs/td_corrected/`, jamais dans un log déjà suivi/commité).
- Boucle 60 runs TD1–TD3, ordre `td2 → td1 → td3`, protocoles `heldout_nu` puis
  `heldout_nu005`, modes `energy_centered`/`none`, graines 42,7,123,2024,999 (une
  instruction par ligne ou séparées par `;`).
- Évaluation : `python3 main_evaluate_td.py 2>&1 | Tee-Object -FilePath outputs/logs/td_corrected/td_evaluate.log`
  → `outputs/logs/td_encoder_results.json`.
- Validation bit à bit d'un run existant : `python3 main_validate_run.py --compare-only <dossier_run>`.
- Hachage des artefacts : `Get-ChildItem -Recurse outputs/models/latent_dynamics_autoencoder_seq/<run> | Get-FileHash -Algorithm SHA256`
  → `outputs/logs/td_corrected/td_artifacts_sha256.txt` (300 fichiers, commit `91199bc`).

### Mapping résultat → artefact / commit
| résultat | artefact / commit |
|---|---|
| Synthèse + Step 8/9 + écarts | README, commits `e49a957` (doc) + ce commit (synthèse + inventaire) |
| Clé `r2_identity_nuD_z` en doublon | `12b57db` |
| Plafond analytique R² 0.9995 | `outputs/logs/experiment_a_seq_analytic_baseline.json` |
| Step 5 window-A (z non aligné) | `outputs/models/latent_dynamics_autoencoder_seq/window_w25_*` |
| Step 6 A-Physics (prereg `212db0f`) | `window_w25_heldout_nu_z*` + `PREREGISTRATION_experiment_a_physics_v1.md` |
| A1 (prereg `5623042`, verdict `ccb139a`) | `outputs/prereg/a1_intra_trajectory_prereg.md` + `main_analyze_a1_intra_trajectory.py` |
| B1 (prereg `f431e66`, amend. `fc1075b`, artefacts `a6cb108`) | `w25_heldout_nu005_z*` + `outputs/prereg/b1_interpolation_prereg*` |
| TD0 corrigé (10 runs heldout_nu) | dossiers `td0_heldout_nu_*`, commit `1a93648` + log postfix |
| TD0 nu005 + TD1–TD3 (70 runs) | dossiers `td0_heldout_nu005_*`, `td{1,2,3}_*`, commit `91199bc` + SHA-256 |
| Runs erronés archivés (hors verdict) | `td0_indist_nu001_*` + `outputs/prereg/td0_indist_nu001_archive_note.md`, commit `d844a7a` |
| Verdicts D1/D2/D3 + C6' | `outputs/logs/td_encoder_results.json` (`verdict_principal_D1 = "INDETERMINE"`) |
| Prereg TD + amendment 1 | `outputs/prereg/td_encoder_prereg.{md,json}` (`515cf5b`) + `td_encoder_prereg_amendment_1.md` (`182c8fc`) |

### Note sur la clé `r2_identity`
- Les artefacts existants (JSON de métriques) ne contiennent **pas** la clé
  `r2_identity_nuD_z` / `r2_identity` (ajoutée en `12b57db`, code seul) : une réévaluation
  produit un JSON avec une clé de plus et des **valeurs identiques** — différence de
  format, pas de non-reproductibilité.
