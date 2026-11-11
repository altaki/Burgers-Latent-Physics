# Burgers-Latent-Physics

Can neural networks discover physically meaningful latent scalar observables from the
dynamics of the viscous 1D Burgers equation?

    u_t + u u_x = nu * u_xx      on [0, L] periodic

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
- `zero_mean=True` kept for fourier / random_smooth (made explicit in config); gaussian_bumps
  unchanged (no zero_mean parameter → keeps the conserved-mass control on that subpopulation)
- IC families unchanged: fourier (max_mode=3), gaussian_bumps (n_bumps=2), random_smooth
  (spectral_decay=3.0)
- solver unchanged: L = 2π, RK4, dt = 2e-4, T = 2.0, save_every = 100 → 101 snapshots, Δt = 0.02
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

## Step 6 — Experiment A-Physics (implemented, run pending)

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
- **Temporal alignment**: z_t is computed from window [t−w+1, t] only; the target uses E_{t+1}
  (outside the window) but it is a *training target*, not an input — no leakage at inference.
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
- **Runs**: `aphys` (energy_centered, λ_z=1, λ_mono=0.1) on **3 seeds** (42, 7, 123);
  `asup` (supervised) on 1 seed as redundancy control; optional `aphys_mono0` (λ_mono=0)
  on 1 seed to isolate the hinge's residual effect. w=25 only (C2: w=50 ≈ w=25).
- **Predictions** (w=25, ν=0.01 held-out): ceiling = analytic baseline (Spearman ≈ 1.0,
  R² ≈ 0.9999); realistic with a trainable encoder: **Spearman(z,νD) ≈ 0.85–0.95,
  R² affine ≈ 0.6–0.9**. Falsification: Spearman @ν=0.01 < 0.8 (below probe-on-h = 0.655
  would even mean the anchoring hurts) → the scalar bottleneck cannot preserve νD.
- **Success criteria**: verdict v2 (`verdict_a_seq`): r_holdout ≥ 0.8, R²_affine ≥ 0.5,
  calibration slope ∈ [0.5, 2], beats ν-oracle, structural = rank ≥ 3 + rec RMSE ≤ 0.15
  (monotone not_applicable); IDENTIFIABLE / PARTIAL / NOT IDENTIFIABLE.

Run (torch env):

```
python main_analytic_nuD_baseline.py                       # baseline analytique (sans réseau)
for s in 42 7 123; do
  python main_train_latent_seq.py --tag aphys --seed $s --z-target energy_centered --z-weight 1.0 --mono-weight 0.1
done
python main_train_latent_seq.py --tag asup --seed 42 --z-target supervised --z-weight 1.0 --mono-weight 0.1
python main_analyze_window_a_latents.py        # inclut les runs _z*_s* automatiquement
```

Read `outputs/logs/experiment_a_seq_summary_aphys_s{seed}.json` → `verdict.window_a`
(`verdict_version: v2_aphysis`, `monotone_check: not_applicable`).

Dataset facts measured on the npz (nu_sweep_v1): Nx=1024, 101 snapshots, dt_saved=0.02
(T=2.0), max|mass0| = 3.49e-15 (zero_mean applied to ALL IC families, incl. gaussian_bumps),
max|u| = 1.79. Note: this differs from the legacy spec (Nx=128, solver dt=1e-4,
save_every=200) — the nu_sweep_v1 npz is the ground truth.

## Open questions

- ~~Does z track dissipation or the whole gradient/shock complex?~~ **Answered (Architecture A +
  nu_sweep_v1 audit): the old dissipation-like z was collapse-induced; on nu_sweep_v1 the νD/D
  degeneracy is broken globally (Spearman 0.642 vs 1.0 legacy) but NOT within a ν stratum, and
  D / max|u_x| / TV remain one shock complex (0.95–0.99).**
- Is νD separable from D by a *trained model* at held-out ν? → **Answered (Experiment C +
  C2): NO at w=5 (all variants NOT IDENTIFIABLE; νD head ≈ c·D head, c ≈ 0.1), but YES largely
  once the temporal window is long enough — C2: w=25/50 give R² νD = +0.69 and Spearman 0.97
  at held-out ν=0.01, beating the ν-oracle, via the ν-agnostic channel dE/dt = −νD.**
- Can a *latent* z (Architecture A loss) align with νD at held-out ν? → **Answered (window-A,
  Step 5): NO — w=25 r_holdout = 0.249, w=50 = −0.035 (NOT IDENTIFIABLE); z is a generic
  monotone decay scalar, no collapse (rank 6–7). The νD info is in the window but loss A
  does not select it. Next: per-dim h eval, then physics-informed z (z ≈ −ΔE/Δt).**
- Which loss term creates the structured z — predictor or monotonicity hinge? (A0 ablation, not run)
- ~~Do residual h dims encode shock position / Fourier content once collapse is fixed?~~ **Partially:
  with the collapse fixed, all h dims are static IC labels (shock_location, Fourier amps, spectral
  centroid, mass); no dynamic dim emerged at fixed ν. Note: on nu_sweep_v1 mass0 ≈ 0 everywhere
  (zero_mean for all families), so the mass label is gone.**
- Does the alignment persist at larger T? (nu_sweep_v1 covers ν ∈ {0.01, 0.05, 0.20};
  larger T not covered)
