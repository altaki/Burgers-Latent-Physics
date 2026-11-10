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
ml_train_latent.py            # LatentAutoencoder + LatentDynamicsModel + LatentDynamicsAutoencoder training
ml_evaluate.py                # extraction, regression/correlation/monotonicity metrics +
                              # latent PCA/rank evaluation for Architecture A
evaluate + identify + compare + visualize entry points:
main_generate_data.py, main_train_supervised.py, main_train_latent.py, main_evaluate.py,
main_identify_latent_variable.py, main_compare_z_and_dissipation.py,
main_visualize_results.py, main_visualize_latent_results.py, main_experiment_c.py
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

## Step 4 — Experiment C: supervised separability baseline on nu_sweep_v1 (implemented, validated, production run pending)

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

## Open questions

- ~~Does z track dissipation or the whole gradient/shock complex?~~ **Answered (Architecture A +
  nu_sweep_v1 audit): the old dissipation-like z was collapse-induced; on nu_sweep_v1 the νD/D
  degeneracy is broken globally (Spearman 0.642 vs 1.0 legacy) but NOT within a ν stratum, and
  D / max|u_x| / TV remain one shock complex (0.95–0.99).**
- Is νD separable from D by a *trained model* at held-out ν? → **Experiment C (Step 4),
  implemented, run pending** — verdict per variant in `outputs/logs/experiment_c_summary.json`.
- Which loss term creates the structured z — predictor or monotonicity hinge? (A0 ablation, not run)
- ~~Do residual h dims encode shock position / Fourier content once collapse is fixed?~~ **Partially:
  with the collapse fixed, all h dims are static IC labels (shock_location, Fourier amps, spectral
  centroid, mass); no dynamic dim emerged at fixed ν. Note: on nu_sweep_v1 mass0 ≈ 0 everywhere
  (zero_mean for all families), so the mass label is gone.**
- Does the alignment persist at larger T? (nu_sweep_v1 covers ν ∈ {0.01, 0.05, 0.20};
  larger T not covered)
