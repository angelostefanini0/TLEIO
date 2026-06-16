# TLEIO Filter Improvement Implementation Plan

This plan converts the OpenVINS comparison into a precise implementation sequence for improving the TLEIO filter. The goal is to make changes that are measurable, reversible, and locally testable. The plan prioritizes correctness and diagnosability before larger algorithmic changes such as FEJ.

## Scope

In scope:

1. Measurement covariance handling, including regressed diagonal sigmas.
2. Chi-square gating and update diagnostics.
3. Measurement residual/Jacobian cleanup and numerical tests.
4. IMU propagation/covariance discretization audit and tests.
5. Clone first-estimate Jacobian support inspired by OpenVINS FEJ.
6. Covariance symmetry/SPD diagnostics.
7. Optional small-block update algebra after behavior is locked down.

Out of scope for the first implementation pass:

1. Adding visual feature updates, landmarks, or OpenVINS feature nullspace projection.
2. Changing the learned model interface beyond consuming already saved sigma columns.
3. Changing the number of clones or the basic five-clone, four-edge update window.
4. Rewriting the runner into a general streaming estimator.

## Current Baseline To Preserve

Before editing behavior, preserve the current baseline by running and saving:

1. One short `src/main_filter.py` run on the default sequence.
2. The current trajectory, diagnostics summary, rejected update count, mean residual norm, and mean delta norm.
3. A small smoke run from `inspect_functions/test_filter.py` if its data dependencies are present.

If any command cannot run due to missing local data, record that in the implementation notes and continue with unit tests.

## Phase 1: Measurement Interface Cleanup And Regressed Covariance

### Motivation

The filter already has most of the plumbing for diagonal regressed covariance, but `run_filter()` disables it with `relative_sigmas = None`. Also, `measurement_triplet.py` still contains comments and unused functions from an older quaternion/6D residual interface, which makes later filter changes risky.

### Files

1. `src/main_filter.py`
2. `src/filter/measurement_triplet.py`
3. `src/filter/scekf.py`
4. New tests under `tests/`, preferably `tests/test_measurement_triplet.py` and `tests/test_main_filter_inputs.py`

### Changes

1. Update `_load_relative_motion_table()` documentation and validation to explicitly accept:
   - `N x 5`: `t0 t1 px py pz`
   - `N x 8`: `t0 t1 px py pz sigma_x sigma_y sigma_z`
   - Keep `N x 9` support only if it is genuinely used elsewhere; otherwise document it as unsupported for the current EKF update.
2. Modify `_build_anchor_times_from_relative_motions()` to return `relative_sigmas = relative_motion_table[:, 5:8]` when at least 8 columns are present.
3. Remove the forced `relative_sigmas = None` in `run_filter()`.
4. Add a `RunnerConfig` flag:
   - `use_regressed_covariance: bool = True`
   - If false, ignore loaded sigma columns and use `assumed_sigma_rel_t`.
5. Add `min_regressed_sigma_m` and `max_regressed_sigma_m` config fields.
   - Clip loaded sigmas to this interval before squaring.
   - Reject non-finite or negative sigmas with a clear `ValueError`.
6. Keep the EKF covariance format diagonal in the first pass:
   - Fill the 12 diagonal entries from the four `sigma_x/y/z` rows.
   - Do not introduce full 12x12 learned covariance unless the model actually outputs correlations.
7. Update docstrings in `measurement_triplet.py`:
   - It is a four-edge translation-only update.
   - Measurement shape is `(4, 3)`.
   - Residual dimension is 12.
8. Remove or quarantine `normalize_triplet_measurement()` because it assumes quaternion columns that are not present.
   - Preferred: delete it if no callers exist.
   - Alternative: rename to a TODO/dead-code section only if removing it would complicate notebooks.

### Tests

1. `test_load_relative_motion_table_accepts_5_and_8_columns`
2. `test_build_anchor_times_extracts_sigmas_for_8_columns`
3. `test_regressed_sigmas_fill_12d_covariance_diagonal`
4. `test_negative_or_nan_sigmas_are_rejected`
5. `test_covariance_flag_can_disable_regressed_sigmas`

### Acceptance Criteria

1. Existing translation-only files still run unchanged.
2. Files with sigma columns change the measurement covariance used by `ekf.update()`.
3. The saved diagnostics report whether fixed or regressed covariance was used.

## Phase 2: Dynamic Chi-Square Gating And Update Diagnostics

### Motivation

OpenVINS derives chi-square thresholds from the tested residual dimension. TLEIO currently uses a hard-coded value. The current value happens to match 12 DoF at 95 percent, but the comment says 6 DoF and future residual changes could silently break gating.

### Files

1. `src/filter/scekf.py`
2. `src/main_filter.py`
3. Tests under `tests/test_scekf_update.py`

### Changes

1. Replace `CHI2_THRESHOLD` with config-driven fields:
   - `chi2_confidence: float = 0.95`
   - `chi2_multiplier: float = 1.0`
   - `enable_chi2_gating: bool = True`
2. Compute threshold at runtime from `residual.shape[0]`.
   - Use `scipy.stats.chi2.ppf`, since SciPy is already a dependency.
   - Store the computed threshold in the update return dictionary.
3. Return additional update diagnostics:
   - `mahalanobis_sq`
   - `chi2_threshold`
   - `residual_dim`
   - `used_regressed_covariance` if passed down from the runner, or report this in runner-level diagnostics.
4. Update `run_filter()` to aggregate:
   - rejected update count
   - accepted update count
   - mean/max Mahalanobis distance
   - mean chi-square ratio `mahalanobis_sq / chi2_threshold`
5. Keep the sign convention unchanged:
   - `build_triplet_update()` returns `d residual / d error`.
   - Correction remains `delta_x = -K @ residual`.

### Tests

1. `test_chi2_threshold_matches_residual_dimension`
2. `test_gating_rejects_large_residual`
3. `test_gating_can_be_disabled`
4. `test_update_returns_gating_diagnostics`

### Acceptance Criteria

1. With the current 12D residual and 95 percent confidence, the threshold is approximately `21.026`.
2. Changing the residual dimension in a test changes the threshold automatically.
3. Existing results are nearly unchanged when using the same threshold and covariance.

## Phase 3: Residual And Jacobian Numerical Validation

### Motivation

TLEIO's update sign convention is subtle: the code applies `-K @ residual` because the Jacobian is for the residual itself. Before changing FEJ or covariance math, lock this down with finite-difference tests.

### Files

1. `src/filter/measurement_triplet.py`
2. `tests/test_measurement_triplet.py`

### Changes

1. Add a test helper that creates five synthetic clones with nontrivial rotations and translations.
2. Build measurements from the exact clone relative translations.
3. Perturb each clone rotation and position by a small error-state vector.
4. Compare finite-difference residual changes with the analytical Jacobian from `build_triplet_update()`.
5. Confirm that applying a small correction in the expected direction reduces residual norm.

### Tests

1. `test_pair_translation_residual_zero_for_perfect_measurement`
2. `test_triplet_jacobian_matches_finite_difference`
3. `test_negative_kalman_step_sign_reduces_residual_in_linearized_case`

### Acceptance Criteria

1. Analytical Jacobian error is below a documented tolerance, for example `1e-5` to `1e-4`.
2. The test covers rotations away from identity.
3. The test fails if the residual sign or clone perturbation convention is accidentally flipped.

## Phase 4: IMU Propagation And Process Noise Audit

### Motivation

OpenVINS is careful about interval-bounded IMU propagation and continuous-to-discrete process noise. TLEIO already builds exact anchor-to-anchor IMU segments in `main_filter.py`; the next step is to test it and clarify the process-noise units.

### Files

1. `src/main_filter.py`
2. `src/filter/scekf.py`
3. `src/filter/imu_buffer.py` if shared segment logic is moved there
4. `tests/test_imu_segments.py`
5. `tests/test_propagation.py`

### Changes

1. Add explicit checks in `_build_exact_imu_segment()`:
   - no duplicate segment times
   - no zero or negative `dt`
   - final timestamp equals `end_time_s`
   - all interpolated values are finite
2. Consider adding the exact start sample to the segment only if the propagation model is changed to use paired samples. Do not add it blindly, because current `ImuMeasurement.dt` represents the interval from the previous timestamp to this sample.
3. Document IMU noise units in `RunnerConfig` and `ImuMSCKF`.
4. Decide one process noise convention:
   - Option A: `sigma_ng`, `sigma_na`, `sigma_nbg`, `sigma_nba` are continuous-time noise densities.
   - Option B: they are per-sample discrete standard deviations.
5. If choosing Option A, update `propagate_covariance()` so `G` contains interval factors and `Q_c` follows the same dimensional convention as OpenVINS.
6. If choosing Option B, keep the current propagation but rename config/docstrings to avoid implying continuous densities.
7. Add stationary-IMU propagation tests:
   - nominal state remains approximately stationary when initialized consistently with gravity
   - covariance diagonal grows monotonically for attitude, velocity, position, and biases under process noise
   - covariance remains symmetric

### Tests

1. `test_exact_imu_segment_hits_requested_end_time`
2. `test_exact_imu_segment_rejects_zero_dt`
3. `test_stationary_imu_nominal_state_stays_consistent`
4. `test_process_noise_covariance_growth_is_monotonic`
5. `test_propagate_covariance_preserves_symmetry`

### Acceptance Criteria

1. Propagation segments are exact and never silently contain zero-`dt` samples.
2. The selected noise convention is documented in code and tests.
3. A process-noise change is accompanied by before/after trajectory diagnostics.

## Phase 5: FEJ-Style Clone Jacobians

### Motivation

OpenVINS uses first-estimate Jacobians to improve estimator consistency and preserve unobservable directions. For TLEIO, the transferable idea is to compute learned relative-translation residuals at the current clone poses, but compute the Jacobian at the clone poses stored when each clone was created.

### Files

1. `src/filter/scekf.py`
2. `src/filter/measurement_triplet.py`
3. Tests under `tests/test_fej_triplet.py`

### Changes

1. Extend `State` with:
   - `clone_Rs_fej`
   - `clone_ps_fej`
2. In `initialize_with_state()`, clear both FEJ lists.
3. In `augment_clone()`, append current `R` and `p` to both the nominal clone lists and FEJ clone lists.
4. In `_inject_error_state()`, update only `clone_Rs` and `clone_ps`; do not update FEJ lists.
5. In `marginalize_oldest_clone()`, pop from both nominal and FEJ lists.
6. Add `use_fej: bool = True` config/args.
7. Modify `build_triplet_update()`:
   - residual uses current `state.clone_Rs` and `state.clone_ps`
   - Jacobian uses FEJ clone poses when `use_fej` is true
   - Jacobian uses current clone poses when `use_fej` is false
8. Keep FEJ optional for A/B testing.

### Tests

1. `test_clone_fej_values_created_on_augmentation`
2. `test_fej_values_do_not_change_after_update_injection`
3. `test_marginalization_keeps_fej_and_nominal_clone_lists_aligned`
4. `test_fej_toggle_changes_jacobian_after_clone_correction`
5. `test_fej_residual_uses_current_state_not_first_estimate`

### Acceptance Criteria

1. FEJ can be turned on/off from config.
2. With no prior correction, FEJ and non-FEJ Jacobians match.
3. After a clone correction, FEJ and non-FEJ Jacobians differ while residual remains current-state residual.
4. Main filter diagnostics compare FEJ on/off on the same sequence.

## Phase 6: Covariance Diagnostics And Repair Policy

### Motivation

OpenVINS treats negative covariance diagonals as a serious diagnostic signal. TLEIO frequently calls `enforce_symmetry_and_pos_def()`, which can hide filter inconsistency. The plan is not to remove repair immediately, but to make it observable and configurable.

### Files

1. `src/filter/utils/math_utils.py`
2. `src/filter/scekf.py`
3. Tests under `tests/test_covariance_policy.py`

### Changes

1. Add a covariance-check helper that reports:
   - symmetry error
   - minimum diagonal
   - minimum eigenvalue for small matrices
   - whether jitter/eigenvalue clipping was applied
2. Add `covariance_repair_mode`:
   - `"strict"`: raise on significant negative eigenvalues/diagonals
   - `"jitter"`: add small diagonal jitter only
   - `"clip"`: current positive-definite enforcement behavior
3. Default development mode should be `"strict"` or `"jitter"`; avoid silent clipping as the default if tests pass.
4. Wrap covariance operations in named checkpoints:
   - after propagation
   - after augmentation
   - after update
   - after marginalization
5. Return or log covariance repair diagnostics in update info.

### Tests

1. `test_covariance_check_accepts_symmetric_psd_matrix`
2. `test_covariance_check_rejects_large_negative_diagonal_in_strict_mode`
3. `test_jitter_mode_repairs_tiny_negative_eigenvalue`
4. `test_update_does_not_require_nontrivial_covariance_repair_on_nominal_case`

### Acceptance Criteria

1. No routine nontrivial covariance repair is needed in the nominal tests.
2. If repair occurs during a real run, diagnostics identify when it happened.
3. Strict mode is available for development.

## Phase 7: Small-Block Update Algebra

### Motivation

OpenVINS updates through a small measurement variable order instead of dense full-state Jacobians. TLEIO's current dense update is acceptable for five clones, so this phase should happen only after Phases 1-6 pass and behavior is stable.

### Files

1. `src/filter/scekf.py`
2. `src/filter/measurement_triplet.py`
3. Tests under `tests/test_block_update.py`

### Changes

1. Add an update helper that accepts:
   - full covariance `P`
   - residual `r`
   - small Jacobian `H_small`
   - state column indices touched by `H_small`
   - measurement covariance `R`
2. Compute:
   - `P_small = P[np.ix_(cols, cols)]`
   - `S = H_small @ P_small @ H_small.T + R`
   - `PHt = P[:, cols] @ H_small.T`
   - `K = solve(S.T, PHt.T).T`
3. Apply the same state correction as the dense path.
4. Keep dense and block paths behind a `use_block_update` flag until equivalence is proven.

### Tests

1. `test_block_update_matches_dense_update_for_random_psd_covariance`
2. `test_block_update_matches_dense_update_for_triplet_jacobian`
3. `test_block_update_handles_cross_covariance_with_imu_state`

### Acceptance Criteria

1. Dense and block paths match to numerical tolerance.
2. Block update becomes default only after equivalence tests pass.

## Phase 8: Evaluation Protocol

Each phase that changes estimator behavior must be evaluated with the same protocol:

1. Run fixed covariance, FEJ off.
2. Run fixed covariance, FEJ on.
3. Run regressed covariance, FEJ off.
4. Run regressed covariance, FEJ on.

For each run record:

1. ATE/RPE metrics already produced by `filter_diagnostics`.
2. Rejected update count.
3. Mean/max Mahalanobis distance.
4. Mean/max chi-square ratio.
5. Mean residual norm.
6. Mean correction norm.
7. Any covariance repair events.

Compare against the baseline saved before Phase 1. Do not accept a behavior-changing phase if it improves one sequence while producing unexplained covariance repairs, excessive rejection, or obvious inconsistency in the diagnostic plots.

## Implementation Order Summary

1. Add tests and cleanup for measurement shape/covariance loading.
2. Enable and validate regressed diagonal covariance.
3. Add dynamic chi-square gating and diagnostics.
4. Add residual/Jacobian finite-difference tests.
5. Harden exact IMU segment construction and audit process-noise units.
6. Implement FEJ clone storage and FEJ Jacobian option.
7. Add covariance diagnostics and repair policy.
8. Add small-block update algebra only after the dense path is validated.

## Rollback Strategy

Every behavior-changing feature should be behind a config flag until the evaluation protocol passes:

1. `use_regressed_covariance`
2. `enable_chi2_gating`
3. `use_fej`
4. `covariance_repair_mode`
5. `use_block_update`

This allows one-factor-at-a-time debugging and makes it possible to recover the current behavior without reverting code.

