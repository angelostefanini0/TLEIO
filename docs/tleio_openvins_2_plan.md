# TLEIO OpenVINS-Inspired Filter Plan 2

This document is the second implementation plan for improving the TLEIO filter by taking additional base-estimator ideas from OpenVINS while keeping TLEIO's intrinsic problem definition fixed.

The first implementation pass added regressed covariance support, dynamic chi-square gating, FEJ-style clone Jacobians, covariance repair, a block update path, and tests. The ME004 ablations showed that most of the large gain came from using regressed covariance, while the OpenVINS-inspired filter-logic changes produced smaller but real gains. This second plan therefore focuses on deeper MSCKF infrastructure that can still be transferred without changing the measurement model.

## Fixed Constraints

These constraints must not be changed in this implementation pass:

1. TLEIO remains a translation-only learned relative-motion filter.
2. The learned measurement remains four consecutive relative translations, shape `4 x 3`, producing a 12D residual.
3. The clone window length remains five poses.
4. No OpenVINS visual feature tracking, landmark state, reprojection residual, or feature nullspace update is introduced.
5. No network architecture, training target, or output file format is changed.
6. The existing output trajectory format remains `timestamp_s px py pz qx qy qz qw`.

The allowed changes are base-filter changes around propagation, update conditioning, covariance handling, FEJ consistency, diagnostics, and test coverage.

## Current Baseline

The implementation target is the current `main_filter_pietro` branch.

As of this plan, the intended practical defaults are:

```text
use_regressed_covariance = True
use_fej = True
enable_chi2_gating = True
covariance_repair_mode = "jitter"
imu_noise_model = "discrete"
use_block_update = False
```

Before implementing this plan, save a new baseline run on ME004:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004
```

Copy the resulting trajectory and plots to:

```text
outputs/comparison_ME004/openvins_2_baseline/
```

Record these metrics in the implementation notes:

1. raw position RMSE printed by `main_filter.py`
2. raw rotation RMSE printed by `main_filter.py`
3. Sim3 aligned/scaled ATE
4. number of rejected updates
5. mean residual norm
6. mean correction norm
7. covariance repair events

The changes below must be evaluated against this baseline, not against the old `main` branch.

## OpenVINS References To Use

The implementation should refer to these OpenVINS source locations:

1. `open_vins/ov_msckf/src/state/Propagator.cpp`
   - `Propagator::propagate_and_clone()`
   - `Propagator::predict_and_compute()`
   - `Propagator::select_imu_readings()`
   - summed transition/noise propagation pattern
2. `open_vins/ov_msckf/src/update/UpdaterHelper.cpp`
   - `measurement_compress_inplace()`
   - Givens/QR-style update conditioning
3. `open_vins/ov_msckf/src/update/UpdaterMSCKF.cpp`
   - chi-square gating after constructing the marginalized/compressed residual system
4. `open_vins/ov_msckf/src/state/StateHelper.cpp`
   - covariance block removal during marginalization
   - state ordering checks
5. `open_vins/ov_msckf/src/update/UpdaterHelper.cpp`
   - FEJ use in Jacobian construction

Do not port visual-feature logic. Use these files only for estimator mechanics.

## Phase 0: Safety, Baseline, And Reproducibility

### Goal

Make every later change measurable and reversible.

### Files

1. `docs/tleio_openvins_2_plan.md`
2. `docs/tleio_filter_implementation_changes.md`
3. `src/main_filter.py`
4. `scripts/filter_diagnostics.py`
5. optional new helper script under `scripts/testing/` or `scripts/filter_ablation.py`

### Instructions

1. Confirm the current branch is `main_filter_pietro`.
2. Confirm there are no uncommitted tracked changes unrelated to this task.
3. Run:

   ```bash
   python -m pytest tests
   ```

4. Run the ME004 baseline command listed above.
5. Save the baseline outputs under `outputs/comparison_ME004/openvins_2_baseline/`.
6. Generate or reuse a small metric script that computes:
   - raw ATE RMSE
   - SE3 aligned ATE
   - Sim3 aligned/scaled ATE
   - scale factor
   - rotation RMSE
7. Store the baseline metrics in:

   ```text
   outputs/comparison_ME004/openvins_2_baseline/metrics.md
   ```

### Acceptance Criteria

1. `pytest` passes before any behavior changes.
2. The baseline output folder contains trajectory, plots, and metrics.
3. The baseline metrics are reproducible by rerunning the metric script.

## Phase 1: Propagation Refactor With OpenVINS-Style Interval Pairs

### Motivation

OpenVINS propagates over pairs of IMU samples and computes one transition/noise contribution per interval. TLEIO currently stores `ImuMeasurement` objects with `dt` already attached to a sample. This works, but it makes midpoint/trapezoidal integration and exact OpenVINS-style noise accumulation awkward.

The goal is to add a clearer propagation path that uses explicit interval pairs without changing filter outputs until the new path is enabled.

### Files

1. `src/filter/imu_buffer.py`
2. `src/main_filter.py`
3. `src/filter/scekf.py`
4. `tests/test_imu_segments.py` or `tests/test_main_filter_inputs.py`
5. `tests/test_propagation.py`

### New Data Structure

Add a lightweight interval representation:

```python
@dataclass(frozen=True)
class ImuInterval:
    t0: float
    t1: float
    accel0: np.ndarray
    gyro0: np.ndarray
    accel1: np.ndarray
    gyro1: np.ndarray

    @property
    def dt(self) -> float:
        return self.t1 - self.t0
```

The existing `ImuMeasurement` path must remain available until all tests and ME004 results are reviewed.

### Implementation Steps

1. Add `ImuInterval` in `src/filter/imu_buffer.py`.
2. Add `_build_exact_imu_intervals()` in `src/main_filter.py`.
3. `_build_exact_imu_intervals()` must:
   - include exact start and exact end times
   - include all raw IMU samples strictly inside the interval
   - interpolate accel/gyro at start and end if needed
   - build consecutive pairs `(sample_i, sample_i+1)`
   - reject non-finite values
   - reject duplicate timestamps
   - reject non-positive `dt`
4. Keep `_build_exact_imu_segment()` unchanged initially.
5. Add a config flag:

   ```python
   imu_interval_mode: str = "sample_dt"
   ```

   Supported values:

   ```text
   "sample_dt"     current behavior
   "paired_samples" new OpenVINS-style behavior
   ```

6. Add CLI:

   ```bash
   --imu_interval_mode {sample_dt,paired_samples}
   ```

7. Add `ImuMSCKF.propagate_intervals(intervals)` without deleting `propagate(imu_data)`.
8. In the first implementation, `propagate_intervals()` must use the same nominal integration convention as the current code, but with explicit interval inputs. Behavior should be very close to `sample_dt`.
9. Only after equivalence tests pass, allow later phases to introduce midpoint/RK4 integration on the interval path.

### Tests

1. `test_exact_imu_intervals_include_requested_start_and_end`
2. `test_exact_imu_intervals_are_strictly_increasing`
3. `test_exact_imu_intervals_interpolate_boundary_values`
4. `test_propagate_intervals_matches_sample_dt_for_constant_imu`
5. `test_paired_sample_mode_runs_me004_smoke`

### Acceptance Criteria

1. Existing default behavior is unchanged when `imu_interval_mode="sample_dt"`.
2. The new interval path runs on synthetic tests.
3. Constant-IMU propagation through old and new paths agrees within a documented tolerance.
4. ME004 can run with both interval modes.

## Phase 2: OpenVINS-Style Summed Transition And Noise Accumulation

### Motivation

OpenVINS accumulates transition and process noise over all small IMU intervals, then applies one covariance update:

```text
Phi_summed = Phi_i * Phi_summed
Q_summed = Phi_i * Q_summed * Phi_i.T + Q_i
P_ii_new = Phi_summed * P_ii * Phi_summed.T + Q_summed
P_ci_new = P_ci * Phi_summed.T
P_ic_new = Phi_summed * P_ic
```

TLEIO currently updates covariance at each sample. Both are mathematically related, but the OpenVINS form makes it easier to audit and test full-interval propagation.

### Files

1. `src/filter/scekf.py`
2. `tests/test_propagation.py`

### Implementation Steps

1. Add a function:

   ```python
   accumulate_imu_transition_and_noise(intervals, state, noise_params, model) -> tuple[Phi, Qd]
   ```

2. For each interval:
   - compute `Phi_i`
   - compute `Q_i`
   - update:

     ```python
     Phi = Phi_i @ Phi
     Q = Phi_i @ Q @ Phi_i.T + Q_i
     Q = 0.5 * (Q + Q.T)
     ```

3. Add `ImuMSCKF.propagate_intervals()` support for:

   ```python
   covariance_propagation_mode: str = "per_sample"
   ```

   Supported values:

   ```text
   "per_sample" current behavior
   "summed" OpenVINS-style accumulated transition/noise
   ```

4. Add CLI:

   ```bash
   --covariance_propagation_mode {per_sample,summed}
   ```

5. For the `summed` mode:
   - propagate the nominal state sequentially over intervals
   - accumulate `Phi` and `Q`
   - update the IMU covariance block once at the end
   - update IMU-to-clone cross-covariance blocks consistently
6. Do not change clone count or marginalization order.
7. Store diagnostics:
   - total interval `dt`
   - number of intervals
   - min/max interval `dt`
   - norm of `Phi_summed`
   - min eigenvalue of `Q_summed`

### Tests

1. `test_summed_covariance_matches_per_sample_for_short_constant_sequence`
2. `test_summed_covariance_preserves_cross_covariance_shape`
3. `test_summed_noise_is_symmetric_psd`
4. `test_summed_mode_rejects_empty_intervals`

### Acceptance Criteria

1. The old `per_sample` path remains the default until ME004 ablation proves otherwise.
2. The `summed` path is numerically stable on synthetic tests.
3. ME004 run completes with `--imu_interval_mode paired_samples --covariance_propagation_mode summed`.

## Phase 3: Midpoint/RK4 Nominal IMU Integration Option

### Motivation

OpenVINS supports more careful mean propagation than the current simple zero-order integration. TLEIO can add a midpoint option without changing measurements or state size.

### Files

1. `src/filter/scekf.py`
2. `tests/test_propagation.py`

### Implementation Steps

1. Add config:

   ```python
   nominal_integration_method: str = "euler"
   ```

   Supported values:

   ```text
   "euler"     current behavior
   "midpoint"  average accel/gyro over interval
   ```

2. Add CLI:

   ```bash
   --nominal_integration_method {euler,midpoint}
   ```

3. Implement midpoint only on `ImuInterval` inputs:
   - `gyro_mid = 0.5 * (gyro0 + gyro1) - bg`
   - `accel_mid = 0.5 * (accel0 + accel1) - ba`
   - integrate rotation using `mat_exp(gyro_mid * dt)`
   - integrate velocity and position using the previous rotation or optionally the half-step rotation; document the chosen convention
4. Do not add RK4 in the first implementation unless midpoint is stable and tests pass.
5. Add a placeholder enum value only if needed; do not expose unimplemented RK4 in CLI.

### Tests

1. `test_midpoint_matches_euler_for_constant_imu`
2. `test_midpoint_reduces_error_for_linearly_varying_gyro_synthetic_case`
3. `test_midpoint_rejects_non_interval_inputs`

### Acceptance Criteria

1. Midpoint is optional.
2. Euler remains available.
3. Midpoint improves or matches synthetic integration cases.
4. ME004 ablation decides whether midpoint becomes default.

## Phase 4: FEJ Consistency Audit And Full FEJ Mode

### Motivation

The first pass added FEJ clone poses for measurement Jacobians. OpenVINS uses FEJ more systematically: propagated IMU FEJ values are maintained, clone FEJ values are set at clone creation, and Jacobians read FEJ values consistently when the estimator is in FEJ mode.

TLEIO should make FEJ rules explicit and testable.

### Files

1. `src/filter/scekf.py`
2. `src/filter/measurement_triplet.py`
3. `tests/test_measurement_triplet.py`
4. `tests/test_scekf_update.py`

### Implementation Steps

1. Add explicit comments and assertions for FEJ invariants:
   - `len(clone_Rs) == len(clone_Rs_fej)`
   - `len(clone_ps) == len(clone_ps_fej)`
   - FEJ clones are appended at augmentation time
   - FEJ clones are never modified by `_inject_error_state()`
   - FEJ clones are marginalized with their current clone counterpart
2. Add a method:

   ```python
   state.assert_clone_fej_consistency()
   ```

3. Call the assertion after:
   - initialization
   - augmentation
   - update
   - marginalization
4. Add optional current-IMU FEJ storage:

   ```python
   R_fej, v_fej, p_fej
   ```

   This is only needed if propagation Jacobians will use FEJ values. Do not add it unless Phase 2 or Phase 3 requires it.
5. Add config:

   ```python
   fej_scope: str = "clone_update"
   ```

   Supported values:

   ```text
   "clone_update"  current FEJ behavior
   "full"          clone update FEJ plus propagation FEJ where applicable
   ```

6. If `fej_scope="full"` is implemented:
   - propagation Jacobians must use FEJ attitude/velocity/position where this matches the OpenVINS derivation
   - residual values must still be evaluated at the current estimate
   - tests must show clone FEJ values stay frozen
7. Do not make `fej_scope="full"` default until it improves real-sequence metrics.

### Tests

1. `test_fej_clone_lists_remain_aligned`
2. `test_error_injection_does_not_modify_fej_clones`
3. `test_marginalization_removes_matching_fej_clone`
4. `test_fej_residual_value_equals_non_fej_residual_value`
5. `test_fej_jacobian_differs_when_clone_has_been_corrected`

### Acceptance Criteria

1. Current FEJ behavior is fully tested.
2. If full FEJ is added, it is behind a flag and ablated on ME004.
3. No hidden FEJ mutation occurs during update injection.

## Phase 5: Whitened And Conditioned Measurement Update

### Motivation

OpenVINS compresses and conditions residual systems before update. TLEIO's residual is only 12D, so full OpenVINS feature nullspace projection is not applicable. However, whitening by measurement covariance and using numerically stable solves is applicable.

### Files

1. `src/filter/scekf.py`
2. `src/filter/utils/math_utils.py`
3. `tests/test_scekf_update.py`
4. `tests/test_measurement_conditioning.py`

### Implementation Steps

1. Add update mode:

   ```python
   update_solve_method: str = "innovation"
   ```

   Supported values:

   ```text
   "innovation" current solve through S = HPH.T + R
   "whitened"   Cholesky-whiten residual and Jacobian first
   "qr"         QR-based solve/compression if useful
   ```

2. Add CLI:

   ```bash
   --update_solve_method {innovation,whitened,qr}
   ```

3. For `whitened`:
   - compute Cholesky or eigen repair of `R`
   - solve `L y = residual`
   - solve `L H_w = H`
   - run update with `R_w = I`
   - compute Mahalanobis distance as `y.T @ solve(H_w P H_w.T + I, y)`
4. If `R` is diagonal, use fast diagonal whitening.
5. If Cholesky fails:
   - apply the configured covariance repair policy to `R`
   - retry
   - log a diagnostic
6. For `qr`:
   - only implement if `H.rows > H.cols` or if future compression needs it
   - since current residual is 12D and state is larger, QR compression will usually do nothing
   - do not make QR default if it is behavior-equivalent
7. Return update diagnostics:
   - `update_solve_method`
   - `condition_number_R`
   - `condition_number_S`
   - `whitening_applied`

### Tests

1. `test_whitened_update_matches_innovation_update_for_identity_R`
2. `test_whitened_update_matches_innovation_update_for_diagonal_R`
3. `test_whitened_update_handles_extreme_regressed_sigmas`
4. `test_condition_number_diagnostics_are_returned`

### Acceptance Criteria

1. Whitened update agrees with the current update in well-conditioned cases.
2. Whitened update remains stable for very small/large sigma values.
3. ME004 ablation decides whether whitening becomes default.

## Phase 6: Covariance Marginalization And State Ordering Audit

### Motivation

OpenVINS uses strict state variable ordering and generic covariance block marginalization. TLEIO currently uses direct index construction for the fixed state layout. Because the clone count and layout are fixed, a full variable system is unnecessary, but the index logic should be made auditable and tested.

### Files

1. `src/filter/scekf.py`
2. `tests/test_state_layout.py`
3. `tests/test_scekf_update.py`

### Implementation Steps

1. Add state layout helpers:

   ```python
   IMU_DIM = 15
   CLONE_DIM = 6
   clone_slice(clone_index) -> slice
   expected_covariance_dim() -> int
   ```

2. Replace ad hoc offset calculations with these helpers.
3. Add `state.assert_covariance_shape_matches_clones()`.
4. Add `state.clone_count_from_covariance()` for diagnostics.
5. In `augment_clone()`:
   - assert old covariance dimension before expansion
   - assert new covariance dimension after expansion
6. In `marginalize_oldest_clone()`:
   - assert at least one clone exists
   - compute the marginalized slice using the helper
   - verify covariance dimensions after block removal
   - verify FEJ/current clone lists stay aligned
7. Keep marginalization policy unchanged: always remove the oldest clone after each update attempt.

### Tests

1. `test_clone_slice_matches_covariance_layout`
2. `test_covariance_shape_matches_clone_count_after_augmentation`
3. `test_covariance_shape_matches_clone_count_after_marginalization`
4. `test_marginalization_preserves_remaining_cross_correlations`
5. `test_state_layout_helpers_reject_invalid_clone_index`

### Acceptance Criteria

1. No behavior change is expected from this phase.
2. Indexing bugs become test failures instead of silent covariance corruption.

## Phase 7: Innovation Consistency And Gating Calibration

### Motivation

In the ME004 feature-grid runs, chi-square gating rejected zero updates. This means one of three things:

1. the sequence has no update outliers,
2. the regressed covariance is conservative,
3. the gating statistic is not informative enough.

OpenVINS uses chi-square gating after residual construction and conditioning. TLEIO should log enough normalized innovation statistics to understand whether gating is useful.

### Files

1. `src/filter/scekf.py`
2. `src/main_filter.py`
3. `scripts/filter_diagnostics.py`
4. `tests/test_scekf_update.py`

### Implementation Steps

1. Store per-update diagnostics in `run_filter()`:
   - timestamp
   - Mahalanobis squared
   - chi-square threshold
   - ratio
   - residual norm
   - correction norm
   - min/max measurement sigma for the window
   - update accepted/rejected
2. Save diagnostics to:

   ```text
   outputs/main_filter/<dataset>/<sequence>/update_diagnostics.csv
   ```

3. Add summary statistics:
   - median chi-square ratio
   - 95th percentile chi-square ratio
   - max chi-square ratio
   - number of rejected updates
4. Add optional gating modes:

   ```python
   gating_mode: str = "global"
   ```

   Supported values:

   ```text
   "global" current 12D gate
   "per_edge" optional four independent 3D gates
   ```

5. Implement `per_edge` only if diagnostics show a need. If implemented:
   - compute one 3D Mahalanobis statistic per edge
   - reject or inflate per-edge covariance when one edge is bad
   - keep global gating available
6. Do not make `per_edge` default without ablation.

### Tests

1. `test_update_diagnostics_csv_is_written`
2. `test_chi2_ratio_summary_matches_raw_values`
3. `test_per_edge_gate_rejects_single_bad_edge_if_enabled`
4. `test_global_gate_behavior_unchanged_by_default`

### Acceptance Criteria

1. Every run writes update diagnostics.
2. Gating can be evaluated from saved files without rerunning.
3. Default gating behavior remains global unless ablation proves otherwise.

## Phase 8: Measurement Covariance Calibration Diagnostics

### Motivation

Regressed covariance is the largest observed improvement. The next useful OpenVINS-style step is not to blindly add more update flags, but to verify whether measurement covariance is calibrated against realized residuals.

### Files

1. `src/main_filter.py`
2. `scripts/filter_diagnostics.py`
3. optional new script `scripts/filter_covariance_calibration.py`

### Implementation Steps

1. For each update window, log:
   - raw sigma values for the four edges
   - clipped sigma values
   - residual components
   - normalized residual components `residual_i / sigma_i`
2. Add a calibration script that reads `update_diagnostics.csv` and computes:
   - empirical residual standard deviation per axis
   - mean predicted sigma per axis
   - normalized residual RMS per axis
   - histogram/percentiles of normalized residuals
3. Add optional global sigma scale search:
   - evaluate a grid of `meas_cov_scale`
   - no code behavior change, only diagnostics
4. Do not introduce adaptive covariance updates inside the EKF in this pass unless diagnostics clearly show a stable correction rule.

### Tests

1. `test_covariance_diagnostics_include_sigma_columns_when_available`
2. `test_covariance_calibration_script_runs_on_synthetic_csv`

### Acceptance Criteria

1. We can tell whether the network sigmas are underconfident or overconfident.
2. Any future covariance tuning is data-backed.

## Phase 9: Ablation Protocol

### Goal

Evaluate each change on ME004 with consistent metrics.

### Required Runs

After each behavior-changing phase, run at least:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004
```

For full phase evaluation, run combinations only over newly introduced flags plus the current best defaults. Do not rerun the entire 96-grid unless a core default changes.

Minimum ablations after this plan:

1. baseline current best
2. paired IMU intervals only
3. paired intervals + summed covariance
4. paired intervals + midpoint nominal integration
5. whitened update
6. whitened update + paired/summed propagation
7. full FEJ scope if implemented

Save results under:

```text
outputs/comparison_ME004/openvins_2/
```

Each run folder must contain:

1. `stamped_traj_estimate.txt`
2. plots generated by `main_filter.py`
3. `update_diagnostics.csv`
4. `run.log`
5. one row in `openvins_2_summary.csv`

### Metrics

Use the same metrics for all runs:

1. raw ATE RMSE
2. SE3 aligned ATE
3. Sim3 aligned/scaled ATE
4. Sim3 scale factor
5. raw rotation RMSE
6. max position error
7. max rotation error
8. rejected update count
9. median chi-square ratio
10. 95th percentile chi-square ratio

### Acceptance Criteria

1. A change is considered useful only if it improves at least one target metric without a severe regression in another.
2. Because ME004 is one sequence, no change should become a hard default solely from ME004 unless it is numerically safer and behavior-equivalent.
3. If a change improves Sim3 ATE but worsens raw ATE, document the tradeoff and leave it behind a flag.

## Phase 10: Documentation Update

### Files

1. `docs/tleio_filter_implementation_changes.md`
2. `docs/tleio_openvins_filter_comparison.md`
3. this file

### Instructions

After implementing and evaluating the phases, update `docs/tleio_filter_implementation_changes.md` with:

1. exact files changed
2. new config fields and CLI flags
3. which OpenVINS mechanism each change corresponds to
4. default values
5. ablation summary
6. known non-improvements
7. remaining risks

Do not overwrite the first plan. This second plan should remain as the implementation checklist.

## Implementation Order

Follow this order exactly:

1. Phase 0: baseline and metrics
2. Phase 1: explicit IMU intervals
3. Phase 2: summed transition/noise propagation
4. Phase 3: midpoint nominal integration
5. Phase 4: FEJ consistency audit
6. Phase 5: whitened update
7. Phase 6: state layout and marginalization audit
8. Phase 7: innovation diagnostics
9. Phase 8: covariance calibration diagnostics
10. Phase 9: ablation protocol
11. Phase 10: documentation update

If any phase fails tests or worsens ME004 substantially, stop and document the failure before continuing. Do not stack multiple unvalidated behavior changes.

## Non-Goals

Do not implement the following under this plan:

1. visual feature residuals
2. landmark states
3. OpenVINS feature nullspace projection
4. changing clone count
5. changing the learned measurement output
6. adaptive online covariance estimation inside the EKF
7. automatic parameter optimization
8. committing output files to git

## Expected Outcome

This plan may not yield a large ME004 ATE improvement, because the main remaining limitation is the translation-only learned measurement. The expected value is:

1. a more OpenVINS-like propagation/update backbone,
2. stronger numerical conditioning,
3. clearer FEJ and marginalization invariants,
4. better innovation and covariance diagnostics,
5. a filter that is safer to tune and extend.

The most promising performance-changing phases are:

1. Phase 2: summed propagation/noise accumulation,
2. Phase 3: midpoint integration,
3. Phase 5: whitened update under extreme regressed covariance,
4. Phase 8: covariance calibration diagnostics leading to later tuning.

