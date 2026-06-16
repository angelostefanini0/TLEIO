# TLEIO OpenVINS-Inspired Filter Plan 3

This document is the third implementation plan for improving the TLEIO filter by extracting additional estimator ideas from OpenVINS while preserving TLEIO's fixed learned relative-translation problem.

The second OpenVINS-inspired pass added explicit IMU intervals, summed transition/noise propagation, midpoint nominal integration, whitened update conditioning, state-layout checks, FEJ consistency assertions, and update diagnostics. On ME004, `paired_midpoint` became the best tested configuration:

```text
raw ATE RMSE:              9.406739 m
SE3 aligned ATE RMSE:      5.576098 m
Sim3 scaled/aligned ATE:   5.575106 m
rotation RMSE:             3.396047 deg
```

The default conservative configuration remains:

```text
use_regressed_covariance = True
use_fej = True
enable_chi2_gating = True
covariance_repair_mode = "jitter"
imu_noise_model = "discrete"
imu_interval_mode = "sample_dt"
covariance_propagation_mode = "per_sample"
nominal_integration_method = "euler"
update_solve_method = "innovation"
gating_mode = "global"
fej_scope = "clone_update"
```

This third plan focuses on the next most plausible transferable OpenVINS ideas:

1. measurement covariance calibration from innovation statistics.
2. robust per-edge gating and covariance inflation.
3. a more physically consistent midpoint propagation variant.
4. optional NEES/NIS-style consistency diagnostics.

These are expected to produce smaller but more meaningful improvements than another broad structural refactor.

## Fixed Constraints

These constraints must not be changed:

1. TLEIO remains a translation-only learned relative-motion filter.
2. The learned measurement remains four consecutive relative translations, shape `4 x 3`, producing a 12D residual.
3. The clone window length remains five poses.
4. No OpenVINS visual feature tracking, landmark state, feature nullspace projection, or reprojection residual is introduced.
5. No network architecture, training target, or output file format is changed.
6. The existing output trajectory format remains:

   ```text
   timestamp_s px py pz qx qy qz qw
   ```

7. The default run must remain backward-compatible unless an ablation clearly justifies changing a default.

Allowed changes:

1. measurement covariance scaling and calibration.
2. robust residual gating/inflation logic.
3. optional update-row selection or covariance inflation.
4. optional midpoint propagation variants.
5. diagnostics and test coverage.

## Current Baseline For This Plan

The implementation target is the current `main_filter_pietro` branch after OpenVINS pass 2.

Before implementing behavior changes, save a baseline using the best OpenVINS-2 mode:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples \
  --nominal_integration_method midpoint
```

Copy outputs to:

```text
outputs/comparison_ME004/openvins_3_baseline/
```

Compute and record:

1. raw ATE RMSE.
2. SE3 aligned ATE RMSE.
3. Sim3 scaled/aligned ATE RMSE.
4. Sim3 scale factor.
5. raw rotation RMSE.
6. max position error.
7. max rotation error.
8. rejected update count.
9. median chi-square ratio.
10. 95th percentile chi-square ratio.
11. max chi-square ratio.
12. mean residual norm.
13. mean correction norm.

Store the baseline metrics in:

```text
outputs/comparison_ME004/openvins_3_baseline/metrics.md
```

All later OpenVINS-3 ablations must compare against this baseline, not against the old `main` branch.

## OpenVINS References To Use

Use these OpenVINS files only for estimator mechanics:

1. `open_vins/ov_msckf/src/update/UpdaterMSCKF.cpp`
   - chi-square gating after residual construction.
   - rejecting inconsistent measurements.
2. `open_vins/ov_msckf/src/update/UpdaterHelper.cpp`
   - measurement conditioning and residual system handling.
3. `open_vins/ov_msckf/src/state/Propagator.cpp`
   - midpoint/RK-style IMU propagation conventions.
4. `open_vins/ov_msckf/src/state/StateHelper.cpp`
   - strict covariance/state consistency checks.
5. OpenVINS parameter handling around chi-square multipliers and measurement noise inflation.

Do not port feature residuals, feature tracks, landmarks, or nullspace projection.

## Phase 0: Safety, Baseline, And Metric Script

### Goal

Make OpenVINS-3 changes measurable and reversible.

### Files

1. `docs/tleio_openvins_3_plan.md`
2. `src/main_filter.py`
3. `scripts/filter_diagnostics.py`
4. optional new script: `scripts/compute_ate_metrics.py`
5. optional new script: `scripts/filter_ablation_openvins3.py`

### Instructions

1. Confirm branch:

   ```bash
   git branch --show-current
   ```

   It must be `main_filter_pietro`.

2. Confirm no unrelated tracked changes are present.
3. Run:

   ```bash
   python -m pytest tests
   ```

4. Run the OpenVINS-3 baseline command listed above.
5. Save baseline artifacts under `outputs/comparison_ME004/openvins_3_baseline/`.
6. Create or reuse an ATE metric script that:
   - skips text headers robustly.
   - infers timestamp units for ground truth and estimates.
   - interpolates ground truth onto estimate timestamps.
   - computes raw ATE.
   - computes SE3-aligned ATE.
   - computes Sim3 scaled/aligned ATE.
   - stores the Sim3 scale factor.
7. Store baseline metrics in `metrics.md`.

### Acceptance Criteria

1. Tests pass before behavior changes.
2. Baseline output folder exists and contains trajectory, plots, diagnostics CSV, run log, and metrics.
3. The metric script reproduces the known OpenVINS-2 best result on ME004 within tolerance:

   ```text
   raw ATE RMSE around 9.406739 m
   Sim3 ATE around 5.575106 m
   ```

## Phase 1: Measurement Covariance Calibration Script

### Motivation

OpenVINS relies heavily on measurement consistency. TLEIO now logs innovation diagnostics, and ME004 shows very low chi-square ratios:

```text
median chi-square ratio around 0.052
95th percentile around 0.074
```

This suggests the effective measurement covariance may be conservative, the residual statistic may be under-sensitive, or adaptive covariance inflation is too strong. Before changing update logic, implement an offline calibration tool.

### Files

1. `scripts/filter_covariance_calibration.py`
2. `src/main_filter.py`
3. `tests/test_covariance_calibration.py`
4. `docs/tleio_openvins_3_implementation.md` after implementation.

### Implementation Steps

1. Add a script:

   ```bash
   python scripts/filter_covariance_calibration.py \
     --diagnostics outputs/main_filter/tartanair/competition_Test_ME004/update_diagnostics.csv
   ```

2. The script must read `update_diagnostics.csv`.
3. It must parse:
   - `residual_0` through `residual_11`
   - `sigma_0` through `sigma_11`
   - `chi2_ratio`
   - `accepted`
   - `rejected`
4. It must compute:
   - empirical residual standard deviation per axis.
   - mean predicted sigma per axis.
   - median predicted sigma per axis.
   - normalized residual RMS per axis.
   - normalized residual median absolute deviation per axis.
   - global normalized residual RMS.
   - median, p95, and max chi-square ratio.
5. It must group components by:
   - edge index `0..3`.
   - axis `x,y,z`.
6. It must write:

   ```text
   covariance_calibration_summary.csv
   covariance_calibration_summary.md
   ```

7. It must include a recommended scalar covariance multiplier:

   ```text
   recommended_meas_cov_scale_multiplier = normalized_residual_rms^2
   ```

   This recommendation must be diagnostic only. It must not automatically modify filter behavior.

8. It must support `--accepted_only` to ignore rejected updates.

### Tests

1. `test_covariance_calibration_script_runs_on_synthetic_csv`
2. `test_covariance_calibration_groups_edges_and_axes_correctly`
3. `test_covariance_calibration_recommends_scale_from_normalized_rms`
4. `test_covariance_calibration_handles_missing_rejections`

### Acceptance Criteria

1. The script runs on ME004 `update_diagnostics.csv`.
2. It produces both CSV and Markdown summaries.
3. It does not change the filter.
4. Its recommended scale is clearly documented as diagnostic.

## Phase 2: Configurable Measurement Covariance Scale Search

### Motivation

The largest improvement so far came from using regressed covariance. The next likely gain is calibrating how strongly the filter trusts that covariance. This phase adds an ablation driver for `meas_cov_scale` and optional per-axis scale factors.

### Files

1. `src/main_filter.py`
2. `scripts/filter_ablation_openvins3.py`
3. `tests/test_main_filter_inputs.py`
4. `tests/test_covariance_calibration.py`

### Implementation Steps

1. Keep existing global `meas_cov_scale`.
2. Add optional per-axis measurement covariance scaling:

   ```python
   meas_cov_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
   ```

3. Add CLI:

   ```bash
   --meas_cov_axis_scale sx sy sz
   ```

4. Apply the per-axis scale inside `_build_joint_covariance_for_window()` after regressed covariance is inserted:

   ```text
   sigma_x <- sigma_x * sx
   sigma_y <- sigma_y * sy
   sigma_z <- sigma_z * sz
   ```

   Equivalently, multiply variance by `sx^2`, `sy^2`, `sz^2`.

5. Validate:
   - exactly three scale values.
   - all finite.
   - all positive.
6. Add an ablation script that runs ME004 over a small grid:

   ```text
   meas_cov_scale in [0.25, 0.5, 0.75, 1.0, 1.25]
   ```

   using the OpenVINS-2 best mode:

   ```bash
   --imu_interval_mode paired_samples
   --nominal_integration_method midpoint
   ```

7. Save each run under:

   ```text
   outputs/comparison_ME004/openvins_3/cov_scale_grid/<run_name>/
   ```

8. Save summary:

   ```text
   outputs/comparison_ME004/openvins_3/cov_scale_grid/summary.csv
   ```

9. Summary columns must include:
   - run name.
   - command flags.
   - raw ATE.
   - SE3 ATE.
   - Sim3 ATE.
   - Sim3 scale.
   - rotation RMSE.
   - rejected update count.
   - median chi-square ratio.
   - p95 chi-square ratio.
   - mean correction norm.

### Tests

1. `test_axis_covariance_scale_changes_expected_diagonal_entries`
2. `test_axis_covariance_scale_rejects_non_positive_values`
3. `test_scale_grid_summary_has_required_columns`

### Acceptance Criteria

1. Default behavior is unchanged when axis scale is `(1,1,1)`.
2. A ME004 scale grid runs end to end.
3. No scale becomes default unless it improves ATE without a severe rotation or gating regression.

## Phase 3: Per-Edge Gating Diagnostics

### Motivation

OpenVINS gates measurements after residual construction. TLEIO currently gates the full 12D stacked residual. Since the learned measurement is four independent consecutive relative translations, a single bad edge can be hidden by the global statistic or can cause the whole update to be rejected.

This phase first adds diagnostics only.

### Files

1. `src/filter/scekf.py`
2. `src/main_filter.py`
3. `tests/test_scekf_update.py`
4. `tests/test_main_filter_inputs.py`

### Implementation Steps

1. Add a helper in `ImuMSCKF`:

   ```python
   _compute_edge_mahalanobis(residual, H, P, R) -> list[dict]
   ```

2. It must split the 12D residual into four 3D edge blocks:

   ```text
   edge 0: residual[0:3]
   edge 1: residual[3:6]
   edge 2: residual[6:9]
   edge 3: residual[9:12]
   ```

3. For each edge compute:

   ```text
   S_i = H_i P H_i.T + R_i
   maha_i = r_i.T solve(S_i, r_i)
   threshold_i = chi2.ppf(confidence, 3) * chi2_multiplier
   ratio_i = maha_i / threshold_i
   ```

4. Add these diagnostics to the update return dict:
   - `edge_mahalanobis_sq`
   - `edge_chi2_thresholds`
   - `edge_chi2_ratios`
5. Add CSV columns in `update_diagnostics.csv`:
   - `edge0_chi2_ratio`
   - `edge1_chi2_ratio`
   - `edge2_chi2_ratio`
   - `edge3_chi2_ratio`
   - `max_edge_chi2_ratio`
6. Do not change gating behavior in this phase.

### Tests

1. `test_edge_mahalanobis_returns_four_edges`
2. `test_edge_chi2_threshold_uses_3d_dimension`
3. `test_global_gate_behavior_unchanged_when_edge_diagnostics_enabled`
4. `test_update_diagnostics_csv_contains_edge_ratios`

### Acceptance Criteria

1. Default trajectory is unchanged.
2. ME004 diagnostics contain per-edge chi-square ratios.
3. We can identify whether any edge is much worse than the others.

## Phase 4: Per-Edge Robust Covariance Inflation

### Motivation

If per-edge diagnostics show occasional bad edges, the OpenVINS-inspired response is to reduce their influence. Since TLEIO must keep the same 12D measurement shape, the safest robust option is covariance inflation rather than dropping state variables or changing the measurement format.

### Files

1. `src/filter/scekf.py`
2. `src/main_filter.py`
3. `tests/test_scekf_update.py`

### Implementation Steps

1. Add config:

   ```python
   edge_robust_mode: str = "off"
   edge_inflation_factor: float = 100.0
   edge_chi2_multiplier: float = 1.0
   ```

2. Supported `edge_robust_mode` values:

   ```text
   "off"      current behavior
   "inflate"  inflate bad edge covariance blocks
   "reject"   reject whole update if any edge fails
   ```

3. Add CLI:

   ```bash
   --edge_robust_mode {off,inflate,reject}
   --edge_inflation_factor FLOAT
   --edge_chi2_multiplier FLOAT
   ```

4. For `inflate`:
   - compute per-edge chi-square ratios before the Kalman gain.
   - for each bad edge, multiply the corresponding `3 x 3` measurement covariance block by `edge_inflation_factor`.
   - recompute the update system after inflation.
   - do not change residual values.
5. For `reject`:
   - if any edge ratio exceeds its threshold, mark the update rejected.
   - return diagnostics showing which edge caused rejection.
6. Store diagnostics:
   - `num_inflated_edges`
   - `inflated_edge_indices`
   - `edge_robust_mode`
7. Keep global gating available after inflation.

### Tests

1. `test_edge_inflation_increases_selected_covariance_block`
2. `test_edge_inflation_reduces_bad_edge_kalman_gain_influence`
3. `test_edge_reject_mode_rejects_single_bad_edge`
4. `test_edge_robust_off_matches_current_update`

### Acceptance Criteria

1. `edge_robust_mode="off"` exactly preserves current behavior.
2. `inflate` and `reject` are opt-in.
3. ME004 ablation determines whether either mode is useful.
4. If no edge is bad on ME004, this phase should not improve ME004 and should remain off.

## Phase 5: Half-Step Midpoint Propagation Variant

### Motivation

The simple midpoint mode from OpenVINS pass 2 improved ME004 ATE but slightly worsened rotation RMSE. Its velocity/position update still uses the previous attitude. A more physically consistent variant uses a half-step attitude for the acceleration direction.

### Files

1. `src/filter/scekf.py`
2. `src/main_filter.py`
3. `tests/test_propagation.py`

### Implementation Steps

1. Extend `nominal_integration_method` supported values:

   ```text
   "euler"
   "midpoint"
   "midpoint_half_R"
   ```

2. Add CLI choice:

   ```bash
   --nominal_integration_method {euler,midpoint,midpoint_half_R}
   ```

3. For `midpoint_half_R`:

   ```text
   gyro_mid = 0.5 * (gyro0 + gyro1) - bg
   accel_mid = 0.5 * (accel0 + accel1) - ba
   R_half = R_prev @ exp(gyro_mid * dt * 0.5)
   R_next = R_prev @ exp(gyro_mid * dt)
   a_world = R_half @ accel_mid + g
   v_next = v_prev + a_world * dt
   p_next = p_prev + v_prev * dt + 0.5 * a_world * dt^2
   ```

4. Keep covariance propagation using the existing linearization at the same interval. Do not attempt a full RK4 covariance derivation in this phase.
5. Keep this mode available only on `ImuInterval` inputs:

   ```bash
   --imu_interval_mode paired_samples
   ```

### Tests

1. `test_midpoint_half_R_matches_midpoint_for_zero_gyro`
2. `test_midpoint_half_R_uses_half_step_rotation_for_acceleration`
3. `test_midpoint_half_R_requires_paired_intervals`
4. `test_midpoint_half_R_runs_me004_smoke`

### Acceptance Criteria

1. Existing `euler` and `midpoint` behavior is unchanged.
2. Synthetic tests prove half-step attitude affects acceleration direction.
3. ME004 ablation decides whether `midpoint_half_R` beats `midpoint`.

## Phase 6: Optional NEES/NIS Consistency Diagnostics

### Motivation

OpenVINS is consistency-oriented. TLEIO now logs NIS-like chi-square ratios, but does not log state-estimation consistency. Since ground truth is available for processed/evaluation sequences, add optional NEES-like diagnostics for position and rotation error.

### Files

1. `scripts/filter_diagnostics.py`
2. `src/main_filter.py`
3. `tests/test_filter_diagnostics.py`

### Implementation Steps

1. Add optional diagnostic output:

   ```text
   consistency_diagnostics.csv
   ```

2. For each saved trajectory state, compute:
   - position error.
   - rotation geodesic error.
3. If covariance snapshots are not available, do not call this true NEES.
4. Add optional covariance snapshot logging only if inexpensive:
   - current position covariance diagonal.
   - latest clone position covariance diagonal.
   - current attitude covariance diagonal.
5. Compute normalized position error using the relevant covariance diagonal only if available.
6. Clearly label this as approximate consistency, not full NEES, unless the exact state error and covariance block are used.

### Tests

1. `test_consistency_diagnostics_runs_without_covariance_snapshots`
2. `test_consistency_diagnostics_labels_approximate_nees`

### Acceptance Criteria

1. Diagnostics do not change filter behavior.
2. The naming is honest: approximate consistency metrics must not be reported as exact NEES.

## Phase 7: ME004 And Multi-Sequence Ablation Protocol

### Motivation

ME004 alone is not enough to promote a default. OpenVINS-inspired changes should be checked on at least one additional TartanAir or EDS sequence where processed data exists.

### Required ME004 Runs

Run at least:

1. OpenVINS-3 baseline:

   ```bash
   python src/main_filter.py \
     --dataset tartanair \
     --sequence competition_Test_ME004 \
     --imu_interval_mode paired_samples \
     --nominal_integration_method midpoint
   ```

2. best covariance scale from Phase 2.
3. per-edge diagnostics only.
4. per-edge `inflate`.
5. per-edge `reject`.
6. `midpoint_half_R`.
7. best combination of:
   - best covariance scale.
   - best edge robust mode.
   - best midpoint variant.

Save under:

```text
outputs/comparison_ME004/openvins_3/
```

### Recommended Additional Sequence Runs

If time allows, run the best candidate on:

```text
data/tartanair/processed_train/office_Easy_P000
data/tartanair/processed_train/office_Hard_P000
```

or any processed EDS sequence already known to run.

Save under:

```text
outputs/comparison_multiseq/openvins_3/
```

### Metrics

Use exactly the same metrics for every run:

1. raw ATE RMSE.
2. SE3 aligned ATE RMSE.
3. Sim3 scaled/aligned ATE RMSE.
4. Sim3 scale factor.
5. raw rotation RMSE.
6. max position error.
7. max rotation error.
8. rejected update count.
9. median chi-square ratio.
10. p95 chi-square ratio.
11. max chi-square ratio.
12. mean residual norm.
13. mean correction norm.

### Acceptance Criteria

1. A change is useful only if it improves ATE or consistency without a severe rotation or rejection-count regression.
2. A default may change only if it improves more than one sequence or is behavior-equivalent and numerically safer.
3. If a change improves ME004 but hurts other sequences, keep it behind a flag.

## Phase 8: Documentation Update

### Files

1. `docs/tleio_openvins_3_implementation.md`
2. `docs/tleio_filter_implementation_changes.md`
3. `docs/tleio_openvins_filter_comparison.md` if needed.

### Instructions

After implementation and evaluation, create:

```text
docs/tleio_openvins_3_implementation.md
```

It must include:

1. exact files changed.
2. new config fields and CLI flags.
3. which OpenVINS idea each change corresponds to.
4. tests added.
5. ME004 ablation summary.
6. multi-sequence summary if run.
7. which changes did not help.
8. final recommended command.
9. remaining risks.

Do not overwrite this plan file.

## Implementation Order

Follow this order exactly:

1. Phase 0: safety, baseline, and metric script.
2. Phase 1: covariance calibration script.
3. Phase 2: measurement covariance scale search.
4. Phase 3: per-edge gating diagnostics.
5. Phase 4: per-edge robust covariance inflation.
6. Phase 5: half-step midpoint propagation.
7. Phase 6: optional consistency diagnostics.
8. Phase 7: ablation protocol.
9. Phase 8: documentation update.

If a phase fails tests or worsens ME004 substantially, stop and document the failure before stacking additional behavior changes.

## Non-Goals

Do not implement:

1. visual residuals.
2. landmark states.
3. OpenVINS feature nullspace projection.
4. changing clone count.
5. changing the learned measurement output shape.
6. changing network training.
7. automatic parameter optimization that silently changes defaults.
8. committing generated output files to git.

## Expected Outcome

The expected gain is modest but meaningful. The most promising performance-changing items are:

1. calibrated measurement covariance scale.
2. half-step midpoint propagation.
3. robust per-edge inflation if bad learned edges exist.

The most promising debugging/consistency items are:

1. covariance calibration summaries.
2. per-edge chi-square diagnostics.
3. approximate consistency diagnostics.

This pass should leave TLEIO with a stronger OpenVINS-like update consistency workflow, not merely another set of flags.
