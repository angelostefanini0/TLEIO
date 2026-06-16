# TLEIO Filter Implementation Changes

This document explains the implementation work done from `docs/tleio_filter_improvement_implementation_plan.md`. It is written as a detailed change log for future debugging and follow-up implementation.

## Summary

The filter now has the OpenVINS-inspired robustness features described in the plan:

1. Regressed diagonal covariance can be consumed from relative-motion files.
2. Chi-square gating is computed from the actual residual dimension.
3. The four-edge translation update is documented and tested as a 12D residual.
4. FEJ-style clone Jacobians are available behind a flag.
5. IMU propagation rejects invalid `dt` values and supports an optional continuous-time noise convention.
6. Covariance repair is centralized and diagnostic.
7. A block Kalman update path is implemented and tested against the dense path.
8. Unit tests were added for the new behavior.

The default behavior remains conservative:

1. `use_regressed_covariance=True`, but it only takes effect when the relative-motion file has sigma columns.
2. `use_fej=False`, so FEJ is available but not forced on yet.
3. `use_block_update=False`, so the dense update remains the default until real-sequence evaluation confirms equivalence.
4. `imu_noise_model="discrete"`, preserving the original process-noise convention by default.
5. `covariance_repair_mode="jitter"`, replacing silent ad hoc repair with logged diagnostics.

## Changed Files

### `src/main_filter.py`

The runner now exposes the filter-improvement switches through `RunnerConfig` and CLI flags.

New `RunnerConfig` fields:

```python
use_regressed_covariance: bool = True
min_regressed_sigma_m: float = 1e-4
max_regressed_sigma_m: float = 1.0
chi2_confidence: float = 0.95
chi2_multiplier: float = 1.0
enable_chi2_gating: bool = True
use_fej: bool = False
use_block_update: bool = False
covariance_repair_mode: str = "jitter"
imu_noise_model: str = "discrete"
```

New CLI flags:

```bash
--fixed_covariance
--use_fej
--block_update
--disable_chi2_gating
--covariance_repair_mode {strict,jitter,clip}
--imu_noise_model {discrete,continuous}
```

#### Relative-Motion Loading

`_load_relative_motion_table()` now explicitly accepts three file shapes:

1. `N x 5`: `t0 t1 px py pz`
2. `N x 8`: `t0 t1 px py pz sigma_x sigma_y sigma_z`
3. `N x 9`: `t0 t1 px py pz qx qy qz qw`

The current EKF still uses only translation measurements. The `N x 9` format is kept for legacy compatibility, but rotation columns are not used by the update.

#### Regressed Covariance

`_build_anchor_times_from_relative_motions()` now extracts sigma columns from `N x 8` files:

```python
relative_sigmas = relative_motion_table[:, 5:8]
```

The previous forced disable line:

```python
relative_sigmas = None
```

was removed from `run_filter()`.

`_sanitize_relative_sigmas()` was added to validate and clip learned sigmas. It rejects:

1. non-finite sigmas
2. negative sigmas
3. malformed shapes
4. invalid min/max clipping config

`_build_joint_covariance_for_window()` was added to build the 12x12 covariance for each four-edge update window. If regressed sigmas are available and enabled, it fills the 12 diagonal entries using four consecutive sigma rows:

```python
np.fill_diagonal(covariance, sigmas.reshape(-1) ** 2)
```

This means the first implementation supports learned diagonal covariance only. It does not assume the model predicts cross-correlations.

#### IMU Segment Hardening

`_build_exact_imu_segment()` now checks that:

1. requested times are inside the IMU stream
2. IMU values are finite
3. segment timestamps are finite
4. segment timestamps are strictly increasing
5. the final segment timestamp equals the requested end time
6. every generated `ImuMeasurement.dt` is positive
7. interpolated gyro and accel values are finite

`_build_anchor_imu_segments()` now rejects non-strictly-increasing IMU timestamps.

#### Runner Diagnostics

`run_filter()` now aggregates and returns:

```python
num_updates_accepted
mean_mahalanobis_sq
max_mahalanobis_sq
mean_chi2_ratio
max_chi2_ratio
used_regressed_covariance
use_fej
use_block_update
covariance_repair_events
```

These values are intended to support the evaluation protocol in the plan.

### `src/filter/measurement_triplet.py`

This module was cleaned up to reflect the actual current measurement model.

The old docstring described a `2 x 3` output, quaternion normalization, and a 6D residual. It now describes the real update:

1. input: `4 x 3` relative translations
2. residual: 12D
3. edges: `(0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4)`
4. covariance: one joint `12 x 12` matrix

The unused `normalize_triplet_measurement()` function was removed because it expected quaternion columns that are not present in the active measurement shape.

#### FEJ Hook

`build_pair_residual_and_local_jacobian()` now accepts optional Jacobian-only poses:

```python
jacobian_R_i=None
jacobian_p_i=None
jacobian_p_j=None
```

The residual is always evaluated at the current clone poses:

```python
residual = t_meas - t_hat
```

When FEJ is enabled, only the Jacobian blocks use first-estimate clone poses. This follows the OpenVINS principle without changing the measurement residual itself.

`build_triplet_update()` now accepts:

```python
use_fej=False
```

When enabled, it reads:

```python
state.clone_Rs_fej
state.clone_ps_fej
```

and uses those poses only for Jacobian construction.

### `src/filter/scekf.py`

This is the largest behavior change. The EKF now has configurable gating, FEJ clone storage, covariance diagnostics, optional block update algebra, and stricter propagation checks.

#### State FEJ Storage

`State` now stores first-estimate clone copies:

```python
self.clone_Rs_fej = []
self.clone_ps_fej = []
```

`initialize_with_state()` clears both current and FEJ clone lists.

`augment_clone()` appends the current pose to both:

```python
self.state.clone_Rs.append(...)
self.state.clone_ps.append(...)
self.state.clone_Rs_fej.append(...)
self.state.clone_ps_fej.append(...)
```

`_inject_error_state()` updates only the current clones. It intentionally does not update FEJ clone lists.

`marginalize_oldest_clone()` removes entries from both current and FEJ clone lists, keeping them aligned.

#### New Filter Flags

`ImuMSCKF.__init__()` now reads:

```python
imu_noise_model
chi2_confidence
chi2_multiplier
enable_chi2_gating
use_fej
use_block_update
covariance_repair_mode
covariance_repair_epsilon
```

These are passed from `RunnerConfig` through `_make_filter_args()`.

#### Dynamic Chi-Square Gating

The hard-coded `CHI2_THRESHOLD` constant was removed.

The threshold is now computed from the residual dimension:

```python
chi2.ppf(self.chi2_confidence, residual_dim)
```

and multiplied by:

```python
self.chi2_multiplier
```

For the current 12D residual and 95 percent confidence, the threshold is approximately `21.026`, matching the previous numeric value but now for the correct reason.

`update()` returns the new gating diagnostics:

```python
mahalanobis_sq
chi2_threshold
residual_dim
rejected
```

If `enable_chi2_gating=False`, these diagnostics are still computed but the update is not rejected.

#### Dense And Block Update Paths

Two helper methods were added:

```python
_compute_dense_kalman_gain(P, H, R)
_compute_block_kalman_gain(P, H, R)
```

The dense path preserves the old algebra:

```python
S = H @ P @ H.T + R
K = P @ H.T @ inv(S)
```

The block path finds the nonzero state columns in `H` and computes:

```python
H_small = H[:, touched_columns]
P_small = P[np.ix_(touched_columns, touched_columns)]
S = H_small @ P_small @ H_small.T + R
PHt = P[:, touched_columns] @ H_small.T
```

This follows the OpenVINS idea of updating with only the variables touched by a measurement while still applying the correction to the full correlated state.

The block path is controlled by:

```python
use_block_update
```

and remains off by default.

#### Covariance Repair Diagnostics

`ImuMSCKF` now stores:

```python
self.covariance_diagnostics = []
```

Covariance repair is applied through:

```python
self._repair_covariance(P, checkpoint)
```

and checkpoints are named:

1. `initialize`
2. `propagate`
3. `augment_clone`
4. `update`
5. `marginalize_oldest_clone`

This makes covariance repair observable instead of silent.

#### IMU Propagation Validation

`propagate()` now rejects non-finite or non-positive `dt` before integrating:

```python
if not np.isfinite(dt) or dt <= 0.0:
    raise ValueError(...)
```

#### Process Noise Model

`propagate_covariance()` now accepts:

```python
imu_noise_model="discrete"
```

Two conventions are supported:

1. `"discrete"` preserves the original TLEIO behavior. The IMU sigmas are interpreted as per-sample values.
2. `"continuous"` treats sigmas as continuous-time densities and includes interval factors in the noise Jacobian. This also injects same-step accelerometer noise into position uncertainty.

The default is `"discrete"` to avoid changing trajectory behavior before real-sequence evaluation.

### `src/filter/utils/math_utils.py`

Covariance handling was centralized.

#### `covariance_diagnostics()`

Returns:

```python
name
finite
symmetry_error
min_diagonal
min_eigenvalue
repair_applied
repair_amount
repair_mode
```

#### `repair_covariance()`

Supports three modes:

1. `"strict"`: raises on significant negative diagonal/eigenvalue or non-finite values.
2. `"jitter"`: symmetrizes and adds diagonal jitter only if the minimum eigenvalue is below `epsilon`.
3. `"clip"`: eigenvalue-clips to `epsilon`.

#### `enforce_symmetry_and_pos_def()`

This function remains for backward compatibility, but now delegates to `repair_covariance(..., mode="jitter")`.

## Added Tests

A new `tests/` directory was added.

### `tests/conftest.py`

Adds the project root and `src/` to `sys.path` so tests can import both `src.main_filter` and `filter.*` modules.

### `tests/test_main_filter_inputs.py`

Covers runner and input handling:

1. `N x 5` and `N x 8` relative-motion loading.
2. extraction of sigma columns from `N x 8` files.
3. construction of a 12D covariance diagonal from four regressed sigma rows.
4. rejection of non-finite and negative sigmas.
5. disabling regressed covariance with the config flag.
6. exact IMU segment end-time interpolation.
7. rejection of duplicate/non-increasing IMU timestamps.

### `tests/test_measurement_triplet.py`

Covers residual and Jacobian correctness:

1. perfect measurements produce zero residual.
2. analytical triplet Jacobians match finite differences.
3. the negative Kalman step sign reduces residual in the linearized system.
4. FEJ changes the Jacobian after clone correction but leaves the residual unchanged.

### `tests/test_propagation.py`

Covers propagation safety:

1. stationary IMU nominal state remains stationary with consistent gravity.
2. continuous process-noise covariance grows monotonically.
3. propagated covariance remains symmetric.
4. non-positive propagation `dt` is rejected.

### `tests/test_scekf_update.py`

Covers EKF behavior:

1. dynamic chi-square threshold matches 12D residual expectation.
2. large residuals are rejected when gating is enabled.
3. gating can be disabled while diagnostics remain available.
4. update return dictionaries contain gating diagnostics.
5. FEJ clone values are created during augmentation.
6. FEJ clone values do not change during error injection.
7. marginalization keeps FEJ and current clone lists aligned.
8. covariance strict mode rejects clearly invalid covariance.
9. jitter mode repairs tiny negative eigenvalues.
10. nominal update does not require nontrivial covariance repair.
11. block update matches dense update for random PSD covariance.
12. block update matches dense update for the real triplet Jacobian shape.

## Verification Performed

The new tests pass:

```bash
python -m pytest tests
```

Result:

```text
27 passed
```

The edited Python modules also compile:

```bash
python -m py_compile \
  src/filter/scekf.py \
  src/filter/measurement_triplet.py \
  src/main_filter.py \
  src/filter/utils/math_utils.py
```

The default baseline run was attempted:

```bash
python src/main_filter.py
```

It could not run because the default processed sequence folder does not exist locally:

```text
data/eds/processed/00_peanuts_dark
```

This means real-sequence evaluation still needs to be run with an available processed sequence, for example one under `data/eds/processed_train/`.

## Important Behavioral Notes

### Regressed Covariance Is Diagonal Only

The current implementation consumes `sigma_x sigma_y sigma_z` per relative edge. It does not consume a full learned 12x12 covariance. If the network later outputs cross-correlations, the measurement file format and covariance builder should be extended deliberately.

### FEJ Is Available But Not Default

`use_fej=False` by default. This is intentional. FEJ should be compared against the baseline using the evaluation protocol before becoming default.

### Block Update Is Available But Not Default

`use_block_update=False` by default. Unit tests show equivalence to the dense update, but real-sequence diagnostics should confirm there are no unexpected numerical differences before enabling it globally.

### IMU Noise Convention Is Preserved By Default

`imu_noise_model="discrete"` preserves the previous covariance propagation interpretation. The OpenVINS-inspired continuous model is available through `imu_noise_model="continuous"` and should be evaluated separately.

### Covariance Repair Is Now Observable

Covariance repair events are collected and returned by `run_filter()` as `covariance_repair_events`. This is useful for identifying whether a change improves trajectory metrics by hiding covariance inconsistency.

## Suggested Next Evaluation

Run the same available sequence in four modes:

```bash
python src/main_filter.py --dataset eds --sequence <sequence> --fixed_covariance
python src/main_filter.py --dataset eds --sequence <sequence> --fixed_covariance --use_fej
python src/main_filter.py --dataset eds --sequence <sequence>
python src/main_filter.py --dataset eds --sequence <sequence> --use_fej
```

Then repeat with:

```bash
--block_update
--imu_noise_model continuous
--covariance_repair_mode strict
```

only after the first four runs are stable.

The quantities to compare are:

1. ATE/RPE from `filter_diagnostics`
2. rejected update count
3. mean/max Mahalanobis distance
4. mean/max chi-square ratio
5. mean residual norm
6. mean correction norm
7. covariance repair events

## Second OpenVINS-Inspired Pass

Implemented from `docs/tleio_openvins_2_plan.md`.

### Files Changed

1. `src/filter/imu_buffer.py`
   - added `ImuInterval`, an explicit pairwise IMU interval with `t0`, `t1`, endpoint accel/gyro samples, and a computed `dt`.
2. `src/main_filter.py`
   - added exact interval construction beside the existing exact sample-dt segment construction.
   - added runner flags for interval propagation, summed covariance propagation, midpoint nominal integration, and conditioned update solves.
   - added per-update innovation and covariance diagnostics saved to `update_diagnostics.csv`.
3. `src/filter/scekf.py`
   - added explicit state-layout helpers and assertions.
   - added `propagate_intervals()` for OpenVINS-style pairwise IMU propagation.
   - split one-step covariance propagation into reusable `(Phi_i, Q_i)` helpers.
   - added summed transition/noise application over the current IMU block and clone cross-covariances.
   - added optional whitened measurement update conditioning.
4. Tests:
   - expanded `tests/test_main_filter_inputs.py`, `tests/test_propagation.py`, and `tests/test_scekf_update.py`.
   - added `tests/test_state_layout.py`.

### New Config Fields And CLI Flags

Current defaults remain conservative and reproduce the previous best behavior:

```text
imu_interval_mode = "sample_dt"
covariance_propagation_mode = "per_sample"
nominal_integration_method = "euler"
update_solve_method = "innovation"
gating_mode = "global"
fej_scope = "clone_update"
```

New CLI flags:

```bash
--imu_interval_mode {sample_dt,paired_samples}
--covariance_propagation_mode {per_sample,summed}
--nominal_integration_method {euler,midpoint}
--update_solve_method {innovation,whitened,qr}
--gating_mode {global}
--fej_scope {clone_update}
```

`summed` covariance propagation and `midpoint` nominal integration require `--imu_interval_mode paired_samples`; the runner raises a `ValueError` if those options are requested with the old sample-dt path.

### OpenVINS Mechanisms Transferred

1. Pairwise IMU propagation intervals, inspired by OpenVINS IMU reading selection and interval propagation.
2. Summed IMU transition/noise accumulation:

   ```text
   Phi = Phi_i @ Phi
   Q = Phi_i @ Q @ Phi_i.T + Q_i
   ```

   followed by one full covariance application to the current IMU block and its clone cross-covariances.
3. Optional midpoint nominal integration over paired IMU endpoints.
4. Whitened measurement update conditioning before computing the Kalman gain.
5. Explicit state-layout and FEJ clone-list consistency checks, similar in spirit to OpenVINS' strict state/covariance bookkeeping.

### Diagnostics Added

Each `main_filter.py` run now writes:

```text
outputs/main_filter/<dataset>/<sequence>/update_diagnostics.csv
```

The CSV includes:

1. update timestamp and anchor index.
2. accepted/rejected flag.
3. Mahalanobis squared value, chi-square threshold, and ratio.
4. residual norm and correction norm.
5. min/max/mean measurement sigma.
6. update solve method and condition numbers for `R` and `S`.
7. whitening flags.
8. all 12 residual components and all 12 sigma components.

### Tests

Verification command:

```bash
python -m pytest tests
```

Result after this pass:

```text
46 passed
```

### ME004 Smoke Ablations

Saved under:

```text
outputs/comparison_ME004/openvins_2/
```

Each run folder contains:

1. `stamped_traj_estimate.txt`
2. `update_diagnostics.csv`
3. trajectory and rotation plots
4. `run.log`

Summary file:

```text
outputs/comparison_ME004/openvins_2/openvins_2_summary.csv
```

Printed RMSE summary:

| run | position RMSE m | rotation RMSE deg | rejected updates | note |
| --- | ---: | ---: | ---: | --- |
| default | 9.571767 | 3.383130 | 0 | previous best defaults |
| paired only | 9.571767 | 3.383130 | 0 | equivalent to default |
| paired + summed | 9.571767 | 3.383130 | 0 | equivalent to default on ME004 |
| paired + midpoint | 9.406739 | 3.396047 | 0 | slightly better position, slightly worse rotation |
| whitened | 9.571767 | 3.383130 | 0 | equivalent to default in this well-conditioned run |
| whitened + paired + summed | 9.571767 | 3.383130 | 0 | equivalent to default |

### Known Non-Improvements

1. Whitening is numerically safer for ill-conditioned measurement covariance, but it is algebraically equivalent to the innovation solve in the current well-conditioned ME004 run.
2. Summed covariance propagation is algebraically equivalent to per-interval covariance application for the current linearized one-step model, so no ME004 change is expected unless numerical conditioning becomes an issue.
3. Full propagation FEJ and per-edge gating were deliberately not implemented as behavior-changing defaults. The plan allowed them only if diagnostics showed a need; ME004 diagnostics still show no rejected updates and very low chi-square ratios.

### Remaining Risks

1. ME004 is only one sequence; midpoint should not become default from this result alone.
2. The current midpoint implementation uses average endpoint accel/gyro and previous attitude for velocity/position integration. This is intentionally simple and should be compared on more sequences before changing defaults.
3. `qr` update mode is exposed but currently falls back to the innovation solve because TLEIO's 12D residual is not overdetermined relative to the state.

## Third OpenVINS-Inspired Pass

Implemented from `docs/tleio_openvins_3_plan.md`.

Added:

1. `scripts/compute_ate_metrics.py` for reproducible raw, SE3, and Sim3 ATE metrics.
2. `scripts/filter_covariance_calibration.py` for offline covariance calibration from `update_diagnostics.csv`.
3. `scripts/filter_ablation_openvins3.py` for ME004 covariance-scale grids.
4. `--meas_cov_scale` and `--meas_cov_axis_scale SX SY SZ`.
5. per-edge chi-square diagnostics in `update_diagnostics.csv`.
6. `--edge_robust_mode {off,inflate,reject}` plus edge inflation controls.
7. `--nominal_integration_method midpoint_half_R`.
8. approximate consistency diagnostics saved as `consistency_diagnostics.csv`.

Verification:

```bash
python -m pytest tests
```

Result:

```text
64 passed
```

ME004 summary:

```text
outputs/comparison_ME004/openvins_3/openvins_3_summary.csv
```

The OpenVINS-3 changes did not beat the OpenVINS-2 best command on ME004. The recommended command remains:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples \
  --nominal_integration_method midpoint
```

Important findings:

1. covariance scale grid did not beat the existing default `meas_cov_scale`.
2. per-edge robust modes stayed inactive because ME004 max edge chi-square ratio was only about `0.269`.
3. `midpoint_half_R` was worse than the simpler midpoint mode on ME004.
4. the new tools are still useful for diagnosing and stress-testing less clean sequences.
