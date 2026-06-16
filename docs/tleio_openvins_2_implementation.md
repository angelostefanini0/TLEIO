# TLEIO OpenVINS-Inspired Filter Implementation 2

This document explains the changes implemented from `docs/tleio_openvins_2_plan.md`.

The goal of this second pass was not to change TLEIO's problem formulation. The filter still uses translation-only learned relative motions, the measurement remains four consecutive 3D relative translations, and the clone window size is unchanged. The implementation focuses only on transferable MSCKF infrastructure from OpenVINS: IMU interval propagation, covariance propagation structure, update conditioning, state-layout checks, FEJ invariants, and diagnostics.

## Summary Of Implemented Changes

Implemented mechanisms:

1. explicit pairwise IMU intervals.
2. optional OpenVINS-style summed IMU transition/noise propagation.
3. optional midpoint nominal IMU integration.
4. optional whitened measurement update.
5. stricter FEJ clone-list and covariance layout assertions.
6. per-update innovation and covariance diagnostics saved to CSV.
7. additional tests for propagation, update conditioning, and state layout.

The default runtime behavior remains conservative:

```text
imu_interval_mode = "sample_dt"
covariance_propagation_mode = "per_sample"
nominal_integration_method = "euler"
update_solve_method = "innovation"
gating_mode = "global"
fej_scope = "clone_update"
```

Therefore a plain run still uses the previous best configuration unless one of the new flags is explicitly passed.

## Files Changed

### `src/filter/imu_buffer.py`

Added `ImuInterval`:

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

This represents one propagation interval bounded by two IMU samples. It is the data structure used by the new OpenVINS-style propagation path.

The old `ImuMeasurement` path was kept intact.

### `src/main_filter.py`

Added exact IMU interval construction:

1. `_build_exact_imu_intervals()`
2. `_build_anchor_imu_intervals()`
3. `_propagate_anchor_segment()`

The interval builder:

1. includes the exact requested start time.
2. includes the exact requested end time.
3. includes raw IMU timestamps strictly inside the interval.
4. interpolates gyro and accel at the exact boundaries.
5. rejects non-finite values.
6. rejects duplicate or non-increasing timestamps.
7. rejects non-positive `dt`.

The dispatcher `_propagate_anchor_segment()` selects between:

```text
sample_dt       old `ImuMeasurement` path
paired_samples  new `ImuInterval` path
```

Added runner config fields and CLI flags:

```bash
--imu_interval_mode {sample_dt,paired_samples}
--covariance_propagation_mode {per_sample,summed}
--nominal_integration_method {euler,midpoint}
--update_solve_method {innovation,whitened,qr}
--gating_mode {global}
--fej_scope {clone_update}
```

Important guardrails:

1. `--covariance_propagation_mode summed` requires `--imu_interval_mode paired_samples`.
2. `--nominal_integration_method midpoint` requires `--imu_interval_mode paired_samples`.
3. `gating_mode` is currently only `global`.
4. `fej_scope` is currently only `clone_update`.

Added per-update diagnostics:

```text
outputs/main_filter/<dataset>/<sequence>/update_diagnostics.csv
```

The CSV contains:

1. anchor index and timestamp.
2. accepted/rejected flag.
3. Mahalanobis squared statistic.
4. chi-square threshold and ratio.
5. residual norm.
6. correction norm.
7. min/max/mean measurement sigma.
8. update solve method.
9. condition number of measurement covariance `R`.
10. condition number of innovation covariance `S`.
11. whitening flags.
12. all 12 residual components.
13. all 12 sigma components.

### `src/filter/scekf.py`

Added state layout helpers:

```python
State.clone_slice(clone_index)
State.expected_covariance_dim()
State.clone_count_from_covariance()
State.assert_covariance_shape_matches_clones()
State.assert_clone_fej_consistency()
```

These make the fixed TLEIO state layout explicit:

```text
15D current IMU error state
6D per pose clone
```

Assertions are now called after:

1. initialization.
2. clone augmentation.
3. update injection.
4. covariance update.
5. clone marginalization.

Added `ImuMSCKF.propagate_intervals()`.

This is the new pairwise IMU propagation path. It supports:

```text
nominal_integration_method = "euler"
nominal_integration_method = "midpoint"
covariance_propagation_mode = "per_sample"
covariance_propagation_mode = "summed"
```

For `euler`, it uses the end sample of the interval so it matches the old sample-dt behavior closely.

For `midpoint`, it uses:

```text
gyro_mid = 0.5 * (gyro0 + gyro1) - bg
accel_mid = 0.5 * (accel0 + accel1) - ba
```

The current midpoint implementation uses the previous attitude for velocity and position integration. It is intentionally simple and kept behind a flag.

Added covariance helpers:

```python
propagate_covariance_components(...)
apply_summed_imu_covariance(...)
```

`propagate_covariance_components()` returns one interval's IMU transition and process noise:

```text
Phi_i, Q_i
```

`apply_summed_imu_covariance()` applies an accumulated transition/noise pair to the full covariance, including current IMU covariance and IMU-clone cross-covariances.

The summed propagation mode accumulates:

```text
Phi_summed = Phi_i @ Phi_summed
Q_summed = Phi_i @ Q_summed @ Phi_i.T + Q_i
```

and then applies the accumulated result once at the end of the anchor-to-anchor propagation interval.

Added optional whitened update conditioning.

The default update remains:

```text
S = H P H.T + R
K = P H.T S^-1
```

With:

```bash
--update_solve_method whitened
```

the filter first computes a Cholesky whitening of `R`:

```text
L L.T = R
residual_w = solve(L, residual)
H_w = solve(L, H)
R_w = I
```

and then performs the EKF update using the whitened residual system. This is algebraically equivalent in well-conditioned cases, but it gives better diagnostics and a safer path for extreme regressed covariance values.

The `qr` mode is accepted by the CLI but currently falls back to the innovation solve. TLEIO's residual is only 12D and the state is larger, so there is no useful row compression to perform in this pass.

## Tests Added Or Expanded

Verification command:

```bash
python -m pytest tests
```

Result after implementation:

```text
46 passed
```

Expanded tests:

1. exact interval construction includes requested start and end.
2. exact interval construction rejects duplicate timestamps.
3. interval propagation matches the old sample-dt path for constant IMU.
4. midpoint matches euler for stationary constant IMU.
5. summed covariance matches per-sample propagation for a short constant sequence.
6. summed covariance preserves covariance shape and symmetry.
7. summed noise is symmetric positive semidefinite.
8. interval propagation rejects non-positive `dt`.
9. whitened update matches innovation update for identity `R`.
10. whitened update matches innovation update for diagonal `R`.
11. whitened update handles extreme regressed sigmas.
12. condition-number diagnostics are returned.
13. clone-slice helpers match the covariance layout.
14. covariance shape matches clone count after augmentation and marginalization.
15. marginalization preserves remaining clone cross-correlations.

New test file:

```text
tests/test_state_layout.py
```

## ME004 Ablations

Smoke ablations were saved under:

```text
outputs/comparison_ME004/openvins_2/
```

Summary file:

```text
outputs/comparison_ME004/openvins_2/openvins_2_summary.csv
```

Each run folder contains:

1. `stamped_traj_estimate.txt`
2. `update_diagnostics.csv`
3. trajectory plots.
4. `run.log`

Printed ME004 metrics:

| run | flags | position RMSE m | rotation RMSE deg | rejected updates |
| --- | --- | ---: | ---: | ---: |
| default | none | 9.571767 | 3.383130 | 0 |
| paired only | `--imu_interval_mode paired_samples` | 9.571767 | 3.383130 | 0 |
| paired + summed | `--imu_interval_mode paired_samples --covariance_propagation_mode summed` | 9.571767 | 3.383130 | 0 |
| paired + midpoint | `--imu_interval_mode paired_samples --nominal_integration_method midpoint` | 9.406739 | 3.396047 | 0 |
| whitened | `--update_solve_method whitened` | 9.571767 | 3.383130 | 0 |
| whitened + paired + summed | `--imu_interval_mode paired_samples --covariance_propagation_mode summed --update_solve_method whitened` | 9.571767 | 3.383130 | 0 |

Interpretation:

1. paired-only propagation is equivalent to the old path on ME004, as intended.
2. summed covariance propagation is equivalent on ME004, which is expected for the current linearized covariance model.
3. whitening is equivalent on ME004, meaning the current update is already well-conditioned for this sequence.
4. midpoint slightly improves raw position RMSE but slightly worsens rotation RMSE.

Because ME004 is only one sequence, midpoint was not made default.

## How To Run The New Modes

Default best-current behavior:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004
```

Paired IMU intervals:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples
```

Paired intervals with summed covariance:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples \
  --covariance_propagation_mode summed
```

Paired intervals with midpoint nominal integration:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples \
  --nominal_integration_method midpoint
```

Whitened update:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --update_solve_method whitened
```

Combined conditioning and summed propagation:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples \
  --covariance_propagation_mode summed \
  --update_solve_method whitened
```

## What Was Not Implemented

The following were deliberately not implemented as behavior-changing defaults:

1. visual-feature residuals.
2. landmark states.
3. OpenVINS feature nullspace projection.
4. changing the clone count.
5. changing the learned measurement format.
6. full propagation FEJ.
7. per-edge gating.
8. online adaptive covariance tuning beyond the existing adaptive covariance code already present in the filter.

Full propagation FEJ and per-edge gating remain possible future work, but ME004 diagnostics did not justify enabling them in this pass.

## Practical Conclusion

This pass makes the TLEIO filter infrastructure closer to OpenVINS in the areas that can transfer to a translation-only learned relative-motion filter:

1. clearer IMU interval propagation.
2. auditable transition/noise accumulation.
3. safer measurement conditioning.
4. stricter covariance and clone bookkeeping.
5. better diagnostics for innovation consistency and covariance calibration.

The immediate ME004 performance gain is small. The main practical value is that future tuning is now safer and easier to diagnose.
