# TLEIO OpenVINS-Inspired Filter Implementation 3

This document explains the implementation of `docs/tleio_openvins_3_plan.md`.

The third pass kept TLEIO's problem definition fixed:

1. translation-only learned relative-motion measurements.
2. four consecutive 3D relative translations per update.
3. five pose clones.
4. no visual features, landmarks, nullspace projection, or network changes.

The work focused on OpenVINS-style consistency tooling and robust update handling.

## Files Changed

### `scripts/compute_ate_metrics.py`

Added a reusable trajectory metric script.

It:

1. skips non-numeric text headers.
2. infers timestamp units.
3. interpolates ground truth to estimate timestamps.
4. computes raw ATE RMSE.
5. computes SE3-aligned ATE RMSE.
6. computes Sim3 scaled/aligned ATE RMSE.
7. reports Sim3 scale.
8. reports raw and SE3-aligned rotation RMSE.

Example:

```bash
python scripts/compute_ate_metrics.py \
  --ground_truth data/tartanair/processed/competition_Test_ME004/anchor_poses.txt \
  --estimated outputs/comparison_ME004/openvins_3/baseline/stamped_traj_estimate.txt
```

### `scripts/filter_covariance_calibration.py`

Added an offline covariance calibration diagnostic script.

It reads `update_diagnostics.csv` and computes:

1. empirical residual standard deviation per edge and axis.
2. mean predicted sigma per edge and axis.
3. median predicted sigma per edge and axis.
4. normalized residual RMS per edge and axis.
5. normalized residual median absolute deviation per edge and axis.
6. global normalized residual RMS.
7. median, p95, and max chi-square ratio.
8. diagnostic recommended scalar multiplier:

   ```text
   recommended_meas_cov_scale_multiplier = normalized_residual_rms^2
   ```

This recommendation is diagnostic only. It does not modify the filter.

On the OpenVINS-3 baseline ME004 diagnostics, the script recommended:

```text
recommended_meas_cov_scale_multiplier = 1.849092923
```

Outputs:

```text
covariance_calibration_summary.csv
covariance_calibration_summary.md
```

### `scripts/filter_ablation_openvins3.py`

Added a small ablation driver for the planned covariance-scale grid.

Default grid:

```text
meas_cov_scale in [0.25, 0.5, 0.75, 1.0, 1.25]
```

The script runs ME004 using:

```bash
--imu_interval_mode paired_samples
--nominal_integration_method midpoint
```

and writes:

```text
outputs/comparison_ME004/openvins_3/cov_scale_grid/summary.csv
```

### `src/main_filter.py`

Added measurement covariance axis scaling:

```python
meas_cov_axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
```

New CLI:

```bash
--meas_cov_scale FLOAT
--meas_cov_axis_scale SX SY SZ
```

`meas_cov_axis_scale` scales regressed sigmas before the 12D covariance window is built. With `(1,1,1)`, behavior is unchanged.

Added edge robust config:

```python
edge_robust_mode: str = "off"
edge_inflation_factor: float = 100.0
edge_chi2_multiplier: float = 1.0
```

New CLI:

```bash
--edge_robust_mode {off,inflate,reject}
--edge_inflation_factor FLOAT
--edge_chi2_multiplier FLOAT
```

Added diagnostics columns to `update_diagnostics.csv`:

```text
edge_robust_mode
num_inflated_edges
inflated_edge_indices
edge_rejected
edge0_chi2_ratio
edge1_chi2_ratio
edge2_chi2_ratio
edge3_chi2_ratio
max_edge_chi2_ratio
```

Extended:

```bash
--nominal_integration_method {euler,midpoint,midpoint_half_R}
```

### `src/filter/scekf.py`

Added per-edge Mahalanobis diagnostics:

```python
_compute_edge_mahalanobis(residual, H, P, R)
```

The 12D update is split into four 3D edge blocks:

```text
edge 0: residual[0:3]
edge 1: residual[3:6]
edge 2: residual[6:9]
edge 3: residual[9:12]
```

For each edge:

```text
S_i = H_i P H_i.T + R_i
maha_i = r_i.T solve(S_i, r_i)
threshold_i = chi2.ppf(confidence, 3) * chi2_multiplier * edge_chi2_multiplier
ratio_i = maha_i / threshold_i
```

Added robust modes:

1. `off`
   - compute edge diagnostics only.
   - default behavior.
2. `inflate`
   - inflate the failed edge's `3 x 3` covariance block by `edge_inflation_factor`.
   - recompute the update system after inflation.
3. `reject`
   - reject the whole update if any individual edge fails its 3D gate.

Added `midpoint_half_R` propagation:

```text
gyro_mid = 0.5 * (gyro0 + gyro1) - bg
accel_mid = 0.5 * (accel0 + accel1) - ba
R_half = R_prev @ exp(gyro_mid * dt * 0.5)
R_next = R_prev @ exp(gyro_mid * dt)
a_world = R_half @ accel_mid + g
v_next = v_prev + a_world * dt
p_next = p_prev + v_prev * dt + 0.5 * a_world * dt^2
```

This mode is only available through paired IMU intervals.

### `scripts/filter_diagnostics.py`

Added approximate consistency diagnostics:

```text
consistency_diagnostics.csv
```

The CSV contains:

1. timestamp.
2. position error components.
3. position error norm.
4. rotation geodesic error.
5. `approximate_nees_available`.

This is not reported as true NEES because full covariance snapshots are not logged.

## Tests

Verification:

```bash
python -m pytest tests
```

Result:

```text
64 passed
```

New or expanded tests cover:

1. covariance calibration script on synthetic CSV.
2. edge/axis grouping in calibration summaries.
3. diagnostic scale recommendation.
4. covariance scale summary schema.
5. per-axis covariance scaling.
6. per-edge Mahalanobis diagnostics.
7. edge reject mode.
8. edge inflation mode.
9. robust mode `off` equivalence.
10. `midpoint_half_R` propagation behavior.
11. approximate consistency diagnostics.

## Baseline

The OpenVINS-3 baseline is the OpenVINS-2 best mode:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples \
  --nominal_integration_method midpoint
```

Saved under:

```text
outputs/comparison_ME004/openvins_3_baseline/
```

Baseline metrics:

| metric | value |
| --- | ---: |
| raw ATE RMSE m | 9.406739 |
| SE3 ATE RMSE m | 5.576098 |
| Sim3 ATE RMSE m | 5.575106 |
| Sim3 scale | 0.995154539 |
| raw rotation RMSE deg | 3.396047 |
| max position error m | 14.955012 |
| max rotation error deg | 5.709175 |
| rejected updates | 0 |
| median chi-square ratio | 0.051624 |
| p95 chi-square ratio | 0.073917 |
| max chi-square ratio | 0.141220 |

## ME004 Ablations

Summary:

```text
outputs/comparison_ME004/openvins_3/openvins_3_summary.csv
```

| run | raw ATE m | SE3 ATE m | Sim3 ATE m | rotation RMSE deg | rejected | max edge ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 9.406739 | 5.576098 | 5.575106 | 3.396047 | 0 | 0.269343 |
| cov scale 1.25 | 9.477178 | 5.594292 | 5.593256 | 3.392496 | 0 | 0.269246 |
| edge inflate | 9.406739 | 5.576098 | 5.575106 | 3.396047 | 0 | 0.269343 |
| edge reject | 9.406739 | 5.576098 | 5.575106 | 3.396047 | 0 | 0.269343 |
| midpoint_half_R | 9.516753 | 5.639296 | 5.637454 | 3.401782 | 0 | 0.269339 |
| best combination | 9.406739 | 5.576098 | 5.575106 | 3.396047 | 0 | 0.269343 |

## Interpretation

### Covariance Scale

The fixed grid did not beat the existing default `meas_cov_scale=1.2649054158337365`.

Best tested grid point:

```text
meas_cov_scale = 1.25
```

but it was slightly worse than the baseline:

```text
baseline raw ATE: 9.406739 m
scale 1.25 raw ATE: 9.477178 m
```

No covariance-scale default should change from this pass.

### Per-Edge Robust Modes

Per-edge diagnostics show ME004 has no bad individual edges:

```text
max_edge_chi2_ratio around 0.269
```

This is far below the rejection threshold. Therefore:

1. `edge_robust_mode=inflate` inflated zero edges.
2. `edge_robust_mode=reject` rejected zero updates.
3. both modes were behavior-equivalent to baseline on ME004.

They are useful robustness tools, but they should remain off by default.

### `midpoint_half_R`

The half-step attitude variant was worse on ME004:

```text
baseline raw ATE:        9.406739 m
midpoint_half_R raw ATE: 9.516753 m
```

It should remain behind the flag and not become default.

## Final Recommended Command

The best OpenVINS-3 command remains the OpenVINS-2 best command:

```bash
python src/main_filter.py \
  --dataset tartanair \
  --sequence competition_Test_ME004 \
  --imu_interval_mode paired_samples \
  --nominal_integration_method midpoint
```

## Remaining Risks

1. ME004 has very clean edge diagnostics, so robust edge handling was not stress-tested on real outliers.
2. Covariance calibration is sequence-specific; it should be repeated on more sequences before tuning defaults.
3. `midpoint_half_R` may help on more aggressive motion, but it did not help ME004.
4. Approximate consistency diagnostics are not full NEES without covariance snapshots.
