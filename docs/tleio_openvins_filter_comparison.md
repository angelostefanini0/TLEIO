# TLEIO vs OpenVINS Filter Comparison

This report compares the current TLEIO filter implementation against the OpenVINS MSCKF implementation in `open_vins/`, with emphasis on what could improve TLEIO's filter behavior. It intentionally avoids high-level state-layout discussion except where it directly affects numerical consistency, observability, or robustness.

## Executive Summary

The most transferable OpenVINS ideas are not the visual frontend, feature state representation, or number of clones. They are the small numerical habits around propagation, linearization, update conditioning, and measurement validation:

1. Use exact update dimensionality for chi-square gating.
2. Improve IMU interval handling by interpolating to requested update times and rejecting zero or missing `dt`.
3. Revisit continuous-to-discrete IMU noise scaling and noise Jacobians.
4. Add FEJ-style fixed linearization points for clone residual Jacobians.
5. Use block/sparse update algebra over only touched variables.
6. Add measurement compression or whitening if TLEIO later stacks more learned constraints.
7. Treat covariance repair as a diagnostic, not as routine eigenvalue clipping.

The most immediate TLEIO fix is the gating mismatch: the update residual is 12D, but `CHI2_THRESHOLD = 21.026` is labelled as 95 percent for 6 DoF in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:61). A 95 percent chi-square threshold for 12 DoF is about 21.026, while 6 DoF is about 12.592. The value is numerically correct for the current 12D translation-only residual, but the comment is wrong and the threshold is hard-coded instead of being derived from `residual.shape[0]`.

## What TLEIO Currently Does

TLEIO propagates a 15D IMU error state plus pose clones. The nominal state is integrated sample by sample in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:124), using a third-order quaternion update for orientation and constant-acceleration updates for velocity and position. Covariance propagation uses a local 15x15 transition, embeds it into the current plus clone covariance, and adds IMU process noise to the IMU block in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:387).

The update is a learned relative-translation constraint over five clones. `main_filter.py` builds the five-clone sliding window, calls `ekf.update(...)`, then marginalizes the oldest clone in [src/main_filter.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/main_filter.py:551). `measurement_triplet.py` converts four consecutive 3D relative translations into a 12D stacked residual and Jacobian in [src/filter/measurement_triplet.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/measurement_triplet.py:164).

TLEIO already uses a few strong choices: Joseph covariance update in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:252), joint 12x12 learned covariance support in [src/filter/measurement_triplet.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/measurement_triplet.py:60), and adaptive measurement covariance inflation in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:301).

## What OpenVINS Does Differently That Matters

OpenVINS handles propagation as a bounded time interval problem, not just as a queue of pre-cut samples. `Propagator::select_imu_readings()` interpolates measurements at the exact start and end times, adds a final sample if the requested integration interval is not exactly reached, and removes zero-`dt` intervals in [open_vins/ov_msckf/src/state/Propagator.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/state/Propagator.cpp:269). TLEIO's `ImuBuffer.get_up_to()` simply pops queued samples with timestamps up to the query, so correctness depends on upstream code already creating exactly bounded intervals.

OpenVINS also accumulates the state transition and process noise across an update interval, then applies one covariance propagation through `StateHelper::EKFPropagation()` in [open_vins/ov_msckf/src/state/Propagator.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/state/Propagator.cpp:71). This is algebraically equivalent to repeated propagation when implemented exactly, but it centralizes covariance handling and makes the `Q` accumulation explicit.

OpenVINS' IMU discretization includes `dt` inside the noise Jacobian `G` and converts continuous densities to interval noise with terms like `sigma_w^2 / dt` and `sigma_wb^2 / dt` in [open_vins/ov_msckf/src/state/Propagator.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/state/Propagator.cpp:459). In TLEIO, `G` lacks the `dt` terms for gyro and accel noise and `Q_d = G Q_c G^T * dt` in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:404). That can be valid depending on whether `sigma_*` are continuous densities or per-sample standard deviations, but it should be made explicit and tested because it directly controls confidence growth.

OpenVINS uses FEJ linearization: residuals are computed at the current state, while Jacobians can be evaluated at stored first-estimate values to preserve observability. One example is the feature update switching to `Rot_fej()` and `pos_fej()` before computing Jacobians in [open_vins/ov_msckf/src/update/UpdaterHelper.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/update/UpdaterHelper.cpp:353). TLEIO currently computes its clone relative-translation Jacobians at the current corrected clone poses in [src/filter/measurement_triplet.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/measurement_triplet.py:120).

OpenVINS gates measurements with the residual dimension actually being tested. MSCKF initializes a chi-square table for many dimensions in [open_vins/ov_msckf/src/update/UpdaterMSCKF.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/update/UpdaterMSCKF.cpp:50), and gates each projected feature residual using `res.rows()` in [open_vins/ov_msckf/src/update/UpdaterMSCKF.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/update/UpdaterMSCKF.cpp:208). TLEIO has a single hard-coded threshold in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:232).

OpenVINS updates with only the covariance blocks touched by a measurement. `StateHelper::EKFUpdate()` builds `P H^T` using the supplied `H_order`, extracts a small marginal covariance, solves the innovation with LLT, and then updates the full covariance in [open_vins/ov_msckf/src/state/StateHelper.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/state/StateHelper.cpp:116). TLEIO forms full dense `H @ P @ H.T` and full `K` in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:228), which is fine at five clones but will matter if more learned windows or constraints are stacked.

## Recommended Changes For TLEIO

### 1. Dynamic Chi-Square Gating

Replace the hard-coded `CHI2_THRESHOLD` with a value derived from the residual dimension and desired confidence. This is a low-risk change and aligns directly with OpenVINS' pattern.

Recommended behavior:

```python
from scipy.stats import chi2

threshold = chi2.ppf(self.chi2_confidence, residual.shape[0])
if mahalanobis_sq > self.chi2_multiplier * threshold:
    reject
```

Also log the raw `mahalanobis_sq`, dimension, and threshold. This will make it obvious when the network covariance is systematically under- or over-confident.

### 2. Exact IMU Window Bounding

Port the spirit of `select_imu_readings()`: before propagation to each anchor time, ensure the IMU segment contains samples interpolated exactly at the previous and current anchor timestamps, and reject or warn on zero-`dt` samples.

This matters because a learned relative measurement constrains clone-to-clone motion at exact anchor times. If propagation ends slightly before or after those times, the update is compensating for timestamp error as if it were motion error.

### 3. Audit IMU Noise Units And Discretization

TLEIO should decide and document whether `sigma_ng`, `sigma_na`, `sigma_nbg`, and `sigma_nba` are continuous-time noise densities or already discrete per-sample noises. Then the propagation should match that choice.

If they are continuous densities, OpenVINS' construction is the safer template: put `dt` into the noise Jacobian and use the correct interval covariance in `Q_c`. If they are discrete noises, the current simpler scaling may be acceptable, but the config names should say so. Add a small stationary-IMU covariance-growth test for attitude, velocity, position, and biases.

### 4. Add FEJ For Clone Jacobians

Store first-estimate clone rotations and positions at clone creation. Compute the residual with current clone poses, but compute `H` using the stored clone poses. This is directly analogous to OpenVINS' `Rot_fej()` / `pos_fej()` handling and is one of the more likely improvements for consistency.

The candidate TLEIO location is [src/filter/measurement_triplet.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/measurement_triplet.py:120). The residual can still use current `clone_Rs` and `clone_ps`; only the Jacobian blocks should use the FEJ copies.

### 5. Stop Treating SPD Repair As Normal Operation

`enforce_symmetry_and_pos_def()` is used after augmentation, marginalization, propagation, and updates. Symmetry projection is good; silent positive-definite repair can hide a wrong Jacobian, wrong noise scaling, or overconfident update.

Borrow OpenVINS' diagnostic stance: check for negative diagonals and fail loudly in debug/test modes, as in [open_vins/ov_msckf/src/state/StateHelper.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/state/StateHelper.cpp:102). In production, keep tiny jitter only for numerically negligible eigenvalues and log when nontrivial repair occurs.

### 6. Use Small-Block Update Algebra

TLEIO's learned update touches only clone columns. It does not need to form a full dense Jacobian with the IMU columns all zero. A block update helper similar to OpenVINS' `H_order` approach would:

1. Build `H_small` over the involved clones only.
2. Extract `P_small` for those clones for innovation computation.
3. Build `P H^T` from the relevant covariance columns.
4. Apply the full correction to all correlated states.

This should produce the same result as the current dense math but makes future larger windows cheaper and reduces accidental dimension mistakes.

### 7. Consider Whitening Or Compression If Updates Grow

For the current 12D update, OpenVINS' QR compression is not necessary. But if TLEIO starts stacking multiple learned windows, relative pose heads, or auxiliary constraints, add measurement compression like `UpdaterHelper::measurement_compress_inplace()` in [open_vins/ov_msckf/src/update/UpdaterHelper.cpp](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/open_vins/ov_msckf/src/update/UpdaterHelper.cpp:345). If the learned covariance is full, whiten first with a Cholesky factor of `R`.

## TLEIO-Specific Issues Exposed By The Comparison

The docs and code disagree in `measurement_triplet.py`: comments still mention `2 x 3`, 6D residuals, and quaternion normalization, while the actual update is `4 x 3` translation-only and 12D in [src/filter/measurement_triplet.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/measurement_triplet.py:1). This is more than cosmetic because wrong residual dimension assumptions affect gating and covariance tuning.

`normalize_triplet_measurement()` is currently inconsistent with the active measurement shape. It tries to normalize `measurement[idx, 3:7]`, but the active measurement has only three columns in [src/filter/measurement_triplet.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/measurement_triplet.py:43). It is unused, so it should either be removed or updated when rotation measurements are added.

The update comment says `build_triplet_update()` returns the residual Jacobian and therefore TLEIO applies `delta_x = -K @ residual` in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:214). This is internally coherent only if every Jacobian block is indeed `d residual / d error`. Any future residual sign change should be guarded by a numerical Jacobian test.

`_sync_current_pose_with_latest_clone()` overwrites current IMU pose from the latest clone after every update in [src/filter/scekf.py](/Users/pietrovene/Desktop/ETHZ/Semester%202/3D%20Vision/TLEIO/src/filter/scekf.py:292). This is acceptable because the latest clone is at the current timestamp, but it should remain coupled with a test asserting clone/current timestamp equality and no velocity/bias side effect.

## Suggested Implementation Order

1. Fix residual-dimension comments, derive chi-square threshold dynamically, and add numerical Jacobian tests for the translation residual.
2. Add exact anchor-time IMU interpolation and zero-`dt` checks.
3. Audit and test IMU noise discretization.
4. Add FEJ clone storage and an option to toggle FEJ on/off for A/B evaluation.
5. Add diagnostic covariance checks before any positive-definite repair.
6. Refactor the update into a small-block helper if larger learned residuals are planned.

The first three changes are the best immediate return. FEJ is the most interesting OpenVINS-inspired algorithmic improvement, but it should come after the basic gating and propagation bookkeeping are correct, otherwise its effect will be hard to interpret.
