import os
import logging
import argparse
import numpy as np

from filter.scekf import ImuMSCKF
from filter.imu_buffer import ImuBuffer
from filter.utils.logging import logging as filter_logging
from filter.utils.dotdict import dotdict

log = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser(description="TLEIO EKF filter runner")

    # Data
    parser.add_argument("--data_dir",      type=str, required=True)
    parser.add_argument("--out_dir",        type=str, default="output")
    parser.add_argument("--dataset",        type=str, default="tleio")

    # Filter
    parser.add_argument("--sigma_na",       type=float, default=0.01)
    parser.add_argument("--sigma_ng",       type=float, default=0.001)
    parser.add_argument("--sigma_nba",      type=float, default=1e-4)
    parser.add_argument("--sigma_nbg",      type=float, default=1e-5)

    # Network (Phase 2)
    parser.add_argument("--model_path",     type=str,   default=None,
                        help="Path to trained TLEIO network checkpoint")
    parser.add_argument("--window_time",    type=float, default=1.0,
                        help="Duration (s) of each event+IMU triplet window")

    # Misc
    parser.add_argument("--cpu",            action="store_true")
    parser.add_argument("--verbose",        action="store_true")
    return parser


def run_filter(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load network (Phase 2) ────────────────────────────────────────────────
    # TODO (Phase 2): replace with TLEIO transformer loader
    # network = load_tleio_network(args.model_path, device)
    network = None
    log.warning("Network not loaded — running propagation-only (Phase 1)")

    # ── Initialize filter ─────────────────────────────────────────────────────
    ekf = ImuMSCKF(args)

    imu_buffer = ImuBuffer()

    # ── Initialize state from data (placeholder) ──────────────────────────────
    R0  = np.eye(3)
    v0  = np.zeros(3)
    p0  = np.zeros(3)
    bg0 = np.zeros(3)
    ba0 = np.zeros(3)
    ekf.initialize_with_state(t=0.0, R=R0, v=v0, p=p0, bg=bg0, ba=ba0)

    # ── Main filter loop ───────────────────────────────────────────────────────
    # TODO: replace with actual data loader yielding (imu_batch, event_batch, t)
    data_iter = []  # placeholder

    results = []

    for step, (imu_batch, event_batch, t_meas) in enumerate(data_iter):

        # 1. Buffer incoming IMU
        for meas in imu_batch:
            imu_buffer.add(meas)

        # 2. Propagate up to measurement time
        imu_to_propagate = imu_buffer.get_up_to(t_meas)
        if len(imu_to_propagate) > 0:
            ekf.propagate(imu_to_propagate)

        # 3. Augment clone at measurement time
        ekf.augment_clone()

        # 4. Once 3 clones are accumulated, call update
        if ekf.state.get_clone_count() == 3:

            if network is not None:
                # TODO (Phase 2): run network, get 12D output + covariance
                # network_output = network(event_batch, imu_batch)
                # ekf.update(network_output)
                pass
            else:
                log.debug("Step %d: skipping update (no network)", step)

            # 5. Marginalize oldest clone
            ekf.marginalize_oldest_clone()

        # 6. Log result
        results.append({
            "t":  t_meas,
            "p":  ekf.state.p.copy(),
            "R":  ekf.state.R.copy(),
            "v":  ekf.state.v.copy(),
        })

    # ── Save results ───────────────────────────────────────────────────────────
    out_file = os.path.join(args.out_dir, "traj_estimate.txt")
    with open(out_file, "w") as f:
        for r in results:
            p = r["p"]
            f.write(f"{r['t']:.6f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    log.info("Saved trajectory to %s", out_file)

    return results


if __name__ == "__main__":
    parser = get_parser()
    args   = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run_filter(args)