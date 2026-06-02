from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_table(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64, skiprows=1, ndmin=2)


def translation_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))


def save_rel(path: Path, timestamps: np.ndarray, trans: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.column_stack([timestamps.astype(np.int64), trans.astype(np.float64)])
    np.savetxt(
        path,
        out,
        fmt=["%d", "%d", "%.10f", "%.10f", "%.10f"],
        header="t0_us t1_us px py pz",
        comments="",
    )


def affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_aug = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
    coeff, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    return coeff[: x.shape[1]], coeff[x.shape[1]]


def affine_apply(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b


def diagonal_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sum(x * y, axis=0) / np.maximum(np.sum(x * x, axis=0), 1e-12)
    bias = (x * scale - y).mean(axis=0)
    return scale, bias


def build_raw_edge_features(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_steps = raw.shape[1] // 5
    store: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)

    for row in raw:
        for step_idx in range(n_steps):
            base = step_idx * 5
            key = (int(row[base]), int(row[base + 1]))
            store[key].append(row[base + 2 : base + 5])

    timestamps = []
    features = []
    for key in sorted(store):
        vals = np.stack(store[key], axis=0)
        timestamps.append(key)
        features.append(
            np.concatenate(
                [
                    vals.mean(axis=0),
                    np.median(vals, axis=0),
                    vals.std(axis=0),
                    np.array([len(vals)], dtype=np.float64),
                ],
                axis=0,
            )
        )

    return np.asarray(timestamps, dtype=np.int64), np.asarray(features, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Try DAVIS-240C post-processing rescue candidates and oracle calibrations."
    )
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--gt_rel", type=Path, required=True)
    parser.add_argument("--raw", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--calib_fraction",
        type=float,
        default=1.0,
        help="Fraction of the sequence used to fit affine/diagonal corrections. 1.0 is oracle.",
    )
    args = parser.parse_args()

    pred_table = load_table(args.pred)
    gt_table = load_table(args.gt_rel)
    n = min(len(pred_table), len(gt_table))
    pred_table = pred_table[:n]
    gt_table = gt_table[:n]
    timestamps = pred_table[:, :2].astype(np.int64)
    gt = gt_table[:, 2:5]

    candidates: list[tuple[str, np.ndarray]] = []
    p = pred_table[:, 2:5]
    candidates.append(("input", p))
    candidates.append(("davis_axis", p[:, [1, 2, 0]]))

    calib_n = max(3, int(round(n * args.calib_fraction)))
    calib_slice = slice(0, calib_n)

    for name, cand in list(candidates):
        scale, bias = diagonal_fit(cand[calib_slice], gt[calib_slice])
        candidates.append((f"{name}_diag_calib", cand * scale - bias))

        w, b = affine_fit(cand[calib_slice], gt[calib_slice])
        candidates.append((f"{name}_affine_calib", affine_apply(cand, w, b)))

    if args.raw is not None:
        raw = load_table(args.raw)
        raw_ts, raw_feat = build_raw_edge_features(raw)
        gt_by_time = {(int(r[0]), int(r[1])): r[2:5] for r in gt_table}
        keep = np.array([tuple(ts) in gt_by_time for ts in raw_ts], dtype=bool)
        raw_ts = raw_ts[keep]
        raw_feat = raw_feat[keep]
        raw_gt = np.stack([gt_by_time[tuple(ts)] for ts in raw_ts], axis=0)
        raw_calib_n = max(3, int(round(len(raw_feat) * args.calib_fraction)))

        w, b = affine_fit(raw_feat[:raw_calib_n], raw_gt[:raw_calib_n])
        raw_pred = affine_apply(raw_feat, w, b)
        save_rel(args.out_dir / "raw_feature_affine_calib.txt", raw_ts, raw_pred)
        print(
            f"raw_feature_affine_calib rmse={translation_rmse(raw_pred, raw_gt):.6f} "
            f"mean_error={(raw_pred - raw_gt).mean(axis=0)} "
            f"saved={args.out_dir / 'raw_feature_affine_calib.txt'}"
        )

    for name, cand in candidates:
        out_path = args.out_dir / f"{name}.txt"
        save_rel(out_path, timestamps, cand)
        print(
            f"{name} rmse={translation_rmse(cand, gt):.6f} "
            f"mean_error={(cand - gt).mean(axis=0)} saved={out_path}"
        )


if __name__ == "__main__":
    main()
