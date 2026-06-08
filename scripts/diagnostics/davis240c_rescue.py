from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import permutations, product
from pathlib import Path

import numpy as np


def load_table(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64, skiprows=1, ndmin=2)


def translation_rmse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))


def match_tables(
    pred_table: np.ndarray,
    gt_table: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gt_by_time = {
        (int(row[0]), int(row[1])): row
        for row in gt_table
    }
    matched_pred = []
    matched_gt = []
    for row in pred_table:
        gt_row = gt_by_time.get((int(row[0]), int(row[1])))
        if gt_row is not None:
            matched_pred.append(row)
            matched_gt.append(gt_row)

    if not matched_pred:
        raise ValueError("Prediction and GT files have no matching timestamp pairs.")
    return np.stack(matched_pred), np.stack(matched_gt)


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


def scalar_fit(x: np.ndarray, y: np.ndarray) -> float:
    denominator = float(np.sum(x * x))
    if denominator <= 1e-12:
        return 1.0
    return max(0.0, float(np.sum(x * y) / denominator))


def search_signed_permutations(
    pred: np.ndarray,
    gt: np.ndarray,
    calib_mask: np.ndarray,
) -> list[dict[str, object]]:
    results = []
    holdout_mask = ~calib_mask

    for perm in permutations(range(3)):
        for signs_tuple in product((-1, 1), repeat=3):
            signs = np.asarray(signs_tuple, dtype=np.float64)
            transformed = pred[:, perm] * signs
            scale = scalar_fit(transformed[calib_mask], gt[calib_mask])
            calibrated = transformed * scale
            results.append(
                {
                    "perm": perm,
                    "signs": signs_tuple,
                    "scale": scale,
                    "prediction": calibrated,
                    "calib_rmse": translation_rmse(
                        calibrated[calib_mask], gt[calib_mask]
                    ),
                    "holdout_rmse": (
                        translation_rmse(calibrated[holdout_mask], gt[holdout_mask])
                        if np.any(holdout_mask)
                        else float("nan")
                    ),
                    "full_rmse": translation_rmse(calibrated, gt),
                }
            )

    return sorted(results, key=lambda result: float(result["calib_rmse"]))


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
        default=None,
        help="Fraction of the sequence used to fit affine/diagonal corrections. 1.0 is oracle.",
    )
    parser.add_argument(
        "--calib_seconds",
        type=float,
        default=5.0,
        help=(
            "Initial duration used to fit signed-axis scale and optional "
            "calibrations. Defaults to the DAVIS benchmark's first 5 seconds."
        ),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of signed-axis candidates to print.",
    )
    args = parser.parse_args()

    pred_table = load_table(args.pred)
    gt_table = load_table(args.gt_rel)
    pred_table, gt_table = match_tables(pred_table, gt_table)
    n = len(pred_table)
    timestamps = pred_table[:, :2].astype(np.int64)
    gt = gt_table[:, 2:5]

    candidates: list[tuple[str, np.ndarray]] = []
    p = pred_table[:, 2:5]
    candidates.append(("input", p))
    candidates.append(("davis_axis", p[:, [1, 2, 0]]))

    if args.calib_fraction is not None:
        if not 0 < args.calib_fraction <= 1:
            raise ValueError("--calib_fraction must be in (0, 1].")
        calib_n = max(3, int(round(n * args.calib_fraction)))
        calib_mask = np.arange(n) < calib_n
        calib_description = f"first {args.calib_fraction:.3f} of matched rows"
    else:
        if args.calib_seconds <= 0:
            raise ValueError("--calib_seconds must be positive.")
        first_t0_us = int(timestamps[0, 0])
        calib_end_us = first_t0_us + int(round(args.calib_seconds * 1e6))
        calib_mask = timestamps[:, 1] <= calib_end_us
        if np.count_nonzero(calib_mask) < 3:
            raise ValueError("Calibration interval contains fewer than three motions.")
        calib_description = f"first {args.calib_seconds:g} seconds"

    for name, cand in list(candidates):
        scale, bias = diagonal_fit(cand[calib_mask], gt[calib_mask])
        candidates.append((f"{name}_diag_calib", cand * scale - bias))

        w, b = affine_fit(cand[calib_mask], gt[calib_mask])
        candidates.append((f"{name}_affine_calib", affine_apply(cand, w, b)))

    signed_axis_results = search_signed_permutations(p, gt, calib_mask)
    best = signed_axis_results[0]
    best_path = args.out_dir / "best_signed_axis_scaled.txt"
    save_rel(best_path, timestamps, best["prediction"])

    print(
        f"Matched rows: {n}; calibration: {calib_description} "
        f"({np.count_nonzero(calib_mask)} rows); holdout: "
        f"{np.count_nonzero(~calib_mask)} rows"
    )
    print("Top signed-axis candidates (mapping output=[input indices] * signs):")
    for rank, result in enumerate(signed_axis_results[: args.top_k], start=1):
        print(
            f"{rank:2d}. perm={result['perm']} signs={result['signs']} "
            f"scale={result['scale']:.8f} "
            f"calib_rmse={result['calib_rmse']:.6f} "
            f"holdout_rmse={result['holdout_rmse']:.6f} "
            f"full_rmse={result['full_rmse']:.6f}"
        )
    print(f"Saved best signed-axis candidate: {best_path}")

    if args.raw is not None:
        raw = load_table(args.raw)
        raw_ts, raw_feat = build_raw_edge_features(raw)
        gt_by_time = {(int(r[0]), int(r[1])): r[2:5] for r in gt_table}
        keep = np.array([tuple(ts) in gt_by_time for ts in raw_ts], dtype=bool)
        raw_ts = raw_ts[keep]
        raw_feat = raw_feat[keep]
        raw_gt = np.stack([gt_by_time[tuple(ts)] for ts in raw_ts], axis=0)
        if args.calib_fraction is not None:
            raw_calib_n = max(3, int(round(len(raw_feat) * args.calib_fraction)))
            raw_calib_mask = np.arange(len(raw_feat)) < raw_calib_n
        else:
            raw_calib_mask = raw_ts[:, 1] <= (
                int(raw_ts[0, 0]) + int(round(args.calib_seconds * 1e6))
            )

        w, b = affine_fit(raw_feat[raw_calib_mask], raw_gt[raw_calib_mask])
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
