from pathlib import Path
import json
import csv
import math
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from scipy.stats import ks_2samp, wasserstein_distance
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


TARTAN_ROOT = Path("data/tartanair/precomputed_test")
DAVIS_ROOT  = Path("data/davis240c/precomputed_checkpoint_compatible")
OUT_ROOT    = Path("outputs/distribution_analysis/tartan_vs_davis")
VOXEL_NAME  = "derotated_voxels.npy"

MAX_FRAMES_PER_SEQUENCE = 3000
RNG_SEED = 7
EPS = 1e-12

OUT_ROOT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(RNG_SEED)


def find_voxel_file(seq_dir: Path):
    preferred = seq_dir / VOXEL_NAME
    if preferred.exists():
        return preferred

    candidates = list(seq_dir.glob("*.npy"))
    if not candidates:
        return None

    # Prefer files with voxel in name.
    voxel_candidates = [p for p in candidates if "voxel" in p.name.lower()]
    if voxel_candidates:
        return sorted(voxel_candidates)[0]

    return sorted(candidates)[0]


def sample_indices(n, max_n):
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def summarize_array(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            "mean": np.nan, "std": np.nan,
            "p01": np.nan, "p05": np.nan, "p25": np.nan,
            "p50": np.nan, "p75": np.nan, "p95": np.nan, "p99": np.nan,
            "min": np.nan, "max": np.nan,
        }

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "p01": float(np.percentile(x, 1)),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def compute_frame_metrics(voxels, indices):
    """
    voxels shape expected [N, B, H, W], usually [N, 5, 336, 448].
    Computes one row of metrics per sampled voxel frame.
    """
    metrics = {
        "nnz_frac": [],
        "pos_frac": [],
        "neg_frac": [],
        "abs_mean": [],
        "abs_sum": [],
        "pos_sum": [],
        "neg_sum_abs": [],
        "signed_sum": [],
        "polarity_balance": [],
        "x_centroid": [],
        "y_centroid": [],
        "temporal_centroid": [],
    }

    bin_abs_sums = None

    for idx in indices:
        v = np.asarray(voxels[idx], dtype=np.float32)  # [B,H,W]
        B, H, W = v.shape
        total = float(B * H * W)

        abs_v = np.abs(v)
        pos = np.maximum(v, 0.0)
        neg = np.maximum(-v, 0.0)

        nnz = np.count_nonzero(v)
        pos_nnz = np.count_nonzero(v > 0)
        neg_nnz = np.count_nonzero(v < 0)

        abs_sum = float(abs_v.sum())
        pos_sum = float(pos.sum())
        neg_sum_abs = float(neg.sum())
        signed_sum = float(v.sum())

        metrics["nnz_frac"].append(nnz / total)
        metrics["pos_frac"].append(pos_nnz / total)
        metrics["neg_frac"].append(neg_nnz / total)
        metrics["abs_mean"].append(abs_sum / total)
        metrics["abs_sum"].append(abs_sum)
        metrics["pos_sum"].append(pos_sum)
        metrics["neg_sum_abs"].append(neg_sum_abs)
        metrics["signed_sum"].append(signed_sum)
        metrics["polarity_balance"].append((pos_sum - neg_sum_abs) / (pos_sum + neg_sum_abs + EPS))

        # Spatial centroid of absolute event mass, normalized to [0,1].
        if abs_sum > EPS:
            mass_hw = abs_v.sum(axis=0)  # [H,W]
            xs = np.arange(W, dtype=np.float32)
            ys = np.arange(H, dtype=np.float32)
            x_c = float((mass_hw.sum(axis=0) * xs).sum() / (abs_sum * max(W - 1, 1)))
            y_c = float((mass_hw.sum(axis=1) * ys).sum() / (abs_sum * max(H - 1, 1)))

            bin_mass = abs_v.reshape(B, -1).sum(axis=1)
            bs = np.arange(B, dtype=np.float32)
            t_c = float((bin_mass * bs).sum() / ((bin_mass.sum() + EPS) * max(B - 1, 1)))
        else:
            x_c = np.nan
            y_c = np.nan
            t_c = np.nan

        metrics["x_centroid"].append(x_c)
        metrics["y_centroid"].append(y_c)
        metrics["temporal_centroid"].append(t_c)

        this_bin_abs = abs_v.reshape(B, -1).sum(axis=1).astype(float)
        if bin_abs_sums is None:
            bin_abs_sums = [[] for _ in range(B)]
        for b in range(B):
            bin_abs_sums[b].append(float(this_bin_abs[b]))

    if bin_abs_sums is not None:
        for b, vals in enumerate(bin_abs_sums):
            metrics[f"bin{b}_abs_sum"] = vals

    return metrics


def collect_dataset(dataset_name, root):
    print(f"\n=== Dataset: {dataset_name} ===")
    print("root:", root)

    all_frame_metrics = {}
    sequence_rows = []

    seq_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not seq_dirs:
        raise RuntimeError(f"No sequence directories found in {root}")

    for seq_dir in seq_dirs:
        voxel_file = find_voxel_file(seq_dir)
        if voxel_file is None:
            print(f"SKIP {seq_dir.name}: no npy voxel file")
            continue

        vox = np.load(voxel_file, mmap_mode="r")
        if vox.ndim != 4:
            print(f"SKIP {seq_dir.name}: expected 4D [N,B,H,W], got {vox.shape}")
            continue

        N, B, H, W = vox.shape
        idx = sample_indices(N, MAX_FRAMES_PER_SEQUENCE)
        m = compute_frame_metrics(vox, idx)

        for k, vals in m.items():
            all_frame_metrics.setdefault(k, []).extend(vals)

        row = {
            "dataset": dataset_name,
            "sequence": seq_dir.name,
            "voxel_file": str(voxel_file),
            "N_total": int(N),
            "N_sampled": int(len(idx)),
            "B": int(B),
            "H": int(H),
            "W": int(W),
        }

        for key in ["nnz_frac", "pos_frac", "neg_frac", "abs_mean", "abs_sum",
                    "pos_sum", "neg_sum_abs", "polarity_balance",
                    "x_centroid", "y_centroid", "temporal_centroid"]:
            s = summarize_array(m[key])
            row[f"{key}_mean"] = s["mean"]
            row[f"{key}_p50"] = s["p50"]
            row[f"{key}_p95"] = s["p95"]

        for b in range(B):
            key = f"bin{b}_abs_sum"
            if key in m:
                s = summarize_array(m[key])
                row[f"{key}_mean"] = s["mean"]
                row[f"{key}_p50"] = s["p50"]

        sequence_rows.append(row)

        print(
            f"{dataset_name:10s} {seq_dir.name:35s} "
            f"N={N:6d} sampled={len(idx):5d} shape=[{B},{H},{W}] "
            f"nnz={row['nnz_frac_mean']:.6e} "
            f"abs_mean={row['abs_mean_mean']:.6e} "
            f"pol_bal={row['polarity_balance_mean']:+.4f}"
        )

    return all_frame_metrics, sequence_rows


def write_sequence_csv(rows, path):
    if not rows:
        return
    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_global_summary(dataset_metrics, path):
    out = {}
    for dataset, metrics in dataset_metrics.items():
        out[dataset] = {}
        for k, vals in metrics.items():
            out[dataset][k] = summarize_array(vals)

    path.write_text(json.dumps(out, indent=2))
    return out


def write_distribution_tests(tartan_m, davis_m, path):
    rows = []
    common = sorted(set(tartan_m.keys()) & set(davis_m.keys()))

    for k in common:
        a = np.asarray(tartan_m[k], dtype=float)
        b = np.asarray(davis_m[k], dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]

        row = {
            "metric": k,
            "n_tartan": int(a.size),
            "n_davis": int(b.size),
            "tartan_mean": float(np.mean(a)) if a.size else np.nan,
            "davis_mean": float(np.mean(b)) if b.size else np.nan,
            "ratio_davis_over_tartan_mean": float((np.mean(b) + EPS) / (np.mean(a) + EPS)) if a.size and b.size else np.nan,
            "delta_mean_davis_minus_tartan": float(np.mean(b) - np.mean(a)) if a.size and b.size else np.nan,
            "tartan_p50": float(np.percentile(a, 50)) if a.size else np.nan,
            "davis_p50": float(np.percentile(b, 50)) if b.size else np.nan,
            "ratio_davis_over_tartan_p50": float((np.percentile(b, 50) + EPS) / (np.percentile(a, 50) + EPS)) if a.size and b.size else np.nan,
        }

        if HAS_SCIPY and a.size and b.size:
            ks = ks_2samp(a, b)
            row["ks_stat"] = float(ks.statistic)
            row["ks_pvalue"] = float(ks.pvalue)
            row["wasserstein"] = float(wasserstein_distance(a, b))
        else:
            row["ks_stat"] = np.nan
            row["ks_pvalue"] = np.nan
            row["wasserstein"] = np.nan

        rows.append(row)

    keys = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    return rows


def plot_metric(tartan_m, davis_m, metric, out_dir):
    if not HAS_MPL:
        return

    a = np.asarray(tartan_m.get(metric, []), dtype=float)
    b = np.asarray(davis_m.get(metric, []), dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if a.size == 0 or b.size == 0:
        return

    # Downsample for plotting.
    max_plot = 50000
    if a.size > max_plot:
        a = rng.choice(a, size=max_plot, replace=False)
    if b.size > max_plot:
        b = rng.choice(b, size=max_plot, replace=False)

    plt.figure(figsize=(8, 5))
    plt.hist(a, bins=80, alpha=0.55, density=True, label="TartanEvent")
    plt.hist(b, bins=80, alpha=0.55, density=True, label="DAVIS240C")
    plt.xlabel(metric)
    plt.ylabel("density")
    plt.title(f"Distribution comparison: {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_{metric}.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.boxplot([a, b], labels=["TartanEvent", "DAVIS240C"], showfliers=False)
    plt.ylabel(metric)
    plt.title(f"Boxplot: {metric}")
    plt.tight_layout()
    plt.savefig(out_dir / f"box_{metric}.png", dpi=180)
    plt.close()


def main():
    tartan_m, tartan_rows = collect_dataset("TartanEvent", TARTAN_ROOT)
    davis_m, davis_rows = collect_dataset("DAVIS240C", DAVIS_ROOT)

    write_sequence_csv(tartan_rows + davis_rows, OUT_ROOT / "per_sequence_distribution_stats.csv")
    summary = write_global_summary(
        {"TartanEvent": tartan_m, "DAVIS240C": davis_m},
        OUT_ROOT / "global_distribution_summary.json",
    )
    test_rows = write_distribution_tests(
        tartan_m, davis_m,
        OUT_ROOT / "distribution_shift_tests.csv",
    )

    key_metrics = [
        "nnz_frac",
        "pos_frac",
        "neg_frac",
        "abs_mean",
        "abs_sum",
        "pos_sum",
        "neg_sum_abs",
        "polarity_balance",
        "x_centroid",
        "y_centroid",
        "temporal_centroid",
        "bin0_abs_sum",
        "bin1_abs_sum",
        "bin2_abs_sum",
        "bin3_abs_sum",
        "bin4_abs_sum",
    ]

    for metric in key_metrics:
        plot_metric(tartan_m, davis_m, metric, OUT_ROOT)

    print("\n" + "=" * 100)
    print("KEY GLOBAL COMPARISON")
    print("=" * 100)

    for metric in ["nnz_frac", "abs_mean", "abs_sum", "pos_sum", "neg_sum_abs", "polarity_balance", "temporal_centroid"]:
        ta = summary["TartanEvent"][metric]
        da = summary["DAVIS240C"][metric]
        ratio = (da["mean"] + EPS) / (ta["mean"] + EPS)
        print(
            f"{metric:22s} "
            f"Tartan mean={ta['mean']:.6e} p50={ta['p50']:.6e} | "
            f"DAVIS mean={da['mean']:.6e} p50={da['p50']:.6e} | "
            f"ratio D/T={ratio:.4f}"
        )

    print("\nTop distribution shifts by KS statistic:")
    valid = [r for r in test_rows if np.isfinite(r.get("ks_stat", np.nan))]
    valid = sorted(valid, key=lambda r: r["ks_stat"], reverse=True)
    for r in valid[:12]:
        print(
            f"{r['metric']:22s} "
            f"KS={r['ks_stat']:.4f} "
            f"W={r['wasserstein']:.6e} "
            f"mean_ratio_D/T={r['ratio_davis_over_tartan_mean']:.4f}"
        )

    print("\nSaved:")
    print(" ", OUT_ROOT / "per_sequence_distribution_stats.csv")
    print(" ", OUT_ROOT / "global_distribution_summary.json")
    print(" ", OUT_ROOT / "distribution_shift_tests.csv")
    if HAS_MPL:
        print(" ", OUT_ROOT / "hist_*.png")
        print(" ", OUT_ROOT / "box_*.png")


if __name__ == "__main__":
    main()
