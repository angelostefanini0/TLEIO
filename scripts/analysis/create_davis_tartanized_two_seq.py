from pathlib import Path
import shutil
import os
import numpy as np

TARTAN_ROOT = Path("data/tartanair/precomputed_test")
DAVIS_ROOT  = Path("data/davis240c/precomputed_checkpoint_compatible")
OUT_ROOT    = Path("data/davis240c/precomputed_checkpoint_compatible_tartanized_spread_polbin_2seq")

SEQS = ["boxes_6dof", "boxes_translation"]
VOXEL_NAME = "derotated_voxels.npy"

MAX_TARTAN_FRAMES_PER_SEQ = 700
EPS = 1e-12

# Spatial spreading: this changes density/support, not only amplitude.
PASSES = 2
AXIAL_WEIGHT = 0.35
DIAG_WEIGHT = 0.15

# Per-bin/per-polarity scale clipping.
MIN_SCALE = 0.05
MAX_SCALE = 20.0

rng = np.random.default_rng(7)


def hardlink_or_copy(src, dst):
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def copy_side_files(src_seq, dst_seq):
    if dst_seq.exists():
        shutil.rmtree(dst_seq)
    dst_seq.mkdir(parents=True, exist_ok=True)

    for item in src_seq.iterdir():
        if item.name == VOXEL_NAME:
            continue

        dst_item = dst_seq / item.name

        if item.is_dir():
            shutil.copytree(item, dst_item, symlinks=True)
        else:
            hardlink_or_copy(item, dst_item)


def shift_zero(a, dy, dx):
    """
    Zero-padded shift, no wraparound.
    """
    out = np.zeros_like(a)

    H, W = a.shape

    y_src0 = max(0, -dy)
    y_src1 = min(H, H - dy)
    x_src0 = max(0, -dx)
    x_src1 = min(W, W - dx)

    y_dst0 = max(0, dy)
    y_dst1 = min(H, H + dy)
    x_dst0 = max(0, dx)
    x_dst1 = min(W, W + dx)

    out[y_dst0:y_dst1, x_dst0:x_dst1] = a[y_src0:y_src1, x_src0:x_src1]
    return out


def spread_map(a):
    """
    Spatially spreads positive or negative event mass.
    This increases density/support without changing sign semantics.
    """
    out = a.astype(np.float32, copy=True)

    for _ in range(PASSES):
        axial = (
            shift_zero(out,  1,  0) +
            shift_zero(out, -1,  0) +
            shift_zero(out,  0,  1) +
            shift_zero(out,  0, -1)
        )

        diag = (
            shift_zero(out,  1,  1) +
            shift_zero(out,  1, -1) +
            shift_zero(out, -1,  1) +
            shift_zero(out, -1, -1)
        )

        out = out + AXIAL_WEIGHT * axial + DIAG_WEIGHT * diag

    return out


def find_voxel(seq_dir):
    p = seq_dir / VOXEL_NAME
    if p.exists():
        return p
    cands = sorted(seq_dir.glob("*.npy"))
    vcands = [x for x in cands if "voxel" in x.name.lower()]
    if vcands:
        return vcands[0]
    if cands:
        return cands[0]
    return None


def sample_indices(n, max_n):
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def collect_tartan_targets():
    """
    Estimate target per-bin positive/negative mass and density from Tartan.
    """
    pos_sums = []
    neg_sums = []
    pos_nnz = []
    neg_nnz = []

    for seq_dir in sorted(p for p in TARTAN_ROOT.iterdir() if p.is_dir()):
        vf = find_voxel(seq_dir)
        if vf is None:
            continue

        vox = np.load(vf, mmap_mode="r")
        idxs = sample_indices(vox.shape[0], MAX_TARTAN_FRAMES_PER_SEQ)

        for idx in idxs:
            v = np.asarray(vox[idx], dtype=np.float32)
            pos = np.maximum(v, 0)
            neg = np.maximum(-v, 0)

            pos_sums.append(pos.sum(axis=(1, 2)))
            neg_sums.append(neg.sum(axis=(1, 2)))
            pos_nnz.append((v > 0).sum(axis=(1, 2)))
            neg_nnz.append((v < 0).sum(axis=(1, 2)))

    target = {
        "pos_sum": np.median(np.stack(pos_sums), axis=0),
        "neg_sum": np.median(np.stack(neg_sums), axis=0),
        "pos_nnz": np.median(np.stack(pos_nnz), axis=0),
        "neg_nnz": np.median(np.stack(neg_nnz), axis=0),
    }

    return target


def describe_targets(target):
    print("Tartan targets per temporal bin:")
    print("bin | pos_sum       neg_sum_abs   pos_nnz     neg_nnz")
    print("----|--------------------------------------------------")
    for b in range(len(target["pos_sum"])):
        print(
            f"{b:3d} | "
            f"{target['pos_sum'][b]:13.6e} "
            f"{target['neg_sum'][b]:13.6e} "
            f"{target['pos_nnz'][b]:9.1f} "
            f"{target['neg_nnz'][b]:9.1f}"
        )


def transform_frame(v, target):
    """
    v: [B,H,W]
    1. Split positive/negative.
    2. Spatially spread each polarity separately.
    3. Match per-bin positive/negative mass toward Tartan median.
    """
    B, H, W = v.shape
    out = np.zeros_like(v, dtype=np.float32)

    for b in range(B):
        pos = np.maximum(v[b], 0)
        neg = np.maximum(-v[b], 0)

        pos_sp = spread_map(pos)
        neg_sp = spread_map(neg)

        pos_sum = float(pos_sp.sum())
        neg_sum = float(neg_sp.sum())

        s_pos = float(target["pos_sum"][b] / max(pos_sum, EPS))
        s_neg = float(target["neg_sum"][b] / max(neg_sum, EPS))

        s_pos = float(np.clip(s_pos, MIN_SCALE, MAX_SCALE))
        s_neg = float(np.clip(s_neg, MIN_SCALE, MAX_SCALE))

        out[b] = s_pos * pos_sp - s_neg * neg_sp

    return out


def quick_stats(v):
    pos = np.maximum(v, 0)
    neg = np.maximum(-v, 0)
    return {
        "nnz_frac": np.count_nonzero(v) / v.size,
        "pos_sum": float(pos.sum()),
        "neg_sum": float(neg.sum()),
        "abs_sum": float(np.abs(v).sum()),
        "balance": float((pos.sum() - neg.sum()) / (pos.sum() + neg.sum() + EPS)),
    }


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Collecting Tartan targets...")
    target = collect_tartan_targets()
    describe_targets(target)

    for seq in SEQS:
        src_seq = DAVIS_ROOT / seq
        dst_seq = OUT_ROOT / seq

        vf = find_voxel(src_seq)
        if vf is None:
            print(f"SKIP {seq}: no voxel file")
            continue

        print()
        print("=" * 100)
        print(f"Tartanizing {seq}")
        print("=" * 100)

        copy_side_files(src_seq, dst_seq)

        vox = np.load(vf, mmap_mode="r")
        out_file = dst_seq / VOXEL_NAME

        out = np.lib.format.open_memmap(
            out_file,
            mode="w+",
            dtype=np.float32,
            shape=vox.shape,
        )

        before_examples = []
        after_examples = []

        for i in range(vox.shape[0]):
            v = np.asarray(vox[i], dtype=np.float32)
            vt = transform_frame(v, target)
            out[i] = vt

            if i in {0, vox.shape[0] // 2, vox.shape[0] - 1}:
                before_examples.append((i, quick_stats(v)))
                after_examples.append((i, quick_stats(vt)))

        out.flush()

        print(f"saved: {out_file}")
        print("Before/after example stats:")
        for (i, b), (_, a) in zip(before_examples, after_examples):
            print(f"idx={i}")
            print(f"  before nnz={b['nnz_frac']:.4f} abs_sum={b['abs_sum']:.3e} balance={b['balance']:+.3f}")
            print(f"  after  nnz={a['nnz_frac']:.4f} abs_sum={a['abs_sum']:.3e} balance={a['balance']:+.3f}")

    print()
    print("Done.")
    print("Output root:", OUT_ROOT)


if __name__ == "__main__":
    main()
