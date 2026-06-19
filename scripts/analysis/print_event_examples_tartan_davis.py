from pathlib import Path
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


TARTAN_ROOT = Path("data/tartanair/precomputed_test")
DAVIS_ROOT  = Path("data/davis240c/precomputed_checkpoint_compatible")
OUT_ROOT    = Path("outputs/distribution_analysis/event_examples")
VOXEL_NAME  = "derotated_voxels.npy"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

EPS = 1e-12


def find_voxel_file(seq_dir: Path):
    p = seq_dir / VOXEL_NAME
    if p.exists():
        return p

    candidates = sorted(seq_dir.glob("*.npy"))
    voxel_candidates = [x for x in candidates if "voxel" in x.name.lower()]
    if voxel_candidates:
        return voxel_candidates[0]
    if candidates:
        return candidates[0]
    return None


def describe_frame(v, dataset, seq, idx):
    """
    v: [B,H,W]
    """
    B, H, W = v.shape
    total = B * H * W

    pos = np.maximum(v, 0)
    neg = np.maximum(-v, 0)
    abs_v = np.abs(v)

    nnz = np.count_nonzero(v)
    pos_nnz = np.count_nonzero(v > 0)
    neg_nnz = np.count_nonzero(v < 0)

    pos_sum = float(pos.sum())
    neg_sum = float(neg.sum())
    abs_sum = float(abs_v.sum())
    signed_sum = float(v.sum())

    polarity_balance = (pos_sum - neg_sum) / (pos_sum + neg_sum + EPS)

    print()
    print(f"--- Example frame | {dataset} | {seq} | idx={idx} ---")
    print(f"shape: B={B}, H={H}, W={W}")
    print(f"nonzero pixels/tokens: {nnz}/{total} = {nnz / total:.6e}")
    print(f"positive nnz:          {pos_nnz}/{total} = {pos_nnz / total:.6e}")
    print(f"negative nnz:          {neg_nnz}/{total} = {neg_nnz / total:.6e}")
    print(f"pos_sum:               {pos_sum:.6e}")
    print(f"neg_sum_abs:           {neg_sum:.6e}")
    print(f"abs_sum/event_mass:    {abs_sum:.6e}")
    print(f"signed_sum:            {signed_sum:.6e}")
    print(f"polarity_balance:      {polarity_balance:+.6f}   [-1 all negative, +1 all positive]")

    print()
    print("Per temporal bin:")
    print("bin | pos_nnz    neg_nnz    pos_sum        neg_sum_abs    abs_sum        balance")
    print("----|-------------------------------------------------------------------------")

    for b in range(B):
        vb = v[b]
        pb = np.maximum(vb, 0)
        nb = np.maximum(-vb, 0)
        ab = np.abs(vb)

        pb_sum = float(pb.sum())
        nb_sum = float(nb.sum())
        ab_sum = float(ab.sum())
        bal = (pb_sum - nb_sum) / (pb_sum + nb_sum + EPS)

        print(
            f"{b:3d} | "
            f"{np.count_nonzero(vb > 0):8d} "
            f"{np.count_nonzero(vb < 0):8d} "
            f"{pb_sum:13.6e} "
            f"{nb_sum:13.6e} "
            f"{ab_sum:13.6e} "
            f"{bal:+9.5f}"
        )

    # Top activated locations collapsed over bins.
    flat = abs_v.sum(axis=0).reshape(-1)
    if flat.size > 0 and flat.max() > 0:
        topk = min(10, flat.size)
        ids = np.argpartition(flat, -topk)[-topk:]
        ids = ids[np.argsort(flat[ids])[::-1]]

        print()
        print("Top spatial event locations collapsed over bins:")
        print("rank | y    x    abs_mass")
        print("-----|-------------------")
        for r, flat_id in enumerate(ids, 1):
            y, x = divmod(int(flat_id), W)
            print(f"{r:4d} | {y:4d} {x:4d} {flat[flat_id]:11.6e}")


def save_example_images(v, dataset, seq, idx):
    if not HAS_MPL:
        return

    B, H, W = v.shape

    # Collapsed positive / negative map.
    pos_map = np.maximum(v, 0).sum(axis=0)
    neg_map = np.maximum(-v, 0).sum(axis=0)
    abs_map = np.abs(v).sum(axis=0)

    vmax = max(float(pos_map.max()), float(neg_map.max()), EPS)

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[..., 0] = pos_map / vmax          # positive = red
    rgb[..., 2] = neg_map / vmax          # negative = blue
    rgb[..., 1] = 0.25 * abs_map / (float(abs_map.max()) + EPS)

    out = OUT_ROOT / f"{dataset}_{seq}_idx{idx:06d}_collapsed_polarity.png"

    plt.figure(figsize=(8, 5))
    plt.imshow(np.clip(rgb, 0, 1))
    plt.title(f"{dataset} | {seq} | idx={idx} | collapsed polarity\nred=positive, blue=negative")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()

    # One image per bin.
    for b in range(B):
        vb = v[b]
        pos = np.maximum(vb, 0)
        neg = np.maximum(-vb, 0)
        vmax_b = max(float(pos.max()), float(neg.max()), EPS)

        rgb_b = np.zeros((H, W, 3), dtype=np.float32)
        rgb_b[..., 0] = pos / vmax_b
        rgb_b[..., 2] = neg / vmax_b

        out_b = OUT_ROOT / f"{dataset}_{seq}_idx{idx:06d}_bin{b}_polarity.png"

        plt.figure(figsize=(8, 5))
        plt.imshow(np.clip(rgb_b, 0, 1))
        plt.title(f"{dataset} | {seq} | idx={idx} | bin={b}\nred=positive, blue=negative")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_b, dpi=180)
        plt.close()


def analyze_dataset(dataset, root):
    print()
    print("=" * 100)
    print(f"DATASET: {dataset}")
    print(f"ROOT:    {root}")
    print("=" * 100)

    if not root.exists():
        print(f"ERROR: missing root {root}")
        return

    seq_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not seq_dirs:
        print("ERROR: no sequence folders found")
        return

    for seq_dir in seq_dirs:
        vf = find_voxel_file(seq_dir)
        if vf is None:
            print(f"SKIP {seq_dir.name}: no voxel npy")
            continue

        vox = np.load(vf, mmap_mode="r")
        if vox.ndim != 4:
            print(f"SKIP {seq_dir.name}: expected [N,B,H,W], got {vox.shape}")
            continue

        N, B, H, W = vox.shape

        # Take three representative frames: start-ish, middle, end-ish.
        candidate_indices = sorted(set([
            min(max(0, N // 10), N - 1),
            min(max(0, N // 2), N - 1),
            min(max(0, 9 * N // 10), N - 1),
        ]))

        print()
        print(f"Sequence: {seq_dir.name}")
        print(f"voxel_file: {vf}")
        print(f"shape: N={N}, B={B}, H={H}, W={W}")
        print(f"example_indices: {candidate_indices}")

        for idx in candidate_indices:
            v = np.asarray(vox[idx], dtype=np.float32)
            describe_frame(v, dataset, seq_dir.name, idx)
            save_example_images(v, dataset, seq_dir.name, idx)


def main():
    analyze_dataset("TartanEvent", TARTAN_ROOT)
    analyze_dataset("DAVIS240C", DAVIS_ROOT)

    print()
    print("=" * 100)
    print("Saved qualitative polarity/event examples in:")
    print(OUT_ROOT)
    print("=" * 100)


if __name__ == "__main__":
    main()
