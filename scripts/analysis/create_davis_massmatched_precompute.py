from pathlib import Path
import shutil
import numpy as np

TARTAN_ROOT = Path("data/tartanair/precomputed_test")
DAVIS_ROOT  = Path("data/davis240c/precomputed_checkpoint_compatible")

OUT_ABS     = Path("data/davis240c/precomputed_checkpoint_compatible_massmatch_abs_sum")
OUT_NZMEAN  = Path("data/davis240c/precomputed_checkpoint_compatible_massmatch_nzmean")

VOXEL_NAME = "derotated_voxels.npy"
MAX_TARTAN_FRAMES_PER_SEQ = 1000
MAX_SCALE = 50.0
MIN_SCALE = 0.02
EPS = 1e-12

rng = np.random.default_rng(7)


def find_voxel(seq_dir):
    p = seq_dir / VOXEL_NAME
    if p.exists():
        return p
    cands = sorted(seq_dir.glob("*.npy"))
    if not cands:
        return None
    vcands = [p for p in cands if "voxel" in p.name.lower()]
    return vcands[0] if vcands else cands[0]


def sample_indices(n, max_n):
    if n <= max_n:
        return np.arange(n)
    return np.sort(rng.choice(n, size=max_n, replace=False))


def frame_stats(v):
    abs_v = np.abs(v)
    abs_sum = float(abs_v.sum())
    nnz = int(np.count_nonzero(v))
    nzmean = abs_sum / max(nnz, 1)
    return abs_sum, nzmean


print("Collecting Tartan target distribution...")
tartan_abs_sums = []
tartan_nzmeans = []

for seq_dir in sorted(p for p in TARTAN_ROOT.iterdir() if p.is_dir()):
    vf = find_voxel(seq_dir)
    if vf is None:
        continue
    vox = np.load(vf, mmap_mode="r")
    idxs = sample_indices(vox.shape[0], MAX_TARTAN_FRAMES_PER_SEQ)
    for i in idxs:
        a, n = frame_stats(np.asarray(vox[i], dtype=np.float32))
        if a > EPS:
            tartan_abs_sums.append(a)
            tartan_nzmeans.append(n)

target_abs_sum = float(np.median(tartan_abs_sums))
target_nzmean = float(np.median(tartan_nzmeans))

print(f"Tartan target median abs_sum: {target_abs_sum:.6e}")
print(f"Tartan target median nonzero abs mean: {target_nzmean:.6e}")


def copy_side_files(src_seq, dst_seq):
    if dst_seq.exists():
        shutil.rmtree(dst_seq)
    dst_seq.mkdir(parents=True, exist_ok=True)

    for item in src_seq.iterdir():
        if item.name == VOXEL_NAME:
            continue
        dst_item = dst_seq / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)


def write_scaled_dataset(out_root, mode):
    print()
    print("=" * 100)
    print(f"Writing {mode} matched dataset:")
    print(out_root)
    print("=" * 100)

    out_root.mkdir(parents=True, exist_ok=True)

    for seq_dir in sorted(p for p in DAVIS_ROOT.iterdir() if p.is_dir()):
        vf = find_voxel(seq_dir)
        if vf is None:
            print("SKIP", seq_dir.name, "no voxel file")
            continue

        dst_seq = out_root / seq_dir.name
        copy_side_files(seq_dir, dst_seq)

        vox = np.load(vf, mmap_mode="r")
        out_file = dst_seq / VOXEL_NAME

        out = np.lib.format.open_memmap(
            out_file,
            mode="w+",
            dtype=np.float32,
            shape=vox.shape,
        )

        scales = []

        for i in range(vox.shape[0]):
            v = np.asarray(vox[i], dtype=np.float32)
            abs_sum, nzmean = frame_stats(v)

            if mode == "abs_sum":
                scale = target_abs_sum / max(abs_sum, EPS)
            elif mode == "nzmean":
                scale = target_nzmean / max(nzmean, EPS)
            else:
                raise ValueError(mode)

            scale = float(np.clip(scale, MIN_SCALE, MAX_SCALE))
            out[i] = v * scale
            scales.append(scale)

        out.flush()

        scales = np.asarray(scales)
        print(
            f"{seq_dir.name:25s} shape={vox.shape} "
            f"scale_mean={scales.mean():.3f} "
            f"scale_p50={np.percentile(scales,50):.3f} "
            f"scale_p95={np.percentile(scales,95):.3f} "
            f"saved={out_file}"
        )


write_scaled_dataset(OUT_ABS, "abs_sum")
write_scaled_dataset(OUT_NZMEAN, "nzmean")

print()
print("Done.")
print("ABS_SUM dataset: ", OUT_ABS)
print("NZMEAN dataset:  ", OUT_NZMEAN)
