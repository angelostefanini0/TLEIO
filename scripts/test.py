import argparse
import json
from pathlib import Path
import sys
#
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from learning.network.build_model import build_model, normalize_checkpoint_state_dict
from learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset
from learning.dataloader.events_to_voxel.precomputed_voxel_clip import PrecomputedVoxelClipDataset

"python scripts/test.py --sequence_dir data/eds/testing --checkpoint_file checkpoints/noquat_normalized_v1_epoch100_checkpoint_best.pth --output_file data/eds/predicted_relative_motions/sequence_02/v1_predicted_relative_motions.txt"


class RepresentationScaleDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, scale: float):
        self.dataset = dataset
        self.scale = float(scale)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        if self.scale == 1.0:
            return sample

        sample = dict(sample)
        sample["representation"] = sample["representation"] * self.scale
        return sample


def load_inference_args(checkpoint_file: Path):
    args_file = checkpoint_file.parent / "args.txt"

    if not args_file.exists():
        raise ValueError(
            "Make sure there is an args.txt file inside the checkpoint folder specified as an argument"
        )

    with open(args_file, "r") as f:
        loaded = json.load(f)

    if "model_params" not in loaded:
        raise ValueError(
            "Make sure the args.txt file inside the checkpoint folder has a model_params key"
        )

    loaded["checkpoint"] = None
    loaded["checkpoint_path"] = str(checkpoint_file.parent)
    loaded["distributed"] = False
    loaded["world_size"] = 1
    loaded["rank"] = 0
    loaded["local_rank"] = 0
    loaded["is_main_process"] = True
    loaded.setdefault("precomputed_voxels", False)
    loaded.setdefault("voxel_filename", "derotated_voxels.npy")
    loaded.setdefault("derotation_slices", 100)
    return loaded


def apply_precomputed_voxel_args(args_dict, dataset):
    for key in (
        "num_bins",
        "downsampling_factor",
        "denoising",
        "denoise_dt_us",
        "denoise_radius",
        "denoise_min_supporters",
        "denoise_same_polarity_only",
        "derotate",
        "derotation_slices",
    ):
        value = getattr(dataset, key, None)
        if value is not None:
            args_dict[key] = value


def build_inference_dataset(sequence_dir: Path, args_dict):
    sequence_dir = sequence_dir.resolve()

    dataset_root = sequence_dir
    requested_sequence = None
    if args_dict["precomputed_voxels"]:
        voxel_filename = args_dict["voxel_filename"]
        if (sequence_dir / voxel_filename).exists():
            dataset_root = sequence_dir.parent
            requested_sequence = sequence_dir

        dataset = PrecomputedVoxelClipDataset(
            root_path=dataset_root,
            clip_len=args_dict["clip_len"],
            num_bins=None,
            voxel_filename=voxel_filename,
        )
        apply_precomputed_voxel_args(args_dict, dataset)
    else:
        if (sequence_dir / "events.h5").exists():
            dataset_root = sequence_dir.parent
            requested_sequence = sequence_dir

        dataset = MultiEventVoxelClipDataset(
            root_path=dataset_root,
            delta_t_ms=args_dict["delta_t_ms"],
            num_bins=args_dict["num_bins"],
            clip_len=args_dict["clip_len"],
            downsampling_factor=args_dict["downsampling_factor"],
            patch_size=args_dict["patch_size"],
            denoising=args_dict["denoising"],
            denoise_dt_us=args_dict["denoise_dt_us"],
            denoise_radius=args_dict["denoise_radius"],
            denoise_min_supporters=args_dict["denoise_min_supporters"],
            denoise_same_polarity_only=args_dict["denoise_same_polarity_only"],
            derotate=args_dict["derotate"],
            derotation_slices=args_dict["derotation_slices"],
        )

    if requested_sequence is None:
        return dataset

    selected_indices = []
    start_idx = 0
    for seq_idx, seq_info in enumerate(dataset.seq_infos):
        end_idx = dataset.cum_lengths[seq_idx]
        if seq_info["seq_path"].resolve() == requested_sequence:
            selected_indices.extend(range(start_idx, end_idx))
            break
        start_idx = end_idx

    if not selected_indices:
        raise ValueError(f"Sequence not found in dataset: {requested_sequence}")

    return torch.utils.data.Subset(dataset, selected_indices)


def iter_precomputed_voxel_files(root_path: Path, voxel_filename: str):
    if (root_path / voxel_filename).exists():
        yield root_path / voxel_filename
        return

    for seq_path in sorted(p for p in root_path.iterdir() if p.is_dir()):
        voxel_file = seq_path / voxel_filename
        if voxel_file.exists():
            yield voxel_file


def compute_precomputed_mean_abs(root_path: Path, voxel_filename: str, chunk_size: int):
    total_abs = 0.0
    total_count = 0
    voxel_files = list(iter_precomputed_voxel_files(root_path, voxel_filename))

    if not voxel_files:
        raise FileNotFoundError(
            f"No '{voxel_filename}' files found under {root_path}"
        )

    for voxel_file in voxel_files:
        voxels = np.load(voxel_file, mmap_mode="r")
        for start in range(0, voxels.shape[0], chunk_size):
            chunk = np.asarray(voxels[start:start + chunk_size], dtype=np.float32)
            total_abs += np.abs(chunk).sum(dtype=np.float64)
            total_count += chunk.size

    if total_count == 0:
        raise ValueError(f"No voxel values found under {root_path}")

    return total_abs / total_count


def compute_auto_voxel_scale(reference_dir: Path, target_dir: Path, voxel_filename: str, chunk_size: int):
    reference_mean_abs = compute_precomputed_mean_abs(reference_dir, voxel_filename, chunk_size)
    target_mean_abs = compute_precomputed_mean_abs(target_dir, voxel_filename, chunk_size)

    if target_mean_abs <= 0:
        raise ValueError(f"Target voxel mean abs is zero for {target_dir}")

    return reference_mean_abs / target_mean_abs, reference_mean_abs, target_mean_abs


def load_target_stats(checkpoint, device):
    target_mean = checkpoint.get("target_mean")
    target_std = checkpoint.get("target_std")
    if target_mean is None or target_std is None:
        return None, None

    target_mean = torch.as_tensor(target_mean, dtype=torch.float32, device=device).flatten()
    target_std = torch.as_tensor(target_std, dtype=torch.float32, device=device).flatten()

    if target_mean.numel() >= 3:
        target_mean = target_mean[:3]
        target_std = target_std[:3]
    else:
        raise ValueError("Checkpoint target statistics must have at least 3 elements.")

    target_mean = target_mean.view(1, 1, 3)
    target_std = target_std.view(1, 1, 3)
    return target_mean, target_std


def average_overlapping_predictions(prediction_store):
    rows = []
    timestamps = []

    for key in sorted(prediction_store.keys()):
        values = np.stack(prediction_store[key], axis=0)
        rows.append(values.mean(axis=0))
        timestamps.append(key)

    rows_pred = np.asarray(rows, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    return rows_pred, timestamps


def collect_last_step_predictions(loader, model, device, infer_args, target_mean, target_std):
    preds = []
    rel_t0_list = []
    rel_t1_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["representation"].to(device).float()
            anchors = batch["anchors_us"].cpu().numpy()
            y_hat = model(x)
            y_hat_tr = y_hat.view(x.shape[0], infer_args["clip_len"] - 1, 3)

            if target_mean is not None and target_std is not None:
                y_hat_tr = y_hat_tr * target_std + target_mean

            y_hat_tr = y_hat_tr.cpu().numpy()

            for i in range(y_hat_tr.shape[0]):
                anc_i = anchors[i]

                if batch_idx == 0 and i == 0:
                    preds.append(y_hat_tr[i, 0])
                    rel_t0_list.append(int(anc_i[0]))
                    rel_t1_list.append(int(anc_i[1]))

                preds.append(y_hat_tr[i, -1])
                rel_t0_list.append(int(anc_i[-2]))
                rel_t1_list.append(int(anc_i[-1]))

    rows_pred = np.asarray(preds, dtype=np.float64)
    timestamps = np.column_stack(
        [
            np.asarray(rel_t0_list, dtype=np.int64),
            np.asarray(rel_t1_list, dtype=np.int64),
        ]
    )
    return rows_pred, timestamps


def collect_averaged_predictions(loader, model, device, infer_args, target_mean, target_std):
    prediction_store = {}

    with torch.no_grad():
        for batch in loader:
            x = batch["representation"].to(device).float()
            anchors = batch["anchors_us"].cpu().numpy()
            y_hat = model(x)
            y_hat_tr = y_hat.view(x.shape[0], infer_args["clip_len"] - 1, 3)

            if target_mean is not None and target_std is not None:
                y_hat_tr = y_hat_tr * target_std + target_mean

            y_hat_tr = y_hat_tr.cpu().numpy()

            for i in range(y_hat_tr.shape[0]):
                anc_i = anchors[i]
                for step_idx in range(y_hat_tr.shape[1]):
                    key = (int(anc_i[step_idx]), int(anc_i[step_idx + 1]))
                    prediction_store.setdefault(key, []).append(y_hat_tr[i, step_idx])

    return average_overlapping_predictions(prediction_store)


def collect_raw_model_outputs(loader, model, device, infer_args, target_mean, target_std):
    rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["representation"].to(device).float()
            anchors = batch["anchors_us"].cpu().numpy()
            y_hat = model(x)
            y_hat_tr = y_hat.view(x.shape[0], infer_args["clip_len"] - 1, 3)

            if target_mean is not None and target_std is not None:
                y_hat_tr = y_hat_tr * target_std + target_mean

            y_hat_tr = y_hat_tr.cpu().numpy()

            for i in range(y_hat_tr.shape[0]):
                row = []
                anc_i = anchors[i]
                for step_idx in range(y_hat_tr.shape[1]):
                    row.extend(
                        [
                            int(anc_i[step_idx]),
                            int(anc_i[step_idx + 1]),
                            *y_hat_tr[i, step_idx],
                        ]
                    )
                rows.append(row)

    return np.asarray(rows, dtype=np.float64).reshape(-1, 5 * (infer_args["clip_len"] - 1))


def raw_model_output_path(output_file: Path):
    return output_file.with_name(f"{output_file.stem}_raw_model_outputs{output_file.suffix}")


def raw_model_output_header(clip_len: int):
    columns = []
    for step_idx in range(clip_len - 1):
        columns.extend(
            [
                f"step{step_idx}_t0_us",
                f"step{step_idx}_t1_us",
                f"step{step_idx}_px",
                f"step{step_idx}_py",
                f"step{step_idx}_pz",
            ]
        )
    return " ".join(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--raw_model_output_file",
        type=str,
        default=None,
        help="Optional path for one raw model-output row per clip.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers to use during inference.",
    )
    parser.add_argument(
        "--average_overlaps",
        action="store_true",
        help="Average predictions that correspond to the same displacement across overlapping clips.",
    )
    parser.add_argument(
        "--voxel_scale",
        type=float,
        default=1.0,
        help="Multiply input voxel representations by this value at inference time.",
    )
    parser.add_argument(
        "--voxel_scale_reference_dir",
        type=str,
        default=None,
        help=(
            "Optional precomputed training root. If set, compute "
            "scale = mean_abs(reference voxels) / mean_abs(target voxels)."
        ),
    )
    parser.add_argument(
        "--voxel_scale_target_dir",
        type=str,
        default=None,
        help=(
            "Optional precomputed target root for automatic voxel scaling. "
            "Defaults to --sequence_dir."
        ),
    )
    parser.add_argument(
        "--voxel_scale_chunk_size",
        type=int,
        default=64,
        help="Number of precomputed voxel frames to read per chunk when computing mean abs.",
    )
    args_cli = parser.parse_args()

    sequence_dir = Path(args_cli.sequence_dir)
    checkpoint_file = Path(args_cli.checkpoint_file)
    output_file = Path(args_cli.output_file)
    raw_output_file = (
        Path(args_cli.raw_model_output_file)
        if args_cli.raw_model_output_file is not None
        else raw_model_output_path(output_file)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    infer_args = load_inference_args(checkpoint_file)
    infer_args["device"] = str(device)
    dataset = build_inference_dataset(sequence_dir, infer_args)

    effective_voxel_scale = args_cli.voxel_scale
    if args_cli.voxel_scale_reference_dir is not None:
        if not infer_args["precomputed_voxels"]:
            raise ValueError("--voxel_scale_reference_dir requires precomputed voxel inference.")
        auto_scale, ref_mean_abs, target_mean_abs = compute_auto_voxel_scale(
            reference_dir=Path(args_cli.voxel_scale_reference_dir),
            target_dir=Path(args_cli.voxel_scale_target_dir) if args_cli.voxel_scale_target_dir else sequence_dir,
            voxel_filename=infer_args["voxel_filename"],
            chunk_size=args_cli.voxel_scale_chunk_size,
        )
        effective_voxel_scale *= auto_scale
        print(
            "Auto voxel scale: "
            f"reference_mean_abs={ref_mean_abs:.8g} | "
            f"target_mean_abs={target_mean_abs:.8g} | "
            f"auto_scale={auto_scale:.8g}"
        )

    if effective_voxel_scale != 1.0:
        dataset = RepresentationScaleDataset(dataset, effective_voxel_scale)
    print(f"voxel_scale: {effective_voxel_scale:.8g}")

    loader = DataLoader(
        dataset,
        batch_size=infer_args["b_size"],
        shuffle=False,
        num_workers=args_cli.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model, _ = build_model(infer_args, infer_args["model_params"])
    ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
    target_mean, target_std = load_target_stats(ckpt, device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(normalize_checkpoint_state_dict(state_dict))
    model.to(device)
    model.eval()

    if args_cli.average_overlaps:
        rows_pred, timestamps = collect_averaged_predictions(
            loader, model, device, infer_args, target_mean, target_std
        )
    else:
        rows_pred, timestamps = collect_last_step_predictions(
            loader, model, device, infer_args, target_mean, target_std
        )
    raw_model_outputs = collect_raw_model_outputs(
        loader, model, device, infer_args, target_mean, target_std
    )
    out = np.concatenate([timestamps, rows_pred], axis=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    np.savetxt(
        output_file,
        out,
        fmt=["%d", "%d"] + ["%.10f"] * 3,
        header="t0_us t1_us px py pz",
        comments="",
    )
    np.savetxt(
        raw_output_file,
        raw_model_outputs,
        fmt=(["%d", "%d"] + ["%.10f"] * 3) * (infer_args["clip_len"] - 1),
        header=raw_model_output_header(infer_args["clip_len"]),
        comments="",
    )

    print(f"Saved: {output_file}")
    print(f"Saved raw model outputs: {raw_output_file}")
    print(f"num_relative_motions: {len(rows_pred)}")
    print(f"num_raw_model_output_clips: {len(raw_model_outputs)}")
    print(f"average_overlaps: {args_cli.average_overlaps}")


if __name__ == "__main__":
    main()
