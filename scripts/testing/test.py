import argparse
import json
from pathlib import Path
import sys
#
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.learning.network.build_model import build_model, normalize_checkpoint_state_dict
from src.learning.dataloader.events_to_voxel.precomputed_voxel_clip import PrecomputedVoxelClipDataset
from scripts.utils.config import default_config_path, parse_args_with_config

"python scripts/testing/test.py --sequence_dir data/eds/testing --checkpoint_file checkpoints/noquat_normalized_v1_epoch100_checkpoint_best.pth --output_file data/eds/predicted_relative_motions/sequence_02/v1_predicted_relative_motions.txt"


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
    loaded.setdefault("voxel_filename", "derotated_voxels.npy")
    loaded.setdefault("derotation_slices", 100)
    loaded.setdefault("covariance", False)
    loaded.setdefault("normalize_voxel_nonzero", False)
    return loaded


def get_outputs_per_motion(infer_args):
    clip_pairs = infer_args["clip_len"] - 1
    num_classes = infer_args["model_params"]["num_classes"]
    if num_classes % clip_pairs != 0:
        raise ValueError(
            f"Model num_classes={num_classes} is not divisible by clip_len - 1={clip_pairs}."
        )

    outputs_per_motion = num_classes // clip_pairs
    if outputs_per_motion not in {3, 6}:
        raise ValueError(
            f"Unsupported model output width: {outputs_per_motion} per relative motion. "
            "Expected 3 or 6."
        )
    return outputs_per_motion


def apply_precomputed_voxel_args(args_dict, dataset):
    for key in (
        "num_bins",
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
    voxel_filename = args_dict["voxel_filename"]
    if (sequence_dir / voxel_filename).exists():
        dataset_root = sequence_dir.parent
        requested_sequence = sequence_dir

    dataset = PrecomputedVoxelClipDataset(
        root_path=dataset_root,
        clip_len=args_dict["clip_len"],
        num_bins=None,
        voxel_filename=voxel_filename,
        normalize_voxel_nonzero=args_dict.get("normalize_voxel_nonzero", False),
    )
    apply_precomputed_voxel_args(args_dict, dataset)

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


def collect_last_step_predictions(loader, model, device, infer_args, target_mean, target_std, save_covariance=False):
    preds = []
    rel_t0_list = []
    rel_t1_list = []
    outputs_per_motion = get_outputs_per_motion(infer_args)

    if save_covariance and outputs_per_motion != 6:
        raise ValueError("--save_covariance requires a checkpoint with 6 outputs per relative motion.")

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["representation"].to(device).float()
            anchors = batch["anchors_us"].cpu().numpy()
            y_hat = model(x)
            y_hat_full = y_hat.view(x.shape[0], infer_args["clip_len"] - 1, outputs_per_motion)
            y_hat_tr = y_hat_full[..., :3]
            y_hat_cov = y_hat_full[..., 3:] if outputs_per_motion == 6 else None

            if target_mean is not None and target_std is not None:
                y_hat_tr = y_hat_tr * target_std + target_mean

            y_hat_tr = y_hat_tr.cpu().numpy()
            if y_hat_cov is not None:
                y_hat_cov = np.exp(y_hat_cov.cpu().numpy())

            for i in range(y_hat_tr.shape[0]):
                anc_i = anchors[i]

                if batch_idx == 0 and i == 0:
                    row = y_hat_tr[i, 0]
                    if save_covariance:
                        row = np.concatenate([row, y_hat_cov[i, 0]], axis=-1)
                    preds.append(row)
                    rel_t0_list.append(int(anc_i[0]))
                    rel_t1_list.append(int(anc_i[1]))

                row = y_hat_tr[i, -1]
                if save_covariance:
                    row = np.concatenate([row, y_hat_cov[i, -1]], axis=-1)
                preds.append(row)
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


def collect_averaged_predictions(loader, model, device, infer_args, target_mean, target_std, save_covariance=False):
    prediction_store = {}
    outputs_per_motion = get_outputs_per_motion(infer_args)

    if save_covariance and outputs_per_motion != 6:
        raise ValueError("--save_covariance requires a checkpoint with 6 outputs per relative motion.")

    with torch.no_grad():
        for batch in loader:
            x = batch["representation"].to(device).float()
            anchors = batch["anchors_us"].cpu().numpy()
            y_hat = model(x)
            y_hat_full = y_hat.view(x.shape[0], infer_args["clip_len"] - 1, outputs_per_motion)
            y_hat_tr = y_hat_full[..., :3]
            y_hat_cov = y_hat_full[..., 3:] if outputs_per_motion == 6 else None

            if target_mean is not None and target_std is not None:
                y_hat_tr = y_hat_tr * target_std + target_mean

            y_hat_tr = y_hat_tr.cpu().numpy()
            if y_hat_cov is not None:
                y_hat_cov = np.exp(y_hat_cov.cpu().numpy())

            for i in range(y_hat_tr.shape[0]):
                anc_i = anchors[i]
                for step_idx in range(y_hat_tr.shape[1]):
                    value = y_hat_tr[i, step_idx]
                    if save_covariance:
                        value = np.concatenate([value, y_hat_cov[i, step_idx]], axis=-1)
                    key = (int(anc_i[step_idx]), int(anc_i[step_idx + 1]))
                    prediction_store.setdefault(key, []).append(value)

    return average_overlapping_predictions(prediction_store)


def collect_raw_model_outputs(loader, model, device, infer_args, target_mean, target_std):
    rows = []
    outputs_per_motion = get_outputs_per_motion(infer_args)

    with torch.no_grad():
        for batch in loader:
            x = batch["representation"].to(device).float()
            anchors = batch["anchors_us"].cpu().numpy()
            y_hat = model(x)
            y_hat_tr = y_hat.view(
                x.shape[0],
                infer_args["clip_len"] - 1,
                outputs_per_motion,
            )[..., :3]

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


def should_apply_eds_axis_remap(sequence_dir: Path) -> bool:
    return any(part.lower() == "eds" for part in sequence_dir.resolve().parts)


def remap_eds_prediction_axes(rows_pred: np.ndarray) -> np.ndarray:
    """Map model-frame translations to the EDS local target convention."""
    remapped = rows_pred.copy()
    remapped[:, :3] = rows_pred[:, [1, 2, 0]]
    if rows_pred.shape[1] == 6:
        remapped[:, 3:6] = rows_pred[:, [4, 5, 3]]
    return remapped


def remap_eds_raw_model_outputs(raw_model_outputs: np.ndarray, clip_len: int) -> np.ndarray:
    remapped = raw_model_outputs.copy()
    for step_idx in range(clip_len - 1):
        base = step_idx * 5
        remapped[:, base + 2 : base + 5] = raw_model_outputs[:, [base + 3, base + 4, base + 2]]
    return remapped


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
        "--save_covariance",
        action="store_true",
        help="Save diagonal covariance sigma_x sigma_y sigma_z for each predicted relative motion.",
    )

    args_cli = parse_args_with_config(
        parser,
        default_config_path("test"),
        required=("sequence_dir", "checkpoint_file", "output_file"),
    )

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
            loader, model, device, infer_args, target_mean, target_std, args_cli.save_covariance
        )
    else:
        rows_pred, timestamps = collect_last_step_predictions(
            loader, model, device, infer_args, target_mean, target_std, args_cli.save_covariance
        )
    raw_model_outputs = collect_raw_model_outputs(
        loader, model, device, infer_args, target_mean, target_std
    )
    if should_apply_eds_axis_remap(sequence_dir):
        rows_pred = remap_eds_prediction_axes(rows_pred)
        raw_model_outputs = remap_eds_raw_model_outputs(raw_model_outputs, infer_args["clip_len"])
        print("Applied EDS prediction axis remap: [px, py, pz] -> [py, pz, px]")

    out = np.concatenate([timestamps, rows_pred], axis=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    raw_output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if args_cli.save_covariance:
        header = "t0_us t1_us px py pz sigma_x sigma_y sigma_z"
        fmt = ["%d", "%d"] + ["%.10f"] * 6
    else:
        header = "t0_us t1_us px py pz"
        fmt = ["%d", "%d"] + ["%.10f"] * 3

    np.savetxt(
        output_file,
        out,
        fmt=fmt,
        header=header,
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
