import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.learning.network.build_model import build_model
from src.learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset

"python test.py --sequence_dir data/eds/testing --checkpoint_file checkpoints/noquat_normalized_v1_epoch100_checkpoint_best.pth --output_file data/eds/predicted_relative_motions/sequence_02/v1_predicted_relative_motions.txt"

ARGS = {
    "root_dir": "data/eds/processed",
    "b_size": 2,
    "val_split": 0.1,
    "clip_len": 3,
    "delta_t_ms": 50,
    "num_bins": 5,
    "downsampling_factor": 0.7,
    "denoising": False,
    "denoise_dt_us": 2000,
    "denoise_radius": 1,
    "denoise_min_supporters": 2,
    "denoise_same_polarity_only": False,
    "derotate": True,
    "optimizer": "AdamW",
    "lr": 1e-05,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "epoch": 100,
    "pretrained_ViT": False,
    "num_workers": 4,
    "checkpoint_path": "checkpoints",
    "checkpoint": None,
    "embed_dim": 384,
    "patch_size": 16,
    "attention_type": "divided_space_time",
    "depth": 12,
    "heads": 6,
    "dim_head": 64,
    "attn_dropout": 0.1,
    "ff_dropout": 0.1,
    "time_only": False,
    "model_params": {
        "embed_dim": 384,
        "patch_size": 16,
        "attention_type": "divided_space_time",
        "num_frames": 3,
        "num_classes": 6,
        "depth": 12,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    },
    "epoch_init": 1,
    "best_val": float("inf"),
}


def normalize_quat(q):
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return q / n


def average_quaternions(quats):
    quats = normalize_quat(quats)
    q_ref = quats[0].copy()
    aligned = []
    for q in quats:
        aligned.append(-q if np.dot(q_ref, q) < 0 else q)
    aligned = np.stack(aligned, axis=0)
    q = aligned.mean(axis=0)
    return q / np.linalg.norm(q)


def load_inference_args(checkpoint_file: Path):
    args_file = checkpoint_file.parent / "args.txt"
    if not args_file.exists():
        return ARGS.copy()

    with open(args_file, "r") as f:
        loaded = json.load(f)

    if "model_params" not in loaded:
        loaded["model_params"] = {
            "embed_dim": loaded["embed_dim"],
            "patch_size": loaded["patch_size"],
            "attention_type": loaded["attention_type"],
            "num_frames": loaded["clip_len"],
            "num_classes": 3 * (loaded["clip_len"] - 1),
            "depth": loaded["depth"],
            "heads": loaded["heads"],
            "dim_head": loaded["dim_head"],
            "attn_dropout": loaded["attn_dropout"],
            "ff_dropout": loaded["ff_dropout"],
            "time_only": loaded["time_only"],
        }
    else:
        loaded["model_params"]["num_classes"] = 3 * (loaded["clip_len"] - 1)

    loaded.setdefault("downsampling_factor", 1.0)
    loaded.setdefault("denoising", False)
    loaded.setdefault("denoise_dt_us", 1000)
    loaded.setdefault("denoise_radius", 1)
    loaded.setdefault("denoise_min_supporters", 1)
    loaded.setdefault("denoise_same_polarity_only", False)
    loaded.setdefault("derotate", False)
    loaded["checkpoint"] = None
    loaded["checkpoint_path"] = str(checkpoint_file.parent)
    return loaded


def build_inference_dataset(sequence_dir: Path, args_dict):
    sequence_dir = sequence_dir.resolve()

    dataset_root = sequence_dir
    requested_sequence = None
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
    timestamps = np.column_stack([
        np.asarray(rel_t0_list, dtype=np.int64),
        np.asarray(rel_t1_list, dtype=np.int64),
    ])
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--average_overlaps",
        action="store_true",
        help="Average predictions that correspond to the same displacement across overlapping clips.",
    )
    args_cli = parser.parse_args()

    sequence_dir = Path(args_cli.sequence_dir)
    checkpoint_file = Path(args_cli.checkpoint_file)
    output_file = Path(args_cli.output_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_args = ARGS.copy()
    dataset = build_inference_dataset(sequence_dir, infer_args)

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
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

    model.load_state_dict(state_dict)
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
    out = np.concatenate([timestamps, rows_pred], axis=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_file,
        out,
        fmt=["%d", "%d"] + ["%.10f"] * 3,
        header="t0_us t1_us px py pz",
        comments=""
    )

    print(f"Saved: {output_file}")
    print(f"num_relative_motions: {len(rows_pred)}")
    print(f"average_overlaps: {args_cli.average_overlaps}")


if __name__ == "__main__":
    main()
