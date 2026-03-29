import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.learning.network.build_model import build_model
from src.learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset


ARGS = {
    "root_dir": "data/eds/processed",
    "b_size": 2,
    "val_split": 0.1,
    "clip_len": 3,
    "delta_t_ms": 50,
    "num_bins": 5,
    "optimizer": "Adam",
    "lr": 1e-05,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "epoch": 100,
    "weighted_loss": None,
    "pretrained_ViT": False,
    "num_workers": 0,
    "checkpoint_path": "checkpoints",
    "checkpoint": None,
    "embed_dim": 384,
    "patch_size": 16,
    "attention_type": "divided_space_time",
    "depth": 6,
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
        "num_classes": 14,
        "depth": 6,
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


def load_timestamps(sequence_dir: Path):
    rel_file = sequence_dir / "01_peanuts_light/relative_motions.txt"
    data = np.loadtxt(rel_file, comments="#", skiprows=1)
    return data[:, :2].astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_dir", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args_cli = parser.parse_args()

    sequence_dir = Path(args_cli.sequence_dir)
    checkpoint_file = Path(args_cli.checkpoint_file)
    output_file = Path(args_cli.output_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MultiEventVoxelClipDataset(
        root_path=sequence_dir,
        delta_t_ms=ARGS["delta_t_ms"],
        num_bins=ARGS["num_bins"],
        clip_len=ARGS["clip_len"],
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model, _ = build_model(ARGS, ARGS["model_params"])
    ckpt = torch.load(checkpoint_file, map_location=device)

    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            x = batch["representation"].to(device).float()
            y_hat = model(x)
            y_hat = y_hat.view(x.shape[0], ARGS["clip_len"] - 1, 7)
            preds.append(y_hat.cpu().numpy())

    preds = np.concatenate(preds, axis=0)   # [num_clips, 2, 7]

    # merge con averaging delle overlap
    rel_dict = {}
    for clip_idx in range(preds.shape[0]):
        for step in range(preds.shape[1]):
            global_idx = clip_idx + step
            rel_dict.setdefault(global_idx, []).append(preds[clip_idx, step])

    rows_pred = []
    for global_idx in sorted(rel_dict.keys()):
        motions = np.asarray(rel_dict[global_idx], dtype=np.float64)
        trans = motions[:, :3].mean(axis=0)
        quat = average_quaternions(motions[:, 3:])
        rows_pred.append(np.concatenate([trans, quat], axis=0))

    rows_pred = np.stack(rows_pred, axis=0)   # [num_motions, 7]

    timestamps = load_timestamps(sequence_dir)

    n = min(len(timestamps), len(rows_pred))
    out = np.concatenate([timestamps[:n], rows_pred[:n]], axis=1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_file,
        out,
        fmt=["%d", "%d"] + ["%.10f"] * 7,
        header="t0_us t1_us px py pz qx qy qz qw",
        comments=""
    )

    print(f"Saved: {output_file}")
    print(f"num_clips: {preds.shape[0]}")
    print(f"num_relative_motions: {n}")


if __name__ == "__main__":
    main()