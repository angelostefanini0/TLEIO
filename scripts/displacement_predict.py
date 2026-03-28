import sys
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
import matplotlib.pyplot as plt

from src.learning.network.build_model import build_model
from src.learning.network.train import compute_loss, get_optimizer
from src.learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset


def extract_translation(x):
    # assume [r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz]
    return x[..., [3, 7, 11]]


def main():
    args = {
        "root_dir": "data/eds/processed",
        "clip_len": 3,
        "num_bins": 5,
        "delta_t_ms": 50,
        "b_size": 1,
        "checkpoint": None,
        "checkpoint_path": "checkpoints",
        "weighted_loss": None,
        "optimizer": "Adam",
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "epochs": 4,
        "subset_size": 100,
    }

    model_params = {
        "embed_dim": 384,
        "patch_size": 16,
        "attention_type": "divided_space_time",
        "num_frames": args["clip_len"],
        "num_classes": 12 * (args["clip_len"] - 1),
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }

    os.makedirs(args["checkpoint_path"], exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    print("Loading full dataset...")
    full_dataset = MultiEventVoxelClipDataset(
        root_path=Path(args["root_dir"]),
        delta_t_ms=args["delta_t_ms"],
        num_bins=args["num_bins"],
        clip_len=args["clip_len"],
    )

    print(f"Full dataset size: {len(full_dataset)}")
    assert len(full_dataset) > 0, "Dataset vuoto"

    subset_size = min(args["subset_size"], len(full_dataset))
    dataset = torch.utils.data.Subset(full_dataset, range(subset_size))
    print(f"Subset size: {len(dataset)}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args["b_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args["b_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    print("Building model...")
    model, _ = build_model(args, model_params)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), args)

    best_loss = float("inf")
    best_ckpt = os.path.join(args["checkpoint_path"], "subset_train_best.pth")
    last_ckpt = os.path.join(args["checkpoint_path"], "subset_train_last.pth")

    print("\n--- TRAINING ON SUBSET ---")
    for epoch in range(1, args["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            x = batch["representation"].to(device).float()
            y = batch["target"].to(device).float()

            out = model(x)
            out = out.view(x.shape[0], args["clip_len"] - 1, 12)

            loss = compute_loss(out, y, criterion, args)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if batch_idx == 0 or batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch:03d} | batch {batch_idx:03d}/{len(loader)} | "
                    f"loss = {loss.item():.6f}"
                )

        epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch:03d} DONE | mean loss = {epoch_loss:.6f}")

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_loss,
            "args": args,
            "model_params": model_params,
        }

        torch.save(state, last_ckpt)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state["best_val"] = best_loss
            torch.save(state, best_ckpt)
            print(f"Saved new best checkpoint to: {best_ckpt}")

    print("\n--- EVALUATION ON SAME SUBSET ---")
    model.eval()

    all_pred_steps = []
    all_gt_steps = []
    all_times = []

    with torch.no_grad():
        total_eval_loss = 0.0
        n_eval_batches = 0

        for batch_idx, batch in enumerate(eval_loader):
            x = batch["representation"].to(device).float()
            y = batch["target"].to(device).float()
            anchors = batch["anchors_us"]

            out = model(x)
            out = out.view(x.shape[0], args["clip_len"] - 1, 12)

            loss = compute_loss(out, y, criterion, args)
            total_eval_loss += loss.item()
            n_eval_batches += 1

            for i in range(out.shape[0]):
                pred_i = out[i].cpu()
                gt_i = y[i].cpu()
                anc_i = anchors[i].cpu()

                # ricostruzione senza duplicare le transizioni overlappate
                if batch_idx == 0 and i == 0:
                    all_pred_steps.append(pred_i[0])
                    all_gt_steps.append(gt_i[0])
                    all_times.append(int(anc_i[1]))

                all_pred_steps.append(pred_i[-1])
                all_gt_steps.append(gt_i[-1])
                all_times.append(int(anc_i[-1]))

        eval_loss = total_eval_loss / max(n_eval_batches, 1)
        print(f"Eval loss on subset: {eval_loss:.6f}")

    all_pred_steps = torch.stack(all_pred_steps, dim=0)
    all_gt_steps = torch.stack(all_gt_steps, dim=0)

    pred_trans = extract_translation(all_pred_steps)
    gt_trans = extract_translation(all_gt_steps)

    pred_cum = torch.cumsum(pred_trans, dim=0)
    gt_cum = torch.cumsum(gt_trans, dim=0)

    step_mae = (pred_trans - gt_trans).abs().mean().item()
    cum_mae = (pred_cum - gt_cum).abs().mean().item()

    print(f"Step translation MAE: {step_mae:.6f}")
    print(f"Cumulative displacement MAE: {cum_mae:.6f}")

    print("\nFirst 5 predicted translations:")
    print(pred_trans[:5])

    print("\nFirst 5 GT translations:")
    print(gt_trans[:5])

    t0 = all_times[0]
    time_sec = [(t - t0) / 1e6 for t in all_times]

    plt.figure(figsize=(10, 6))
    plt.plot(time_sec, gt_cum[:, 0].numpy(), label="GT x")
    plt.plot(time_sec, pred_cum[:, 0].numpy(), label="Pred x")
    plt.plot(time_sec, gt_cum[:, 1].numpy(), label="GT y")
    plt.plot(time_sec, pred_cum[:, 1].numpy(), label="Pred y")
    plt.plot(time_sec, gt_cum[:, 2].numpy(), label="GT z")
    plt.plot(time_sec, pred_cum[:, 2].numpy(), label="Pred z")
    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative displacement")
    plt.title(f"Predicted vs GT displacement on subset ({subset_size} samples)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join("outputs", f"subset_{subset_size}_displacement.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"\nSaved plot to: {plot_path}")
    print(f"Saved best checkpoint to: {best_ckpt}")
    print(f"Saved last checkpoint to: {last_ckpt}")


if __name__ == "__main__":
    main()