import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.learning.network.build_model import build_model
from src.learning.network.train import compute_loss, get_optimizer
from src.learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset


def print_prediction_summary(pred, target, name=""):
    print(f"\n--- {name} PREDICTION SUMMARY ---")
    print("pred shape  :", pred.shape)
    print("target shape:", target.shape)

    abs_err = (pred - target).abs()
    mae = abs_err.mean().item()
    max_err = abs_err.max().item()

    print(f"MAE     : {mae:.6f}")
    print(f"Max err : {max_err:.6f}")

    pred0 = pred[0].detach().cpu()
    target0 = target[0].detach().cpu()

    print("\nFirst sample prediction:")
    print(pred0)

    print("\nFirst sample target:")
    print(target0)

    print("\nFirst sample abs error:")
    print((pred0 - target0).abs())


def main():
    target_dim = 7  # quaternion (4) + translation (3)

    args = {
        "root_dir": "data/eds/processed",
        "clip_len": 3,
        "num_bins": 5,
        "delta_t_ms": 50,
        "b_size": 2,
        "checkpoint": None,
        "checkpoint_path": "checkpoints",
        "weighted_loss": None,
        "optimizer": "Adam",
        "lr": 1e-4,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "target_dim": target_dim,
    }

    model_params = {
        "embed_dim": 384,
        "patch_size": 16,
        "attention_type": "divided_space_time",
        "num_frames": args["clip_len"],
        "num_classes": target_dim * (args["clip_len"] - 1),
        "depth": 6,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.1,
        "ff_dropout": 0.1,
        "time_only": False,
    }

    save_test_checkpoint = True
    test_reload_checkpoint = True
    overfit_steps = 20

    os.makedirs(args["checkpoint_path"], exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}\n")

    print("Loading dataset...")
    dataset = MultiEventVoxelClipDataset(
        root_path=Path(args["root_dir"]),
        delta_t_ms=args["delta_t_ms"],
        num_bins=args["num_bins"],
        clip_len=args["clip_len"],
    )

    print(f"Dataset size: {len(dataset)}")
    assert len(dataset) > 0, "Dataset vuoto"

    sample = dataset[0]
    print("\n--- SINGLE SAMPLE ---")
    print("representation:", sample["representation"].shape)
    print("target:", sample["target"].shape)
    print("anchors:", sample["anchors_us"].shape)

    assert sample["target"].shape[-1] == target_dim, (
        f"Target dim mismatch: expected {target_dim}, got {sample['target'].shape[-1]}"
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args["b_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(iter(loader))
    x = batch["representation"]
    y = batch["target"]

    print("\n--- BATCH ---")
    print("x:", x.shape)
    print("y:", y.shape)

    assert y.shape[-1] == target_dim, (
        f"Batch target dim mismatch: expected {target_dim}, got {y.shape[-1]}"
    )
    assert y.shape[1] == args["clip_len"] - 1, (
        f"Target temporal dim mismatch: expected {args['clip_len'] - 1}, got {y.shape[1]}"
    )

    x = x.to(device).float()
    y = y.to(device).float()

    print("\nBuilding model...")
    model, _ = build_model(args, model_params)
    model = model.to(device)

    print("\n--- MODEL ---")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    print("\n--- FORWARD ---")
    model.train()
    out = model(x)
    print("raw output:", out.shape)

    B = x.shape[0]
    T = args["clip_len"]

    expected_out_dim = (T - 1) * target_dim
    assert out.shape[1] == expected_out_dim, (
        f"Model output dim mismatch: expected {expected_out_dim}, got {out.shape[1]}"
    )

    out = out.view(B, T - 1, target_dim)
    print("reshaped output:", out.shape)

    print("\n--- LOSS ---")
    criterion = torch.nn.MSELoss()
    loss = compute_loss(out, y, criterion, args)
    print("initial loss:", loss.item())

    print("\n--- BACKWARD ---")
    loss.backward()
    print("Backward pass OK")

    grad_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()
    print(f"Total grad norm: {grad_norm:.4f}")

    print("\n--- OVERFIT ONE BATCH ---")
    model.train()
    optimizer = get_optimizer(model.parameters(), args)

    with torch.no_grad():
        out0 = model(x).view(B, T - 1, target_dim)
        loss0 = compute_loss(out0, y, criterion, args).item()
    print(f"Start overfit loss: {loss0:.6f}")

    for step in range(1, overfit_steps + 1):
        out = model(x)
        out = out.view(B, T - 1, target_dim)
        loss = compute_loss(out, y, criterion, args)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            print(f"Step {step:03d} | loss = {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        out_final = model(x).view(B, T - 1, target_dim)
        loss_final = compute_loss(out_final, y, criterion, args).item()

    print(f"Final overfit loss: {loss_final:.6f}")

    if loss_final < loss0:
        print("Overfit check: OK, la loss è scesa.")
    else:
        print("Overfit check: ATTENZIONE, la loss non è scesa.")

    print_prediction_summary(out_final, y, name="TRAINED MODEL")

    ckpt_path = os.path.join(args["checkpoint_path"], "sanity_checkpoint.pth")

    if save_test_checkpoint:
        state = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": None,
            "sanity_loss_start": loss0,
            "sanity_loss_final": loss_final,
            "args": args,
            "model_params": model_params,
        }
        torch.save(state, ckpt_path)
        print(f"\nSaved test checkpoint to: {ckpt_path}")

    if test_reload_checkpoint:
        print("\n--- RELOAD CHECKPOINT TEST ---")

        reloaded_model, _ = build_model(args, model_params)
        reloaded_model = reloaded_model.to(device)

        checkpoint = torch.load(ckpt_path, map_location=device)
        reloaded_model.load_state_dict(checkpoint["model_state_dict"])
        reloaded_model.eval()

        with torch.no_grad():
            out_reload = reloaded_model(x).view(B, T - 1, target_dim)
            reload_loss = compute_loss(out_reload, y, criterion, args).item()

        print(f"Reloaded model loss: {reload_loss:.6f}")

        diff = (out_reload - out_final).abs().max().item()
        print(f"Max difference vs saved model output: {diff:.10f}")

        print_prediction_summary(out_reload, y, name="RELOADED MODEL")

    print("\n✅ SANITY + OVERFIT + CHECKPOINT TEST COMPLETED")


if __name__ == "__main__":
    main()