import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm


torch.manual_seed(2026)


def maybe_cuda_synchronize() -> None:
    """Synchronize the default CUDA stream when timing GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0

    with tqdm(val_loader, unit="batch") as tepoch:
        for batch in tepoch:
            x = batch["representation" ] # B, C, T, H, W
            y = batch["target"] # B, T - 1, 3
            tepoch.set_description(f"Validating ")
            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict transformation
            estimated_transf = model(x.float()) # model returns [B, (T-1)*3]
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 3) # safe reshaping

            # compute loss
            loss = compute_loss(estimated_transf, y, criterion, args)

            # log loss
            epoch_loss += loss.item()

            tepoch.set_postfix(
                loss=f"{loss.item():.4f}",
            )
            
        avg_total = epoch_loss / len(val_loader)

    return avg_total


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1
    profile_timing = args.get("profile_timing", False)
    profile_warmup_batches = max(0, int(args.get("profile_warmup_batches", 10)))
    waited_for_data_s = 0.0
    compute_s = 0.0
    measured_batches = 0
    measured_samples = 0
    epoch_start = time.perf_counter()
    batch_end = epoch_start

    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")
            data_ready = time.perf_counter()
            data_wait_this_batch = data_ready - batch_end
            
            x = batch["representation"] # B, C, T, H, W
            y = batch["target"] # B, T - 1, 3

            if profile_timing:
                maybe_cuda_synchronize()
                compute_start = time.perf_counter()

            if torch.cuda.is_available():
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

            # predict transformation
            estimated_transf = model(x.float())
            estimated_transf = estimated_transf.view(x.shape[0], args["clip_len"] - 1, 3)

            # compute loss
            loss = compute_loss(estimated_transf, y, criterion, args)

            # compute gradient and do optimizer step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if profile_timing:
                maybe_cuda_synchronize()
                compute_this_batch = time.perf_counter() - compute_start
                if batch_idx >= profile_warmup_batches:
                    waited_for_data_s += data_wait_this_batch
                    compute_s += compute_this_batch
                    measured_batches += 1
                    measured_samples += int(x.shape[0])

            # log loss
            epoch_loss += loss.item()

            if profile_timing and measured_batches > 0:
                avg_data_ms = 1000.0 * waited_for_data_s / measured_batches
                avg_compute_ms = 1000.0 * compute_s / measured_batches
                tepoch.set_postfix(
                    loss=f"{loss.item():.4f}",
                    data_ms=f"{avg_data_ms:.1f}",
                    compute_ms=f"{avg_compute_ms:.1f}",
                )
            else:
                tepoch.set_postfix(
                    loss=f"{loss.item():.4f}",
                )
            # log tensorboard
            tensorboard_writer.add_scalar('training_loss', loss.item(), iter)
            iter += 1
            batch_end = time.perf_counter()
            
        
        avg_total = epoch_loss / len(train_loader)

    timing_summary = None
    if profile_timing:
        epoch_total_s = time.perf_counter() - epoch_start
        total_measured_s = waited_for_data_s + compute_s
        if measured_batches > 0 and total_measured_s > 0:
            timing_summary = {
                "epoch_total_s": epoch_total_s,
                "avg_data_wait_ms": 1000.0 * waited_for_data_s / measured_batches,
                "avg_compute_ms": 1000.0 * compute_s / measured_batches,
                "data_fraction": waited_for_data_s / total_measured_s,
                "compute_fraction": compute_s / total_measured_s,
                "measured_batches": measured_batches,
                "samples_per_s": measured_samples / total_measured_s,
                "batches_per_s": measured_batches / total_measured_s,
                "warmup_batches": profile_warmup_batches,
            }
        else:
            timing_summary = {
                "epoch_total_s": epoch_total_s,
                "measured_batches": measured_batches,
                "warmup_batches": profile_warmup_batches,
            }

    return avg_total, timing_summary
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args, stats):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    init = args["epoch_init"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(init, epochs +1):
        # training for one epoch
        model.train()
        train_loss, timing_summary = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            tensorboard_writer,
            args,
        )

        # validate model
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

            tqdm.write(
                f"Epoch {epoch} | "
                f"train loss={train_loss:.4f} | "
                f"val loss={val_loss:.4f}"
            )

            # if parallelized, you need to extract the raw model
            raw_model = model.module if hasattr(model, "module") else model
            
            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "best_val": best_val,
                "target_mean": stats["mean"],
                "target_std": stats["std"],
            }
            if val_loss < best_val:
                tqdm.write(
                    f"Saving new best model: val_loss {best_val:.6f} -> {val_loss:.6f}"
                )
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            # log validation loss in TensorBoard
            tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        if timing_summary is not None:
            if timing_summary.get("measured_batches", 0) > 0:
                tqdm.write(
                    "Timing | "
                    f"epoch={epoch} | "
                    f"avg_data_wait={timing_summary['avg_data_wait_ms']:.1f} ms | "
                    f"avg_compute={timing_summary['avg_compute_ms']:.1f} ms | "
                    f"data_fraction={100.0 * timing_summary['data_fraction']:.1f}% | "
                    f"throughput={timing_summary['batches_per_s']:.2f} batch/s "
                    f"({timing_summary['samples_per_s']:.2f} sample/s)"
                )
                tensorboard_writer.add_scalar(
                    "timing/avg_data_wait_ms",
                    timing_summary["avg_data_wait_ms"],
                    epoch,
                )
                tensorboard_writer.add_scalar(
                    "timing/avg_compute_ms",
                    timing_summary["avg_compute_ms"],
                    epoch,
                )
                tensorboard_writer.add_scalar(
                    "timing/data_fraction",
                    timing_summary["data_fraction"],
                    epoch,
                )
                tensorboard_writer.add_scalar(
                    "timing/batches_per_s",
                    timing_summary["batches_per_s"],
                    epoch,
                )

        # save checkpoint every 20 epochs
        if not epoch%20:
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))) 
        # save last checkpoint
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))  

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
    return


def build_param_groups(model, weight_decay):
    raw_model = model.module if hasattr(model, "module") else model
    skip_weight_decay = set()
    if hasattr(raw_model, "no_weight_decay"):
        skip_weight_decay = set(raw_model.no_weight_decay())

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        clean_name = name[7:] if name.startswith("module.") else name
        if (
            param.ndim == 1
            or clean_name.endswith(".bias")
            or clean_name in skip_weight_decay
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def get_optimizer(model, args):
    method = args["optimizer"]
    param_groups = build_param_groups(model, args["weight_decay"])

    # initialize the optimizer
    if method == "Adam":
        optimizer = optim.Adam(param_groups, lr=args["lr"])
    elif method == "AdamW":
        optimizer = optim.AdamW(param_groups, lr=args["lr"])
    elif method == "SGD":
        optimizer = optim.SGD(param_groups, lr=args["lr"],
                              momentum=args["momentum"])
    elif method == "RAdam":
        optimizer = optim.RAdam(param_groups, lr=args["lr"])
    elif method == "Adagrad":
        optimizer = optim.Adagrad(param_groups, lr=args["lr"])

    # load checkpoint
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]), weights_only=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def compute_loss(y_hat, y, criterion, args):
    y = y.reshape(y.shape[0], args["clip_len"] - 1, 3).float()
    y_hat = y_hat.reshape(y_hat.shape[0], args["clip_len"] - 1, 3)
    return criterion(y_hat, y)
    
