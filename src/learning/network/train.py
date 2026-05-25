import os
import time
from contextlib import nullcontext
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
import numpy as np


torch.manual_seed(2026)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def get_raw_model(model):
    return model.module if hasattr(model, "module") else model


def get_model_device(model) -> torch.device:
    return next(get_raw_model(model).parameters()).device


def reduce_mean(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return tensor.item()


def reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def maybe_cuda_synchronize() -> None:
    """Synchronize the default CUDA stream when timing GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def get_amp_dtype(args):
    amp_dtype = args.get("amp_dtype", "bfloat16")
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def autocast_context(args):
    if not args.get("amp", False) or not torch.cuda.is_available():
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=get_amp_dtype(args),
    )


def get_outputs_per_motion(args):
    return 6 if args.get("covariance", False) else 3


def val_epoch(model, val_loader, criterion, args, epoch):
    device = get_model_device(model)
    epoch_loss = 0.0
    num_batches = 0
    iterator = tqdm(val_loader, unit="batch") if is_main_process() else val_loader

    for batch in iterator:
        x = batch["representation" ] # B, C, T, H, W
        y = batch["target"] # B, T - 1, 3
        if is_main_process():
            iterator.set_description("Validating ")
        if torch.cuda.is_available():
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

        with autocast_context(args):
            # predict transformation
            outputs_per_motion = get_outputs_per_motion(args)
            estimated_transf = model(x.float())
            estimated_transf = estimated_transf.view(
                x.shape[0],
                args["clip_len"] - 1,
                outputs_per_motion,
            )

            # compute loss on the first three translation dimensions
            loss = compute_loss(estimated_transf, y, criterion, args, epoch)

        epoch_loss += loss.item()
        num_batches += 1

        if is_main_process():
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
            )

    total_loss = reduce_sum(epoch_loss, device)
    total_batches = reduce_sum(float(num_batches), device)
    return total_loss / max(total_batches, 1.0)


def train_epoch(model, train_loader, criterion, optimizer, epoch, tensorboard_writer, args, scaler=None):
    device = get_model_device(model)
    epoch_loss = 0.0
    num_batches = 0
    iter = (epoch - 1) * len(train_loader) + 1
    profile_timing = args.get("profile_timing", False) and is_main_process()
    profile_warmup_batches = max(0, int(args.get("profile_warmup_batches", 10)))
    waited_for_data_s = 0.0
    compute_s = 0.0
    measured_batches = 0
    measured_samples = 0
    epoch_start = time.perf_counter()
    batch_end = epoch_start

    iterator = tqdm(train_loader, unit="batch") if is_main_process() else train_loader
    for batch_idx, batch in enumerate(iterator):
        if is_main_process():
            iterator.set_description(f"Epoch {epoch}")
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

        with autocast_context(args):
            # predict transformation
            outputs_per_motion = get_outputs_per_motion(args)
            estimated_transf = model(x.float())
            estimated_transf = estimated_transf.view(
                x.shape[0],
                args["clip_len"] - 1,
                outputs_per_motion,
            )

            # compute loss on the first three translation dimensions
            loss = compute_loss(estimated_transf, y, criterion, args, epoch)

        # compute gradient and do optimizer step
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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

        epoch_loss += loss.item()
        num_batches += 1

        if is_main_process():
            if profile_timing and measured_batches > 0:
                avg_data_ms = 1000.0 * waited_for_data_s / measured_batches
                avg_compute_ms = 1000.0 * compute_s / measured_batches
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                    data_ms=f"{avg_data_ms:.1f}",
                    compute_ms=f"{avg_compute_ms:.1f}",
                )
            else:
                iterator.set_postfix(
                    loss=f"{loss.item():.4f}",
                )
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('training_loss', loss.item(), iter)
        iter += 1
        batch_end = time.perf_counter()

    total_loss = reduce_sum(epoch_loss, device)
    total_batches = reduce_sum(float(num_batches), device)
    avg_total = total_loss / max(total_batches, 1.0)

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
  

def train(model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args, stats, train_sampler=None, scaler=None):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    init = args["epoch_init"]
    best_val = args["best_val"]
    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(init, epochs +1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
            scaler=scaler,
        )

        # validate model
        val_loss = None
        if val_loader:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args, epoch)

        if is_main_process():
            if val_loss is not None:
                tqdm.write(
                    f"Epoch {epoch} | "
                    f"train loss={train_loss:.4f} | "
                    f"val loss={val_loss:.4f}"
                )
            else:
                tqdm.write(
                    f"Epoch {epoch} | "
                    f"train loss={train_loss:.4f}"
                )

            raw_model = get_raw_model(model)
            state = {
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                "best_val": best_val,
                "target_mean": stats["mean"],
                "target_std": stats["std"],
            }

            if val_loss is not None and val_loss < best_val:
                tqdm.write(
                    f"Saving new best model: val_loss {best_val:.6f} -> {val_loss:.6f}"
                )
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            if tensorboard_writer is not None and val_loss is not None:
                tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        if timing_summary is not None and is_main_process():
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
                if tensorboard_writer is not None:
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

        if is_main_process():
            # save checkpoint every 20 epochs
            if not epoch%20:
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))) 
            # save last checkpoint
            torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))  

            # log loss in TensorBoard
            if tensorboard_writer is not None:
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
        checkpoint = torch.load(
            os.path.join(args["checkpoint_path"], args["checkpoint"]),
            map_location=args.get("device", "cpu"),
            weights_only=False,
        )
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return optimizer


def compute_loss(y_hat, y, criterion, args, epoch):
    MIN_LOG_STD = np.log(1e-3)
    y = y.reshape(y.shape[0], args["clip_len"] - 1, 3).float()
    outputs_per_motion = get_outputs_per_motion(args)
    y_hat = y_hat.reshape(y_hat.shape[0], args["clip_len"] - 1, outputs_per_motion)
    pred = y_hat[..., :3]
    
    if not args.get("covariance", False) or epoch < args["transition_epoch"]:
        loss = criterion(pred, y)
    else:  # Use maximum likelihood loss with diagonal covariance
        pred_logstd = y_hat[..., 3:]
        pred_logstd = torch.maximum(pred_logstd, MIN_LOG_STD * torch.ones_like(pred_logstd))
        loss = ((pred - y).pow(2)) / (2 * torch.exp(2 * pred_logstd)) + pred_logstd
    
    return loss.sum()  # Sum to scalar for loss.item()
    
