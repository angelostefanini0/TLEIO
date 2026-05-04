import os
import sys
import torch
import torch.distributed as dist
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import pickle
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.network.train import *
from learning.network.build_model import *
from learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset
from learning.dataloader.events_to_voxel.precomputed_voxel_clip import PrecomputedVoxelClipDataset
import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")
#

def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # general/data
    parser.add_argument("--root_dir", type=str, default="data/eds/processed")
    parser.add_argument("--val_root_dir", type=str, default="data/eds/processed_validation")
    parser.add_argument("--b_size", type=int, default=4, help="batch size")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="percentage to use as validation data")
    parser.add_argument("--clip_len", type=int, default=3,
                        help="number of frames in window")
    parser.add_argument("--delta_t_ms", type=int, default=50,
                        help="Duration of event aggregation for voxel creation") 
    parser.add_argument("--num_bins", type=int, default=5, help="number of bins in voxel grid")
    parser.add_argument("--downsampling_factor", type=float, default=1.0, 
                        help="downsampling factor for events image")
    parser.add_argument("--denoising", type=str2bool, default=False,
                        help="apply background-activity filtering before voxelization")
    parser.add_argument("--denoise_dt_us", type=int, default=1000,
                        help="temporal support window in microseconds for denoising")
    parser.add_argument("--denoise_radius", type=int, default=1,
                        help="spatial neighborhood radius for denoising")
    parser.add_argument("--denoise_min_supporters", type=int, default=1,
                        help="minimum recent neighboring events required to keep an event")
    parser.add_argument("--denoise_same_polarity_only", type=str2bool, default=False,
                        help="require denoising support to come from the same polarity")
    parser.add_argument("--derotate", type=str2bool, default=False,
                        help="derotate events into a reference frame at the voxel anchor")
    parser.add_argument("--derotation_slices", type=int, default=100,
                        help="number of temporal slices used for event-space derotation")
    parser.add_argument("--precomputed_voxels", type=str2bool, default=True,
                        help="read precomputed voxel .npy files instead of events.h5")
    parser.add_argument("--voxel_filename", type=str, default="derotated_voxels.npy",
                        help="precomputed voxel file name inside each sequence folder")
                       
    # optimization
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["Adam", "AdamW", "SGD", "Adagrad", "RAdam"])
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # training
    parser.add_argument("--epoch", type=int, default=100,
                        help="train iters each timestep")
    parser.add_argument("--pretrained_ViT", type=str2bool, default=False,
                        help="load weights from pre-trained ViT")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of workers for dataloader")
    parser.add_argument("--persistent_workers", type=str2bool, default=True,
                        help="Keep DataLoader workers alive across epochs when num_workers > 0")
    parser.add_argument("--prefetch_factor", type=int, default=2,
                        help="Number of prefetched batches per worker when num_workers > 0")
    parser.add_argument("--amp", type=str2bool, default=False,
                        help="Enable automatic mixed precision on CUDA")
    parser.add_argument("--amp_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"],
                        help="Mixed-precision dtype to use when --amp true")
    parser.add_argument("--profile_timing", type=str2bool, default=False,
                        help="measure average batch wait time versus compute time during training")
    parser.add_argument("--profile_warmup_batches", type=int, default=10,
                        help="number of initial training batches to ignore in timing statistics")

    # checkpoints
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints",
                        help="path to save checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="checkpoint file to resume from")

    # model params
    parser.add_argument("--embed_dim", type=int, default=384)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--attention_type", type=str, default="divided_space_time",
                        choices=["divided_space_time", "space_only", "joint_space_time", "time_only"])
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--ff_dropout", type=float, default=0.1)
    parser.add_argument("--time_only", type=str2bool, default=False)

    parsed = parser.parse_args()
    args = vars(parsed)

    model_params = {
        "embed_dim": args["embed_dim"],
        "patch_size": args["patch_size"],
        "attention_type": args["attention_type"],
        "num_frames": args["clip_len"],
        "num_classes": 3 * (args["clip_len"] - 1),
        "depth": args["depth"],
        "heads": args["heads"],
        "dim_head": args["dim_head"],
        "attn_dropout": args["attn_dropout"],
        "ff_dropout": args["ff_dropout"],
        "time_only": args["time_only"],
    }

    args["model_params"] = model_params

    return args


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://")

    args["distributed"] = distributed
    args["world_size"] = world_size
    args["rank"] = rank
    args["local_rank"] = local_rank
    args["is_main_process"] = rank == 0
    args["device"] = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    return args


def cleanup_distributed(args):
    if args.get("distributed", False) and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

        
if __name__ == "__main__":
    args = setup_distributed(parse_args())
    model_params = args["model_params"]

    try:
        # create checkpoints folder
        if args["is_main_process"] and not os.path.exists(args["checkpoint_path"]):
            os.makedirs(args["checkpoint_path"])
        if args["distributed"]:
            dist.barrier()

        if args["is_main_process"]:
            with open(os.path.join(args["checkpoint_path"], 'args.pkl'), 'wb') as f:
                pickle.dump(args, f)
            with open(os.path.join(args["checkpoint_path"], 'args.txt'), 'w') as f:
                f.write(json.dumps(args))

        # tensorboard writer
        TensorBoardWriter = (
            SummaryWriter(log_dir=args["checkpoint_path"])
            if args["is_main_process"]
            else None
        )

        # train and val dataloader
        if args["is_main_process"]:
            print("Using CUDA: ", torch.cuda.is_available())
            print("Loading data...")
        
    
    if args["precomputed_voxels"]:
        train_data = PrecomputedVoxelClipDataset(
            root_path=Path(args["root_dir"]),
            clip_len=args["clip_len"],
            num_bins=args["num_bins"],
            voxel_filename=args["voxel_filename"],
        )
        val_data = PrecomputedVoxelClipDataset(
            root_path=Path(args["val_root_dir"]),
            clip_len=args["clip_len"],
            num_bins=args["num_bins"],
            voxel_filename=args["voxel_filename"],
        )
    else:
        train_data = MultiEventVoxelClipDataset(
            root_path=Path(args["root_dir"]),
            delta_t_ms=args["delta_t_ms"],
            num_bins=args["num_bins"],
            clip_len=args["clip_len"],
            downsampling_factor=args["downsampling_factor"],
            patch_size=args["patch_size"],
            denoising=args["denoising"],
            denoise_dt_us=args["denoise_dt_us"],
            denoise_radius=args["denoise_radius"],
            denoise_min_supporters=args["denoise_min_supporters"],
            denoise_same_polarity_only=args["denoise_same_polarity_only"],
            derotate=args["derotate"],
            derotation_slices=args["derotation_slices"],
        )
        val_data = MultiEventVoxelClipDataset(
            root_path=Path(args["val_root_dir"]),
            delta_t_ms=args["delta_t_ms"],
            num_bins=args["num_bins"],
            clip_len=args["clip_len"],
            downsampling_factor=args["downsampling_factor"],
            patch_size=args["patch_size"],
            denoising=args["denoising"],
            denoise_dt_us=args["denoise_dt_us"],
            denoise_radius=args["denoise_radius"],
            denoise_min_supporters=args["denoise_min_supporters"],
            denoise_same_polarity_only=args["denoise_same_polarity_only"],
            derotate=args["derotate"],
            derotation_slices=args["derotation_slices"]
        )

        # Compute the mean and std only on training data
        train_data.compute_stats(list(range(len(train_data))))
        stats = {"mean": train_data.train_mean, 
                 "std": train_data.train_std}

        # Normalize validation targets with training statistics
        val_data.train_mean = train_data.train_mean
        val_data.train_std = train_data.train_std

        train_sampler = (
            DistributedSampler(train_data, shuffle=True, drop_last=True)
            if args["distributed"]
            else None
        )
        val_sampler = (
            DistributedSampler(val_data, shuffle=False, drop_last=False)
            if args["distributed"]
            else None
        )

        train_loader = DataLoader(
            train_data,
            batch_size=args["b_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args["persistent_workers"] and args["num_workers"] > 0,
            prefetch_factor=args["prefetch_factor"] if args["num_workers"] > 0 else None,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args["num_workers"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args["persistent_workers"] and args["num_workers"] > 0,
            prefetch_factor=args["prefetch_factor"] if args["num_workers"] > 0 else None,
            drop_last=False,
        )

        # build and load model
        if args["is_main_process"]:
            print("Building model...")
        model, args = build_model(args, model_params)

        # loss and optimizer
        criterion = torch.nn.MSELoss()
        optimizer = get_optimizer(model, args)
        scaler = GradScaler(
            "cuda",
            enabled=(
                args["amp"]
                and torch.cuda.is_available()
                and args["amp_dtype"] == "float16"
            ),
        )
        if args["checkpoint"] is not None:
            checkpoint = torch.load(
                os.path.join(args["checkpoint_path"], args["checkpoint"]),
                map_location=args["device"],
                weights_only=False,
            )
            scaler_state = checkpoint.get("scaler_state_dict")
            if scaler_state is not None and scaler.is_enabled():
                scaler.load_state_dict(scaler_state)

        # train network
        if args["is_main_process"]:
            print(20*"--" +  " Training " + 20*"--")
        train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            TensorBoardWriter,
            args,
            stats,
            train_sampler=train_sampler,
            scaler=scaler,
        )
    finally:
        cleanup_distributed(args)
           
