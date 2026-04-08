import os
import sys
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pickle
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.learning.network.train import get_optimizer, train
from src.learning.network.build_model import build_model
from src.learning.dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"true", "1", "yes", "y"}:
        return True
    if v.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    # general/data
    parser.add_argument("--root_dir", type=str, default="data/eds/processed")
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
                       
    # optimization
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["Adam", "SGD", "Adagrad", "RAdam"])
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # training
    parser.add_argument("--epoch", type=int, default=100,
                        help="train iters each timestep")
    parser.add_argument("--weighted_loss", type=float, default=None,
                        help="float to weight angles in loss function")
    parser.add_argument("--pretrained_ViT", type=str2bool, default=False,
                        help="load weights from pre-trained ViT")
    parser.add_argument("--num_workers", type=int, default=0, 
                        help="Number of workers for dataloader")

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
        "num_classes": 9 * (args["clip_len"] - 1),
        "depth": args["depth"],
        "heads": args["heads"],
        "dim_head": args["dim_head"],
        "attn_dropout": args["attn_dropout"],
        "ff_dropout": args["ff_dropout"],
        "time_only": args["time_only"],
    }

    args["model_params"] = model_params

    return args

        
if __name__ == "__main__":
    args = parse_args()
    model_params = args["model_params"]

    # create checkpoints folder
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], 'args.txt'), 'w') as f:
        f.write(json.dumps(args))

    # tensorboard writer
    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])
    #TODO: Investigate if we need to do normalization within a batch or at the dataset level
    # preprocessing operation
    # preprocess = transforms.Compose([
    #     transforms.Resize((model_params["image_size"])),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.34721234, 0.36705238, 0.36066107],
    #         std=[0.30737526, 0.31515116, 0.32020183]),
    # ])

    # train and val dataloader
    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")
    
    dataset = MultiEventVoxelClipDataset(
        root_path=Path(args["root_dir"]),
        delta_t_ms=args["delta_t_ms"],
        num_bins=args["num_bins"],
        clip_len=args["clip_len"],
        downsampling_factor=args["downsampling_factor"],
        patch_size=args["patch_size"],
    )
    
    nb_val = round(args["val_split"] * len(dataset))

    train_data, val_data = random_split(dataset, [len(dataset) - nb_val, nb_val]) #generator=torch.Generator().manual_seed(2))
    
    #Compute the mean and std of the train split to normalize targets
    dataset.compute_stats(train_data.indices)
    stats = {"mean": dataset.train_mean, 
             "std": dataset.train_std}

    train_loader = DataLoader(
        train_data,
        batch_size=args["b_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # build and load model
    print("Building model...")
    model, args = build_model(args, model_params)
    

    # loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), args)

    # train network
    print(20*"--" +  " Training " + 20*"--")
    train(model, train_loader, val_loader, criterion, optimizer, TensorBoardWriter, args, stats)
