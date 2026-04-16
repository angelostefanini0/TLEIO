import torch
import numpy as np
import os
import torch.nn as nn
from .models.vit import VisionTransformer
from ..dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset
from functools import partial


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_token_info(img_size, patch_size, clip_len):
    if isinstance(patch_size, tuple):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = patch_size

    img_h, img_w = img_size
    grid_h = img_h // patch_h
    grid_w = img_w // patch_w
    spatial_tokens = grid_h * grid_w
    transformer_tokens = 1 + clip_len * spatial_tokens
    return grid_h, grid_w, spatial_tokens, transformer_tokens


def build_model(args, model_params):
    img_size = MultiEventVoxelClipDataset.get_downsampled_size(
        original_height=480,
        original_width=640,
        downsampling_factor=args["downsampling_factor"],
        patch_size=model_params["patch_size"],
    )
    grid_h, grid_w, spatial_tokens, transformer_tokens = compute_token_info(
        img_size=img_size,
        patch_size=model_params["patch_size"],
        clip_len=args["clip_len"],
    )

    print(
        "Token info | "
        f"downsampling={args['downsampling_factor']} | "
        f"img_size={img_size[0]}x{img_size[1]} | "
        f"patch_grid={grid_h}x{grid_w} | "
        f"spatial_tokens/frame={spatial_tokens} | "
        f"transformer_tokens/sample={transformer_tokens}"
    )

    # build and load model
    model = VisionTransformer(
                img_size=img_size,
                in_chans=args["num_bins"],
                num_classes=model_params["num_classes"], 
                patch_size=model_params["patch_size"],
                embed_dim=model_params["embed_dim"],
                depth=model_params["depth"],
                num_heads=model_params["heads"],
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                drop_rate=0.,
                attn_drop_rate=model_params["attn_dropout"],
                drop_path_rate=model_params["ff_dropout"],
                num_frames=args["clip_len"],
                attention_type=model_params["attention_type"]
            )

    if model_params["time_only"]:
        # for timesformer without spatial layers
        for name, module in model.named_modules():
            if hasattr(module, 'attn'):
                # del module.attn
                module.attn = torch.nn.Identity()

    # load checkpoint
    args["epoch_init"] = 1
    args["best_val"] = np.inf
    if args["checkpoint"] is not None:
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]), weights_only=False)
        args["epoch_init"] = checkpoint["epoch"] + 1
        args["best_val"] = checkpoint["best_val"]
        model.load_state_dict(checkpoint['model_state_dict'])

    # Wrap model for multi-GPU training if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel training - number of batches should be a multiple of {torch.cuda.device_count()}")
        model = nn.DataParallel(model)
    else:
        print("Using single GPU or CPU")

    # Move model to device (DataParallel handles GPU placement automatically)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_params = count_parameters(model)
    print(f"Number of Parameters: {n_params}")

    return model, args
