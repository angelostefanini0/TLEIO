import torch
import numpy as np
import os
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from .models.vit import VisionTransformer
from ..dataloader.events_to_voxel.raw_to_clip import MultiEventVoxelClipDataset
from functools import partial


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_checkpoint_state_dict(state_dict):
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def resolve_checkpoint_path(args, checkpoint_name):
    checkpoint_path = os.fspath(checkpoint_name)
    if os.path.isabs(checkpoint_path) or os.path.exists(checkpoint_path):
        return checkpoint_path
    return os.path.join(args["checkpoint_path"], checkpoint_path)


def load_model_weights(model, checkpoint_file, device, strict=True):
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    state_dict = normalize_checkpoint_state_dict(checkpoint["model_state_dict"])
    return checkpoint, model.load_state_dict(state_dict, strict=strict)


def freeze_for_finetuning(model, trainable):
    if trainable == "all":
        return
    if trainable != "head":
        raise ValueError(f"Unsupported finetune_trainable value: {trainable}")

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


def filter_incompatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            skipped.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped.append(key)
            continue
        filtered[key] = value
    return filtered, skipped


def load_finetune_weights(model, checkpoint_file, device, reset_head=False):
    checkpoint = torch.load(
        checkpoint_file,
        map_location=device,
        weights_only=False,
    )
    state_dict = normalize_checkpoint_state_dict(checkpoint["model_state_dict"])
    if reset_head:
        state_dict = {
            key: value for key, value in state_dict.items()
            if not key.startswith("head.")
        }
        filtered_state_dict, skipped = filter_incompatible_state_dict(model, state_dict)
        skipped.extend(["head.weight", "head.bias"])
        load_result = model.load_state_dict(filtered_state_dict, strict=False)
        model.reset_classifier(model.num_classes)
        return checkpoint, load_result, skipped

    filtered_state_dict, skipped = filter_incompatible_state_dict(model, state_dict)
    load_result = model.load_state_dict(filtered_state_dict, strict=False)
    return checkpoint, load_result, skipped


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
    is_main_process = args.get("is_main_process", True)
    device = torch.device(args.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if args.get("input_height") is not None and args.get("input_width") is not None:
        img_size = (int(args["input_height"]), int(args["input_width"]))
        if img_size[0] % model_params["patch_size"] != 0 or img_size[1] % model_params["patch_size"] != 0:
            raise ValueError(
                "Precomputed voxel spatial size must be divisible by patch size. "
                f"img_size={img_size}, patch_size={model_params['patch_size']}."
            )
        input_source = "precomputed_voxel_shape"
    else:
        img_size = MultiEventVoxelClipDataset.get_downsampled_size(
            original_height=480,
            original_width=640,
            downsampling_factor=args["downsampling_factor"],
            patch_size=model_params["patch_size"],
        )
        input_source = "downsampling_factor"
    grid_h, grid_w, spatial_tokens, transformer_tokens = compute_token_info(
        img_size=img_size,
        patch_size=model_params["patch_size"],
        clip_len=args["clip_len"],
    )

    if is_main_process:
        print(
            "Token info | "
            f"downsampling={args['downsampling_factor']} | "
            f"input_source={input_source} | "
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
                attention_type=model_params["attention_type"],
                spatial_rope=model_params.get("spatial_rope", False),
                rope_frequency=model_params.get("rope_frequency", 100.0),
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
    if args["checkpoint"] is not None and args.get("finetune_checkpoint") is not None:
        raise ValueError("Use either --checkpoint to resume or --finetune_checkpoint to initialize, not both.")

    if args["checkpoint"] is not None:
        checkpoint_file = resolve_checkpoint_path(args, args["checkpoint"])
        checkpoint, _ = load_model_weights(
            model,
            checkpoint_file,
            device,
            strict=True,
        )
        args["epoch_init"] = checkpoint["epoch"] + 1
        args["best_val"] = checkpoint["best_val"]
    elif args.get("finetune_checkpoint") is not None:
        checkpoint_file = resolve_checkpoint_path(args, args["finetune_checkpoint"])
        _, load_result, skipped = load_finetune_weights(
            model,
            checkpoint_file,
            device,
            reset_head=args.get("finetune_reset_head", False),
        )
        freeze_for_finetuning(model, args.get("finetune_trainable", "all"))
        if is_main_process:
            print(f"Loaded finetune weights from: {checkpoint_file}")
            if skipped:
                print(f"Skipped incompatible finetune tensors: {skipped}")
            if load_result.missing_keys:
                print(f"Missing finetune keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected finetune keys: {load_result.unexpected_keys}")
            print(f"Finetune trainable scope: {args.get('finetune_trainable', 'all')}")

    model.to(device)

    if args.get("distributed", False):
        if device.type == "cuda":
            model = DDP(model, device_ids=[args["local_rank"]], output_device=args["local_rank"], find_unused_parameters=True)
        else:
            model = DDP(model)
        if is_main_process:
            print(f"Using DDP across {args['world_size']} process(es)")
    else:
        if is_main_process:
            print("Using single GPU or CPU")

    raw_model = model.module if hasattr(model, "module") else model
    n_params = count_parameters(raw_model)
    if is_main_process:
        print(f"Number of Parameters: {n_params}")

    return model, args
