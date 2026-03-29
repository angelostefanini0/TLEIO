import torch
import numpy as np
import os
import torch.nn as nn
from learning.network.models.vit import VisionTransformer
from functools import partial


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(args, model_params):
    # build and load model
    model = VisionTransformer(
                img_size=(480,640),
                in_chans=args["num_bins"],
                num_classes=(args["clip_len"] - 1) * 7,
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
        checkpoint = torch.load(os.path.join(args["checkpoint_path"], args["checkpoint"]))
        args["epoch_init"] = checkpoint["epoch"] + 1
        args["best_val"] = checkpoint["best_val"]
        model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.is_available():
        model.cuda()
    
    return model, args
