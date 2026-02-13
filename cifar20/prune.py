"""Pruning diffusion models"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch_pruning as tp
from accelerate import Accelerator
from diffusers.models.attention import Attention
from diffusers.models.resnet import Downsample2D, Upsample2D
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader

import src.constants as constants
from configs.ddim_config import DDPMConfig
from src.datasets import create_dataset
from src.models.embedding import ConditionalDINOEmbedding, ConditionalEmbedding
from src.models.unet import Unet
from src.utils import get_max_steps


def parse_args():
    """Parsing arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load", type=str, help="path for loading pre-trained model", default=None
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        default=None,
    )
    parser.add_argument(
        "--discrete_label",
        action="store_true",
        help="whether the conditional labels are discrete or continuous",
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )

    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )

    parser.add_argument(
        "--pruning_ratio",
        type=float,
        help="ratio for remaining parameters.",
        default=0.3,
    )

    parser.add_argument(
        "--pruner",
        type=str,
        default="magnitude",
        choices=["taylor", "random", "magnitude", "reinit", "diff-pruning"],
    )
    parser.add_argument(
        "--thr", type=float, default=0.05, help="threshold for diff-pruning"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="The dropout rate for fine-tuning."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="number of diffusion steps for generating images",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1000,
        help="number of diffusion steps during training",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate before a backward/update pass.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="inverse gamma value for EMA decay",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="power value for EMA decay",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="maximum decay magnitude EMA",
    )
    return parser.parse_args()


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


def main(args):
    """Main function for pruning and fine-tuning."""
    # loading images for gradient-based pruning

    seed_everything(args.opt_seed, workers=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print_args(args)

    if args.dataset in ["cifar20"]:
        config = {**DDPMConfig.cifar20_config}
        example_inputs = {
            "x": torch.randn(1, 3, 32, 32).to(device),
            "t": torch.ones((1,)).float().to(device),
            "cemb": torch.zeros((1,)).long().to(device),
        }
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
        example_inputs = {
            "x": torch.randn(1, 3, 256, 256).to(device),
            "t": torch.ones((1,)).long().to(device),
        }
    else:
        raise ValueError(f"dataset={args.dataset} is not one of)")

    train_dataset, clsnum = create_dataset(
        dataset_name=args.dataset,
        train=True,
        discrete_label=args.discrete_label,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 256) // accelerator.state.num_processes,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    local_cfg = config.get("unet_config", {})
    model = Unet(
        in_ch=local_cfg.get("in_ch"),
        mod_ch=local_cfg.get("mod_ch"),
        out_ch=local_cfg.get("out_ch"),
        ch_mul=local_cfg.get("ch_mul"),
        num_res_blocks=local_cfg.get("num_res_blocks"),
        cdim=local_cfg.get("cdim"),
        use_conv=local_cfg.get("use_conv"),
        droprate=local_cfg.get("droprate"),
        dtype=local_cfg.get("dtype"),
    )
    if args.discrete_label:
        cemblayer = ConditionalEmbedding(
            clsnum, local_cfg.get("cdim"), local_cfg.get("cdim")
        )
    else:
        cemblayer = ConditionalDINOEmbedding(
            dim=local_cfg.get("cdim"),
            d_model=local_cfg.get("cdim"),
            dino_dim=train_dataset.embeddings.shape[1],
        )

    clean_images = next(iter(train_dataloader))
    if isinstance(clean_images, (list, tuple)):
        clean_images = clean_images[0]
    clean_images = clean_images.to(device)

    pre_trained_path = args.load

    # Loading pretrained(locked) model
    accelerator.print("Loading pretrained model from {}".format(pre_trained_path))

    # load model and scheduler

    existing_steps = get_max_steps(args.load)

    if existing_steps is not None:
        checkpoint = torch.load(
            os.path.join(pre_trained_path, f"ckpt_{existing_steps}_checkpoint.pt"),
            map_location="cpu",
        )
        model.load_state_dict(checkpoint["net"])
        cemblayer.load_state_dict(checkpoint["cemblayer"])
        print(f"Resuming from {existing_steps} steps!")
    else:
        raise ValueError(f"No pre-trained checkpoints found at {args.load}")

    model.to(device)
    cemblayer.to(device)

    example_inputs["cemb"] = cemblayer(example_inputs["cemb"])

    pruning_params = (
        f"pruner={args.pruner}_pruning_ratio={args.pruning_ratio}_threshold={args.thr}"
    )

    if args.pruning_ratio > 0:
        if args.pruner == "taylor":
            imp = tp.importance.TaylorImportance(
                multivariable=True
            )  # standard first-order taylor expansion
        elif args.pruner == "random" or args.pruner == "reinit":
            imp = tp.importance.RandomImportance()
        elif args.pruner == "magnitude":
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == "diff-pruning":
            imp = tp.importance.TaylorImportance(
                multivariable=False
            )  # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = []
        channel_groups = {}

        if args.dataset == "cifar20":
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    # Stem
                    if m is model.downblocks[0][0]:
                        ignored_layers.append(m)
                    # Output head
                    elif m is model.out[-1]:
                        ignored_layers.append(m)
                    # Attention convs
                    elif any(
                        kw in name for kw in [".proj_q", ".proj_k", ".proj_v", ".proj"]
                    ):
                        ignored_layers.append(m)
                    # ALL middle blocks
                    elif "middleblocks" in name or "middle" in name:
                        ignored_layers.append(m)
        elif args.dataset == "celeba":  # Prunig attention for LDM
            for m in model.modules():
                if isinstance(m, Attention):
                    channel_groups[m.to_q] = m.heads
                    channel_groups[m.to_k] = m.heads
                    channel_groups[m.to_v] = m.heads

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels = m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        accelerator.print(model)
        accelerator.print(
            "#Params: {:.4f} M => {:.4f} M".format(base_params / 1e6, params / 1e6)
        )
        accelerator.print(
            "#MACS: {:.4f} G => {:.4f} G".format(base_macs / 1e9, macs / 1e9)
        )
        model.zero_grad()
        del pruner

        if args.pruner == "reinit":

            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()

            reset_parameters(model)

    if args.pruning_ratio > 0:
        model_outdir = os.path.join(
            args.outdir, args.dataset, "pruned", "models", pruning_params
        )
        os.makedirs(model_outdir, exist_ok=True)

        # Here the entire pruned model has to be saved.
        torch.save(
            {
                "unet": accelerator.unwrap_model(model),
            },
            os.path.join(model_outdir, f"ckpt_steps_{0:0>8}.pt"),
        )
        accelerator.print(f"Pruned checkpoint saved at step {existing_steps}")

    accelerator.print("Done pruning!")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
