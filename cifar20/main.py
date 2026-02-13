"""Main function for group attribution"""
import argparse
import itertools
import json
import os
import time

import torch
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from lightning.pytorch import seed_everything
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from configs.ddim_config import DDPMConfig
from src import constants
from src.datasets import create_dataset
from src.models.diffusion import GaussianDiffusion
from src.models.embedding import ConditionalDINOEmbedding, ConditionalEmbedding
from src.models.unet import Unet
from src.models.utils import get_named_beta_schedule
from src.utils import get_max_steps, print_args


def parse_args():
    """Parser function"""
    parser = argparse.ArgumentParser(description="test for CFG diffusion model")
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar20", help="name for dataset"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "pruned_retrain", "gd"],
        required=True,
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="data removal distribution",
        choices=["all", "loo", "shapley", "shapley_uniform", "datamodel", "percentile"],
        default="all",
    )
    parser.add_argument(
        "--removal_idx",
        type=int,
        help="class/seed to be removed for attribution",
        default=0,
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        default=0.5,
        help="alpha value for datamodel removal",
    )
    parser.add_argument(
        "--percentile_file",
        type=str,
        default=None,
        help="Path to JSON file containing percentile indices",
    )
    parser.add_argument(
        "--percentile_value",
        type=int,
        choices=[5, 10, 20, 30, 40],
        default=10,
        help="Percentile value to use (5, 10, 20, 30, or 40)",
    )
    parser.add_argument(
        "--percentile_type",
        type=str,
        choices=["top", "bottom"],
        default="bottom",
        help="Whether to use top or bottom percentile",
    )
    parser.add_argument(
        "--percentile_mode",
        type=str,
        choices=["remove", "keep"],
        default="remove",
        help=(
            "Mode for percentile removal: 'remove' to remove indices, "
            "'keep' to keep only those indices"
        ),
    )
    parser.add_argument(
        "--attr_method",
        type=str,
        default="shapley",
        help="attribution method",
    )
    parser.add_argument("--dtype", default=torch.float32)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
        "--multiplier", type=float, default=2.5, help="multiplier for warmup"
    )
    parser.add_argument(
        "--wd", type=float, default=1e-4, help="weight decay for optimizer"
    )

    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="decay rate for exponential moving average",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate before a backward/update pass.",
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

    # CFG sampling parameters
    parser.add_argument("--T", type=int, default=1000, help="timesteps for Unet model")

    parser.add_argument(
        "--drop_prob",
        type=float,
        default=0.1,
        help="probability to drop conditional embedding during training",
    )
    parser.add_argument(
        "--w",
        type=float,
        default=1.2,
        help="hyperparameters for classifier-free guidance strength",
    )
    parser.add_argument(
        "--v",
        type=float,
        default=0.3,
        help="hyperparameters for the variance of posterior distribution",
    )
    parser.add_argument(
        "--discrete_label",
        action="store_true",
        help="whether the conditional labels are discrete or continuous",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="threshold for classifier-free guidance",
    )
    # pruning params if used the pruned model
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
        "--genbatch", type=int, default=80, help="batch size for sampling process"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="sampling steps for DDIM"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0,
        help="eta for variance during DDIM sampling process",
    )
    parser.add_argument(
        "--select", type=str, default="linear", help="selection stragies for DDIM"
    )
    parser.add_argument("--ddim", action="store_true", help="whether to use DDIM")

    args = parser.parse_args()

    return args


def main(args):
    """Main function to train diffusion model"""
    print_args(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    seed_everything(args.opt_seed, workers=True)  # Seed for model optimization.

    device = accelerator.device

    # Load percentile indices from JSON file if using percentile removal
    percentile_indices = None
    if args.removal_dist == "percentile":
        if args.percentile_file is None:
            raise ValueError("--percentile_file must be specified.")

        with open(args.percentile_file, "r") as f:
            percentile_data = json.load(f)

        # Get indices based on percentile type and value
        key = f"{args.percentile_type}_{args.percentile_value}pct_indices"
        percentile_indices = percentile_data.get(key)

        if percentile_indices is None:
            raise ValueError(f"Key '{key}' not found in {args.percentile_file}")

        print(
            f"Loaded {len(percentile_indices)} indices from {args.percentile_type} "
            f"{args.percentile_value}% percentile"
        )

    train_dataset, clsnum = create_dataset(
        dataset_name=args.dataset,
        train=True,
        removal_dist=args.removal_dist,
        removal_idx=args.removal_idx,
        datamodel_alpha=args.datamodel_alpha,
        discrete_label=args.discrete_label,
        percentile_indices=percentile_indices,
        percentile_mode=args.percentile_mode,
    )
    num_workers = max(4, os.cpu_count() or 4)

    if args.dataset == "cifar20":
        config = {**DDPMConfig.cifar20_config}
    else:
        raise ValueError((f"dataset={args.dataset}"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 256) // accelerator.state.num_processes,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    training_steps = config.get("training_steps", {}).get(args.method, None)

    local_cfg = config.get("unet_config", {})

    if args.method == "retrain":
        net = Unet(
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
    else:
        # Load pruned model backbone
        pruned_model_path = os.path.join(
            args.outdir,
            f"seed{str(args.opt_seed)}",
            args.dataset,
            "pruned",
            "models",
            (
                f"pruner={args.pruner}"
                + f"_pruning_ratio={args.pruning_ratio}"
                + f"_threshold={args.thr}"
            ),
            f"ckpt_steps_{0:0>8}.pt",
        )
        pruned_model_ckpt = torch.load(
            pruned_model_path, weights_only=False, map_location="cpu"
        )
        net = pruned_model_ckpt["unet"]
        accelerator.print(f"Pruned U-Net initialized from {pruned_model_path}")

    # Compute and print model parameter count
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    accelerator.print(f"U-Net total parameters: {total_params:,}")
    accelerator.print(f"U-Net trainable parameters: {trainable_params:,}")

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

    # load last steps
    # Build conditional directory components
    removal_specific_dir = ""
    if args.removal_dist == "datamodel":
        removal_specific_dir = f"datamodel_alpha={args.datamodel_alpha}"
    elif args.removal_dist == "percentile":
        removal_specific_dir = (
            f"percentile_{args.percentile_type}_{args.percentile_value}pct_"
            f"{args.percentile_mode}_{args.attr_method}"
        )

    removal_idx_dir = (
        f"removal_idx={args.removal_idx}"
        if args.removal_dist not in ["all", "percentile"]
        else ""
    )

    model_outdir = os.path.join(
        args.outdir,
        f"seed{str(args.opt_seed)}",
        args.dataset,
        args.method,
        "models",
        args.removal_dist,
        f"discrete_label={args.discrete_label}",
        removal_specific_dir,
        removal_idx_dir,
        f"lr{args.lr}_wd{args.wd}_ema{args.ema_decay}_steps{training_steps}",
    )
    sample_outdir = os.path.join(
        args.outdir,
        f"seed{str(args.opt_seed)}",
        args.dataset,
        args.method,
        "samples",
        args.removal_dist,
        f"discrete_label={args.discrete_label}",
        removal_specific_dir,
        removal_idx_dir,
        f"lr{args.lr}_wd{args.wd}_ema{args.ema_decay}_steps{training_steps}",
    )

    total_steps_time = 0
    existing_steps = get_max_steps(model_outdir)
    ema_state = None

    if existing_steps is not None:
        checkpoint = torch.load(
            os.path.join(model_outdir, f"ckpt_{existing_steps}_checkpoint.pt"),
            map_location="cpu",
        )
        net.load_state_dict(checkpoint["net"])
        cemblayer.load_state_dict(checkpoint["cemblayer"])
        ema_state = checkpoint.get("net_ema", None)
        total_steps_time = checkpoint.get("total_steps_time", 0)
        last_steps = checkpoint.get("last_steps", 0)
        print(f"Resuming from {existing_steps} steps!")
    else:
        last_steps = 0

    betas = get_named_beta_schedule(num_diffusion_timesteps=args.T)

    # optimizer settings
    optimizer = torch.optim.AdamW(
        itertools.chain(net.parameters(), cemblayer.parameters()),
        lr=args.lr * args.multiplier,
        weight_decay=args.wd,
    )

    net, cemblayer, optimizer, train_loader = accelerator.prepare(
        net, cemblayer, optimizer, train_loader
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, training_steps // 10),
        num_training_steps=training_steps,
    )

    if last_steps != 0:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if accelerator.is_main_process:
        # Make the output directories once in the main process.
        os.makedirs(model_outdir, exist_ok=True)
        os.makedirs(sample_outdir, exist_ok=True)

    diffusion = GaussianDiffusion(
        dtype=args.dtype,
        model=net,
        betas=betas,
        w=args.w,
        v=args.v,
        device=device,
    )

    base_net = accelerator.unwrap_model(net)

    ema_unet = EMAModel(
        accelerator.unwrap_model(net).parameters(), ema_decay=args.ema_decay
    )

    if ema_state is not None:
        ema_unet.load_state_dict(ema_state)
        ema_unet.to(device=accelerator.device, dtype=next(base_net.parameters()).dtype)

    global_step = last_steps

    data_iter = iter(train_loader)
    ckpt_freq = config.get("ckpt_freq").get(args.method, 4000)
    steps_start_time = time.time()

    pbar = tqdm(
        total=training_steps,
        initial=global_step,
        dynamic_ncols=True,
        disable=not accelerator.is_main_process,
        desc="train steps",
    )

    while global_step < training_steps:
        try:
            if args.discrete_label:
                img, lab = next(data_iter)
            else:
                img, lab, emb = next(data_iter)
        except StopIteration:
            # re-create iterator when we exhaust the loader
            data_iter = iter(train_loader)

            if args.discrete_label:
                img, lab = next(data_iter)
            else:
                img, lab, emb = next(data_iter)

        diffusion.model.train()
        cemblayer.train()

        b = img.shape[0]
        x_0 = img.to(device)
        lab = lab.to(device)

        if args.discrete_label:
            cemb = cemblayer(lab.long(), args.drop_prob)
        else:
            cemb = cemblayer(emb, args.drop_prob)

        mask = torch.rand(b, device=device) < args.threshold
        cemb[mask] = 0

        with accelerator.accumulate(net):
            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                loss = diffusion.trainloss(x_0, cemb=cemb)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                optimizer.step()
                scheduler.step()
                ema_unet.step(accelerator.unwrap_model(net).parameters())
                global_step += 1

                pbar.update(1)

                # timings & tqdm info only on actual opt step
                steps_time = time.time() - steps_start_time
                total_steps_time += steps_time
                pbar.set_postfix(
                    {
                        "step": global_step,
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "dt": f"{steps_time:.2f}s",
                        "T": f"{total_steps_time:.1f}s",
                        "batch size": x_0.shape[0],
                        "img": list(x_0.shape[1:]),
                    }
                )
                steps_start_time = time.time()

                # evaluation and save checkpoint
                if (
                    accelerator.is_main_process
                    and global_step > 0
                    and (
                        (global_step) % ckpt_freq == 0
                        or (global_step) == training_steps
                    )
                ):
                    base_net = accelerator.unwrap_model(net)
                    ema_unet.store(base_net.parameters())  # save current params
                    ema_unet.copy_to(
                        base_net.parameters()
                    )  # load EMA params into `net`

                    diffusion.model.eval()
                    cemblayer.eval()
                    # generating samples
                    # pictures of same row belong to the same class

                    with torch.no_grad():
                        sampling_start_time = time.time()
                        per_class = 8
                        genbatch = clsnum * per_class
                        H = W = config["image_size"]

                        if args.discrete_label:

                            # labels: [0,0,...,1,1,...,clsnum-1,clsnum-1]
                            lab = (
                                torch.arange(clsnum, device=device)
                                .repeat_interleave(per_class)
                                .long()
                            )  # [genbatch]

                        else:
                            sigma = 0.05  # noise std; tweak 0.02–0.10
                            renorm = True  # set False to skip L2 renormalization

                            # 1) class means
                            class_means = []
                            for c in range(clsnum):
                                mask = torch.tensor(train_dataset.targets) == c
                                class_embs = (
                                    train_dataset.embeddings[mask].to(device).float()
                                )
                                m = class_embs.mean(dim=0)  # [dino_dim]
                                m = F.normalize(
                                    m, p=2, dim=0
                                )  # keep on unit sphere (optional)
                                class_means.append(m)

                            class_means = torch.stack(
                                class_means, dim=0
                            )  # [clsnum, dino_dim]

                            cond = class_means.repeat_interleave(
                                per_class, dim=0
                            )  # [genbatch, dino_dim]

                            # 3) add small Gaussian noise
                            cond = cond + sigma * torch.randn_like(cond)

                            # 4) renormalize so you stay near the DINO manifold
                            if renorm:
                                cond = F.normalize(cond, p=2, dim=-1)
                            lab = cond  # [genbatch, dino_dim]

                        cemb = cemblayer(lab)  # [genbatch, cdim]
                        genshape = (genbatch, config["in_ch"], H, W)

                        if args.discrete_label:
                            unconds = torch.zeros_like(lab).to(device)
                        else:
                            unconds = (
                                cemblayer.null.detach()
                                .unsqueeze(0)
                                .expand(len(lab), -1)
                                .to(device)
                            )

                        uncond_cemb = cemblayer(unconds, drop_prob=0.0)

                        if args.ddim:
                            generated = diffusion.ddim_sample(
                                genshape,
                                args.num_steps,
                                args.eta,
                                args.select,
                                cemb=cemb,
                                uncond_cemb=uncond_cemb,
                            )
                        else:
                            generated = diffusion.sample(genshape, cemb=cemb)

                        sampling_time = time.time() - sampling_start_time
                        print(f"Sampling time for {genbatch} samples: {sampling_time}.")
                        samples = (generated / 2 + 0.5).clamp(
                            0, 1
                        )  # [genbatch, 3, H, W]

                        save_image(
                            samples,
                            os.path.join(
                                sample_outdir, f"generated_{global_step}_grid.png"
                            ),
                            nrow=per_class,
                        )
                        steps_start_time = time.time()

                    # save checkpoints
                    checkpoint = {
                        "net": base_net.state_dict(),
                        "net_ema": ema_unet.state_dict(),
                        "cemblayer": accelerator.unwrap_model(cemblayer).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "total_steps_time": total_steps_time,
                        "last_steps": global_step,
                        "training_steps": training_steps,
                    }
                    accelerator.save(
                        checkpoint,
                        os.path.join(model_outdir, f"ckpt_{global_step}_checkpoint.pt"),
                    )
                    ema_unet.restore(base_net.parameters())
                    steps_start_time = time.time()

                if global_step >= training_steps:
                    break

    print(f"Total training time: {total_steps_time} seconds")

    return True


if __name__ == "__main__":
    args = parse_args()

    if main(args):
        print("Training done!")
