"""Retraining approximation (unlearning) for obtaining subsets"""
import argparse
import itertools
import os
import time
from copy import deepcopy
from pathlib import Path

import clip
import lpips
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from filelock import FileLock
from lightning.pytorch import seed_everything
from peft import LoraConfig, get_peft_model
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    structural_similarity,
)
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from configs.ddim_config import DDPMConfig
from src import constants
from src.attributions.global_scores import fid_score, inception_score
from src.datasets import TensorDataset, create_dataset
from src.models.diffusion import GaussianDiffusion
from src.models.diffusion_utils import _generate_samples, _load_model
from src.models.embedding import ConditionalDINOEmbedding, ConditionalEmbedding
from src.models.unet import Unet
from src.models.utils import get_named_beta_schedule
from src.utils import print_args


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
        "--batch_size", type=int, help="batch size of dataloader", default=64
    )
    parser.add_argument(
        "--load", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--csv_path", type=str, help="path to the CSV file", required=True
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar20", help="name for dataset"
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "pruned_retrain", "gd", "lora"],
        required=True,
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="data removal distribution",
        choices=["all", "loo", "shapley", "shapley_uniform", "datamodel"],
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
        "--lora_rank", type=int, help="rank of matrix for LORA", default=16
    )
    parser.add_argument(
        "--lora_dropout", type=float, help="rank of matrix for LORA", default=0.05
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
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="directory path of reference samples, from a dataset or a diffusion model",
        default=None,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10240,
        help="number of samples to generated for evaluation.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg"],
        help="Network to use for LPIPS computation",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B/32",
        help="CLIP model to use for distance computation",
    )
    parser.add_argument(
        "--eval_batch",
        type=int,
        default=32,
        help="Batch size for LPIPS and CLIP evaluation",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        choices=["retrain_uncond", "full_cond"],
        default="retrain_uncond",
        help="How to sample for FID: retrained unconditional vs full conditional.",
    )
    parser.add_argument(
        "--device", type=str, help="device used for computation", default="cuda:0"
    )
    parser.add_argument(
        "--full_retrain_path",
        type=str,
        default=None,
        help="Path to full retrain model checkpoint for local behavior comparison",
    )
    parser.add_argument(
        "--n_local_samples",
        type=int,
        default=128,
        help="Number of samples for local behavior comparison",
    )
    args = parser.parse_args()

    return args


def _lpips_paired(lpips_net, x1, x2, batch_size=32):
    """Compute LPIPS distance between paired images in batches."""
    N = x1.shape[0]

    lpips_distances = []
    for i in range(0, N, batch_size):
        batch_x1 = x1[i : i + batch_size]
        batch_x2 = x2[i : i + batch_size]

        with torch.no_grad():
            dist = lpips_net(batch_x1, batch_x2).squeeze()
            lpips_distances.append(dist)

    return torch.cat(lpips_distances, dim=0)


def _clip_paired_cosdist(clip_model, preprocess, x1, x2, batch_size=32):
    """Compute CLIP cosine distance between paired images in batches."""

    N = x1.shape[0]
    device = x1.device

    # CLIP expects 224x224 RGB images
    target_size = 224

    cos_distances = []
    for i in range(0, N, batch_size):
        batch_x1 = x1[i : i + batch_size]
        batch_x2 = x2[i : i + batch_size]

        # Resize to 224x224
        batch_x1_resized = F.interpolate(
            batch_x1,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        batch_x2_resized = F.interpolate(
            batch_x2,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize using CLIP's normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(
            1, 3, 1, 1
        )
        batch_x1_norm = (batch_x1_resized - mean) / std
        batch_x2_norm = (batch_x2_resized - mean) / std

        with torch.no_grad():
            feat1 = clip_model.encode_image(batch_x1_norm)
            feat2 = clip_model.encode_image(batch_x2_norm)

            # Normalize features
            feat1 = F.normalize(feat1, dim=-1)
            feat2 = F.normalize(feat2, dim=-1)

            # Cosine distance = 1 - cosine similarity
            cos_sim = (feat1 * feat2).sum(dim=-1)
            cos_dist = 1.0 - cos_sim
            cos_distances.append(cos_dist)

    return torch.cat(cos_distances, dim=0)


def main(args):
    """Main function to train diffusion model"""
    print_args(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    seed_everything(args.opt_seed, workers=True)  # Seed for model optimization.

    device = accelerator.device

    train_dataset, clsnum = create_dataset(
        dataset_name=args.dataset,
        train=True,
        removal_dist=args.removal_dist,
        removal_idx=args.removal_idx,
        datamodel_alpha=args.datamodel_alpha,
        discrete_label=args.discrete_label,
    )
    num_workers = max(4, os.cpu_count() or 4)

    if args.dataset == "cifar20":
        config = {**DDPMConfig.cifar20_config}
        H = W = 32
        in_ch = 3
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

    if args.method in ["retrain", "lora"]:
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
            f"seed{args.opt_seed}",
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

    total_steps_time = 0
    last_steps = 0

    ema_state = None

    if args.load is not None:
        checkpoint = torch.load(
            args.load,
            map_location="cpu",
        )
        net.load_state_dict(checkpoint["net"])
        cemblayer.load_state_dict(checkpoint["cemblayer"])
        ema_state = checkpoint.get("net_ema", None)
        print(f"Resuming from {args.load}!")
    else:
        raise ValueError("Please provide pretrained a checkpoint for unlearning.")

    betas = get_named_beta_schedule(num_diffusion_timesteps=args.T)

    ema_unet = EMAModel(net.parameters(), ema_decay=args.ema_decay)

    if ema_state is not None:
        ema_unet.load_state_dict(ema_state)
        ema_unet.to(device=accelerator.device, dtype=next(net.parameters()).dtype)

    if args.method in ["lora"]:
        # Initialize LORA adapter

        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=["proj_q", "proj_k", "proj_v"],  # Match AttnBlock modules
            lora_alpha=32,
            lora_dropout=args.lora_dropout,
        )
        net = get_peft_model(net, lora_config)
        net.print_trainable_parameters()

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

    diffusion = GaussianDiffusion(
        dtype=args.dtype,
        model=net,
        betas=betas,
        w=args.w,
        v=args.v,
        device=device,
    )

    global_step = last_steps

    data_iter = iter(train_loader)
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

                base = accelerator.unwrap_model(net)
                if args.method in ["lora"]:
                    # Merge LORA to the base model

                    merged_model = deepcopy(base)
                    merged_model.merge_and_unload()
                    ema_unet.step(merged_model.parameters())
                    del merged_model
                else:
                    ema_unet.step(base.parameters())

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

                if global_step >= training_steps:
                    break

    print(f"Total training time: {total_steps_time} seconds")

    if args.method in ["lora"]:
        net = net.merge_and_unload()
        accelerator.print("LoRA adapters merged into base model for inference")

    final_net = accelerator.unwrap_model(net)
    ema_unet.copy_to(final_net.parameters())

    # Update diffusion model reference
    diffusion.model = final_net
    diffusion.model.eval()

    N = args.n_samples

    if args.sampling_mode == "retrain_uncond":
        # Retrained model, unconditional (NULL embedding)
        if args.discrete_label:
            # assume 0 is the null id used in training
            labels = torch.zeros(N, dtype=torch.long, device=device)
        else:
            # continuous: expand the learned null embedding
            labels = cemblayer.null.detach().unsqueeze(0).expand(N, -1)
    else:
        # Full model, conditional
        if args.discrete_label:
            remained_targets = sorted(set(train_dataset.targets))
            # empirical: frequency in train_dataset
            counts = torch.tensor(
                [
                    (torch.tensor(train_dataset.targets) == c).sum().item()
                    for c in remained_targets
                ],
                device=device,
                dtype=torch.float32,
            )
            probs = counts / counts.sum().clamp_min(1)
            idx = torch.multinomial(probs, num_samples=N, replacement=True)
            labels = torch.tensor(remained_targets, device=device, dtype=torch.long)[
                idx
            ]  # (N,)
        else:
            # continuous embeddings: sample rows from subset embedding bank

            label_gen = torch.Generator(device=device)
            label_gen.manual_seed(args.opt_seed + args.removal_idx)

            emb_bank = train_dataset.embeddings.to(device)  # [M, D]
            M = emb_bank.size(0)
            idx = torch.randint(0, M, (N,), generator=label_gen, device=device)
            labels = emb_bank[idx]

    starttime = time.time()

    generated_samples = _generate_samples(
        args,
        cemblayer,
        diffusion,
        labels,
        in_ch,
        H,
        W,
        device,
    )
    endtime = time.time()

    images_dataset = TensorDataset(generated_samples)

    is_value = inception_score.eval_is(
        images_dataset, args.batch_size, resize=True, normalize=True
    )

    fid_value = fid_score.calculate_fid(
        args.dataset,
        images_dataset,
        args.batch_size,
        args.device,
        args.reference_dir,
    )

    print(f"FID score: {fid_value};" f"inception score: {is_value}")

    # Local model behavior comparison
    local_rows = []
    if args.full_retrain_path is not None:
        print("\nComputing local model behavior comparison...")

        # Load full retrain model
        full_net, full_cemb = _load_model(
            args.full_retrain_path,
            device,
            config["unet_config"],
            clsnum,
            args.discrete_label,
        )
        full_net.eval()
        full_cemb.eval()

        # Create diffusion model for full retrain
        full_diffusion = GaussianDiffusion(
            dtype=args.dtype,
            model=full_net,
            betas=betas,
            w=args.w,
            v=args.v,
            device=device,
        )

        # Prepare labels for local comparison
        N_local = args.n_local_samples
        sample_indices = torch.randperm(len(train_dataset))[:N_local]

        if args.discrete_label:
            local_labels = torch.tensor(
                [train_dataset.targets[i] for i in sample_indices], device=device
            ).long()
            current_cemb_batched = cemblayer(local_labels)
            full_cemb_batched = full_cemb(local_labels)
            current_uncond = cemblayer(
                torch.zeros(N_local, dtype=torch.long, device=device)
            )
            full_uncond = full_cemb(
                torch.zeros(N_local, dtype=torch.long, device=device)
            )
        else:
            local_labels = train_dataset.embeddings[sample_indices].to(device).float()
            current_cemb_batched = torch.stack(
                [cemblayer(label) for label in local_labels], dim=0
            )
            full_cemb_batched = torch.stack(
                [full_cemb(label) for label in local_labels], dim=0
            )
            current_uncond = cemblayer.null.detach().unsqueeze(0).expand(N_local, -1)
            full_uncond = full_cemb.null.detach().unsqueeze(0).expand(N_local, -1)

        # Generate samples with same noise
        with torch.inference_mode():
            gen = torch.Generator(device=device)
            gen.manual_seed(args.opt_seed + args.removal_idx)
            state = gen.get_state()

            x_current = diffusion.ddim_sample(
                (N_local, in_ch, H, W),
                num_steps=args.num_steps,
                eta=args.eta,
                select=args.select,
                cemb=current_cemb_batched,
                uncond_cemb=current_uncond,
                generator=gen,
            )

            gen.set_state(state)
            x_full = full_diffusion.ddim_sample(
                (N_local, in_ch, H, W),
                num_steps=args.num_steps,
                eta=args.eta,
                select=args.select,
                cemb=full_cemb_batched,
                uncond_cemb=full_uncond,
                generator=gen,
            )

        # Convert to [0, 1] range
        def to01(x):
            return (x / 2 + 0.5).clamp(0, 1)

        x_current_01 = to01(x_current).detach().cpu()
        x_full_01 = to01(x_full).detach().cpu()

        # Convert to numpy for metrics
        x_current_np = x_current_01.permute(0, 2, 3, 1).numpy()  # (N, H, W, C)
        x_full_np = x_full_01.permute(0, 2, 3, 1).numpy()

        # Compute R² across all images
        r2_all = r2_score(x_current_np.reshape(-1), x_full_np.reshape(-1))
        print(f"R² (all local images): {r2_all:.6f}")

        # Initialize LPIPS and CLIP models
        print("Loading LPIPS and CLIP models...")
        lpips_net = lpips.LPIPS(net=args.lpips_net).to(device)
        clip_model, _ = clip.load(args.clip_model, device=device)
        clip_model.eval()

        # Compute LPIPS and CLIP distances (in batches)
        # LPIPS expects [-1, 1] range
        lpips_vec = _lpips_paired(
            lpips_net,
            x_current.to(device),
            x_full.to(device),
            batch_size=args.eval_batch,
        )

        # CLIP expects [0, 1] range
        clip_cosdist_vec = _clip_paired_cosdist(
            clip_model,
            None,
            x_current_01.to(device),
            x_full_01.to(device),
            batch_size=args.eval_batch,
        )

        print(f"LPIPS mean: {lpips_vec.mean():.6f}")
        print(f"CLIP cosine distance mean: {clip_cosdist_vec.mean():.6f}")

        # Compute RMS metrics
        rms_lpips = torch.sqrt(torch.mean(lpips_vec ** 2))
        rms_clip_drift = torch.sqrt(2 * torch.mean(clip_cosdist_vec))
        print(f"RMS LPIPS: {rms_lpips:.6f}")
        print(f"RMS CLIP drift: {rms_clip_drift:.6f}")

        # Compute per-image metrics
        for idx in range(N_local):
            mse = mean_squared_error(
                x_current_np[idx : idx + 1], x_full_np[idx : idx + 1]
            )
            nrmse = normalized_root_mse(
                x_current_np[idx : idx + 1], x_full_np[idx : idx + 1]
            )
            ssim = structural_similarity(
                x_current_np[idx], x_full_np[idx], channel_axis=-1, data_range=1.0
            )

            local_rows.append(
                {
                    "dataset": args.dataset,
                    "removal_dist": args.removal_dist,
                    "removal_idx": int(args.removal_idx),
                    "datamodel_alpha": float(args.datamodel_alpha),
                    "method": args.method,
                    "idx": int(idx),
                    "mse": float(mse),
                    "nrmse": float(nrmse),
                    "ssim": float(ssim),
                    "lpips": float(lpips_vec[idx].item()),
                    "clip_cosdist": float(clip_cosdist_vec[idx].item()),
                    "rms_lpips": float(rms_lpips.item()),
                    "rms_clip_drift": float(rms_clip_drift.item()),
                    "r2": float(r2_all),
                    "n_local_samples": int(N_local),
                    "opt_seed": int(args.opt_seed),
                    "guidance_w": float(args.w),
                }
            )

        print(f"Local comparison: MSE={mse:.6f}, NRMSE={nrmse:.6f}, SSIM={ssim:.6f}")

        # Save local metrics
        local_csv_path = (
            Path(args.outdir)
            / args.dataset
            / "local"
            / f"local_metrics_{args.removal_dist}_{args.method}.csv"
        )
        local_csv_path.parent.mkdir(parents=True, exist_ok=True)
        local_lock = FileLock(str(local_csv_path) + ".lock")

        with local_lock:
            write_header = not local_csv_path.exists()
            df_local = pd.DataFrame(local_rows)
            df_local.to_csv(local_csv_path, mode="a", index=False, header=write_header)

        print(f"Local metrics saved to {local_csv_path}")

    row = {
        "dataset": args.dataset,
        "removal_dist": args.removal_dist,
        "removal_idx": args.removal_idx,
        "datamodel_alpha": args.datamodel_alpha,
        "method": args.method,
        "training_steps": training_steps,
        "opt_seed": args.opt_seed,
        "fid": fid_value,
        "inception_score": is_value,
        "inference_time": endtime - starttime,
        "training_time": total_steps_time,
        "n_samples": args.n_samples,
        "guidance_w": args.w,
        "ddim": bool(args.ddim),
        "num_steps": args.num_steps,
        "T": args.T,
        "load": args.load,
    }

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(csv_path) + ".lock")

    with lock:
        write_header = not csv_path.exists()
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", index=False, header=write_header)

    return True


if __name__ == "__main__":
    args = parse_args()

    if main(args):
        print("Model behavior compute done!")
