"""Calculate global model behavior scores for diffusion models."""
import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import torch
from filelock import FileLock
from lightning.pytorch import seed_everything
from torchvision.utils import save_image

import src.constants as constants
from configs.ddim_config import DDPMConfig
from src.attributions.global_scores import fid_score, inception_score
from src.datasets import TensorDataset, create_dataset
from src.models.diffusion import GaussianDiffusion
from src.models.diffusion_utils import _generate_samples, _load_model
from src.models.utils import get_named_beta_schedule


def _parse_hp_from_path(path: str):
    """Parse hyperparameters from checkpoint path."""
    m = re.search(
        r"lr(?P<lr>[\deE\.\-]+)_wd(?P<wd>[\deE\.\-]+)_steps(?P<ep>\d+)",
        path,
    )
    if not m:
        return {"lr": None, "weight_decay": None, "steps": None}
    return {
        "lr": float(m.group("lr")),
        "weight_decay": float(m.group("wd")),
        "steps": int(m.group("steps")),
    }


def parse_args():
    """Parser function"""
    parser = argparse.ArgumentParser(description="test for diffusion model")
    parser.add_argument(
        "--ckpt_path", type=str, help="checkpoint path for trained model", default=None
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar20", help="name for dataset"
    )
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "cfg"],
        required=True,
    )
    parser.add_argument(
        "--removal_dist", type=str, help="data removal distribution", default="all"
    )
    parser.add_argument(
        "--removal_idx", type=int, help="data removal class index", default=0
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="alpha for data-model removal",
        default=0.5,
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
        help="Mode for percentile removal: 'remove' to remove indices",
    )
    parser.add_argument(
        "--csv_path", type=str, help="path to the CSV file", required=True
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--save_img", action="store_true", help="whether to save generated images"
    )
    parser.add_argument(
        "--sample_outdir",
        type=str,
        help="output directory to save all the generated images",
        default=None,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="batch size per device for training Unet model",
    )
    parser.add_argument(
        "--device", type=str, help="device used for computation", default="cuda:0"
    )
    parser.add_argument("--T", type=int, default=1000, help="timesteps for Unet model")
    parser.add_argument("--dtype", default=torch.float32)
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument(
        "--w",
        type=float,
        default=1.4,
        help="hyperparameters for classifier-free guidance strength",
    )
    parser.add_argument(
        "--v",
        type=float,
        default=0.3,
        help="hyperparameters for the variance of posterior distribution",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10240, help="number of samples for generation"
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
        "--discrete_label",
        action="store_true",
        help="whether to use discrete labels for the conditional embedding",
    )
    parser.add_argument(
        "--sampling_mode",
        type=str,
        choices=["retrain_uncond", "full_cond"],
        default="retrain_uncond",
        help="How to sample for FID: retrained unconditional vs full conditional.",
    )
    args = parser.parse_args()

    return args


def main(args):
    """Main function"""
    seed_everything(args.opt_seed, workers=True)  # Seed for model optimization.

    if args.dataset == "cifar20":
        config = {**DDPMConfig.cifar20_config}
        H = W = 32
        in_ch = 3
    else:
        raise ValueError((f"dataset={args.dataset}"))

    device = args.device

    # Load percentile indices from JSON file if using percentile removal
    percentile_indices = None
    if args.removal_dist == "percentile":
        if args.percentile_file is None:
            raise ValueError(
                "percentile_file must be specified when using removal_dist='percentile'"
            )

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

    # Try to load the model class from the `diffusers` package if available.
    local_cfg = config["unet_config"]
    net, cemblayer = _load_model(
        args.ckpt_path, device, local_cfg, clsnum, args.discrete_label
    )

    betas = get_named_beta_schedule(num_diffusion_timesteps=args.T)

    diffusion = GaussianDiffusion(
        dtype=args.dtype,
        model=net,
        betas=betas,
        w=args.w,
        v=args.v,
        device=device,
    )

    generated_samples = []

    diffusion.model.eval()
    cemblayer.eval()

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
    if args.save_img:
        os.makedirs(args.sample_outdir, exist_ok=True)

        for i_global, img in enumerate(generated_samples):
            outfile = os.path.join(
                args.sample_outdir,
                f"cifar20_sample_{i_global}.png",
            )
            save_image(img, outfile)

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

    if args.ckpt_path is None:
        ckpt_path_str = "None"
    else:
        ckpt_path_str = args.ckpt_path
    hp = _parse_hp_from_path(ckpt_path_str)
    row = {
        "dataset": args.dataset,
        "removal_dist": args.removal_dist,
        "removal_idx": args.removal_idx,
        "datamodel_alpha": args.datamodel_alpha,
        "percentile_type": args.percentile_type
        if args.removal_dist == "percentile"
        else None,
        "percentile_value": args.percentile_value
        if args.removal_dist == "percentile"
        else None,
        "percentile_mode": args.percentile_mode
        if args.removal_dist == "percentile"
        else None,
        "method": args.method,
        "opt_seed": args.opt_seed,
        "fid": fid_value,
        "inception_score": is_value,
        "time": endtime - starttime,
        "n_samples": args.n_samples,
        "guidance_w": args.w,
        "ddim": bool(args.ddim),
        "num_steps": args.num_steps,
        "steps": hp["steps"],
        "lr": hp["lr"],
        "weight_decay": hp["weight_decay"],
        "T": args.T,
        "ckpt_path": args.ckpt_path,
    }

    csv_path = Path(args.csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(csv_path) + ".lock")

    with lock:
        write_header = not csv_path.exists()
        df = pd.DataFrame([row])
        df.to_csv(csv_path, mode="a", index=False, header=write_header)


if __name__ == "__main__":
    args = parse_args()
    main(args)
