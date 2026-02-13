"""Generate images based on text prompts for pre-trained models."""
import argparse
import math
import os

import torch
from diffusers import DiffusionPipeline
from tqdm import tqdm

from src.datasets import create_dataset
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text to image generation.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/miniSD-diffusers",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Variant of the model files of the pretrained model identifier from "
            "huggingface.co/models, 'e.g.' fp16"
        ),
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="directory containing LoRA weights to load",
    )
    parser.add_argument(
        "--lora_steps",
        type=int,
        default=None,
        help="number of LoRA fine-tuning steps",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["artbench"],
        default="artbench",
        help="Dataset to determine which prompts to use for image generation",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        choices=["all", "uniform", "shapley", "datamodel", "loo", "aoi"],
        default="all",
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="alpha value for the datamodel removal distribution",
        default=None,
    )
    parser.add_argument(
        "--removal_seed",
        type=int,
        help="random seed for sampling from the removal distribution",
        default=0,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="number of images to generate per prompt",
    )
    parser.add_argument(
        "--ckpt_freq",
        type=int,
        default=25,
        help="number of saved images before saving a checkpoint",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="the resolution of generated image",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="guidance scale for image generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for reproducible image generation",
    )
    parser.add_argument(
        "--cls",
        type=str,
        default=None,
        help="generate images for this class",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output directory to save all the generated images in individual files",
        required=True,
    )
    parser.add_argument(
        "--sep_outdir",
        action="store_true",
        help="whether to store images geneated by each prompt in a separate directory",
    )
    return parser.parse_args()


def main(args):
    """Main function."""
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to("cuda")

    if args.lora_dir is not None:
        weight_name = "pytorch_lora_weights"
        if args.lora_steps is not None:
            weight_name += f"_{args.lora_steps}"
        weight_name += ".safetensors"
        pipeline.unet.load_attn_procs(args.lora_dir, weight_name=weight_name)
        weight_path = os.path.join(args.lora_dir, weight_name)
        print(f"LoRA weights loaded from {weight_path}")

    ckpt_file = f"ckpt_seed={args.seed}"
    if args.cls is not None:
        ckpt_file = f"{args.cls}_" + ckpt_file
    ckpt_file = os.path.join(args.outdir, ckpt_file)

    dataset, _ = create_dataset(
        "artbench",
        train=True,
        removal_dist=args.removal_dist,
        removal_idx=args.removal_seed,
        datamodel_alpha=args.datamodel_alpha,
    )

    n = len(dataset["caption"])
    probs = torch.full((n,), 1.0 / n)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.multinomial(
        probs, num_samples=args.num_images, replacement=True, generator=g
    )
    prompt_list = [dataset["caption"][i] for i in idx.tolist()]

    label_outdir = os.path.join(args.outdir)
    os.makedirs(os.path.join(args.outdir), exist_ok=True)

    batch_size = getattr(args, "batch_size", 32)
    device_str = str(pipeline.device)

    pbar = tqdm(total=len(prompt_list), desc="Generating for post-impressionism")
    i_global = 0

    for b in range(math.ceil(len(prompt_list) / batch_size)):
        batch_prompts = prompt_list[b * batch_size : (b + 1) * batch_size]
        # per-image deterministic generators
        gens = [
            torch.Generator(device=device_str).manual_seed(args.seed + i_global + j)
            for j in range(len(batch_prompts))
        ]

        ctx = (
            torch.autocast("cuda") if device_str.startswith("cuda") else torch.no_grad()
        )
        with torch.inference_mode(), ctx:
            out = pipeline(
                prompt=batch_prompts,
                num_inference_steps=100,
                generator=gens,
                height=args.resolution,
                width=args.resolution,
                guidance_scale=args.guidance_scale,
            )
        images = out.images

        # save
        for img in images:
            outfile = os.path.join(
                label_outdir,
                f"post_impressionism_seed={args.seed}_sample_{i_global}.png",
            )
            img.save(outfile)
            i_global += 1
            pbar.update(1)

    pbar.close()
    print(f"Images saved to {label_outdir}")
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
