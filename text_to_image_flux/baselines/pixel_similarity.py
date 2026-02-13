"""Run the pixel similarity baseline for data attribution."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers import FluxPipeline
from torchvision import transforms
from tqdm import tqdm

from src.constants import DATASET_DIR
from src.datasets import FashionDatasetWrapper, create_dataset
from src.utils import print_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the pixel similarity baseline for data attribution."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
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
        "--reference_lora_dir",
        type=str,
        default=None,
        help="directory for reference LoRA weights",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["fashion"],
        default="fashion",
        help="Dataset to determine which prompts to use for image generation",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="number of images to generate for computing model behaviors",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="the resolution of generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for reproducible image generation",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="brand",
        choices=["brand"],
        help="unit for how to group images",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        default="all",
        help="removal distribution for dataset",
    )
    parser.add_argument(
        "--removal_seed",
        type=int,
        default=0,
        help="seed for dataset removal",
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        default=0.5,
        help="alpha for datamodel removal",
    )
    parser.add_argument(
        "--with_prior_preservation",
        action="store_true",
        help="whether to use prior preservation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output parent directory where the attribution scores will be saved",
        required=True,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="number of subprocesses for the training data loader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for the training data loader",
    )
    return parser.parse_args()


def load_pipeline(args):
    """Load diffusion model pipeline."""
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to("cuda")
    return pipeline


def main(args):
    """Main function."""
    accelerator = Accelerator()
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load FLUX pipeline
    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch_dtype,
    )

    # Load LoRA weights
    if args.reference_lora_dir is not None:
        pipeline.load_lora_weights(
            args.reference_lora_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        print(f"Load LORA from {args.reference_lora_dir}")

    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    reference_generator = torch.Generator(device=accelerator.device)
    reference_generator.manual_seed(args.seed)

    # Load dataset for sampling prompts
    dataset, _ = create_dataset(
        "fashion",
        train=True,
        removal_dist="all",
    )

    # Sample prompts deterministically
    n = len(dataset["prompt"])
    if n == 0:
        raise ValueError("Dataset returned 0 captions")

    probs = torch.full((n,), 1.0 / n)
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.multinomial(
        probs, num_samples=args.num_images, replacement=True, generator=g
    )
    prompt_list = [dataset["prompt"][i] for i in idx.tolist()]

    # Load training dataset for computing similarity
    train_dataset, _ = create_dataset(
        "fashion",
        train=True,
        removal_dist=args.removal_dist,
        removal_idx=args.removal_seed,
        datamodel_alpha=args.datamodel_alpha,
    )

    # Set up transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # Wrap with FashionDatasetWrapper
    wrapped_train_dataset = FashionDatasetWrapper(
        hf_dataset=train_dataset,
        size=args.resolution,
        pad_to_square=True,
        custom_instance_prompts=True,
    )

    def collate_fn(examples):
        pixel_values = torch.stack([example["instance_images"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        brands = [example["filename"] for example in examples]
        return {"pixel_values": pixel_values, "brands": brands}

    train_dataloader = torch.utils.data.DataLoader(
        wrapped_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # Find the image indices for each group (brand)
    group_df = pd.read_csv(
        os.path.join(
            DATASET_DIR,
            "fashion-product",
            "top100_brands.csv",
        )
    )

    # Create mapping from brand to indices
    group_indices_dict = {}
    train_brands = [train_dataset[i]["brand"] for i in range(len(train_dataset))]

    for i in group_df.index:
        group_name = group_df.iloc[i, 0]  # First column is brand name
        group_indices = [
            idx for idx, brand in enumerate(train_brands) if brand == group_name
        ]
        group_indices_dict[i] = np.array(group_indices)

    # Generate images and compute pixel similarity
    pixel_similarity_mat = []
    print(f"Generating {args.num_images} images and computing similarities...")

    with torch.no_grad():
        for i in tqdm(range(args.num_images)):
            # Pre-calculate prompt embeds (T5 doesn't support autocast)
            prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt_list[i], prompt_2=prompt_list[i]
            )

            # Generate reference image with FLUX
            reference_img = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                guidance_scale=3.5,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=50,
                generator=reference_generator,
            ).images[0]

            # Transform and flatten reference image
            flat_reference_img = (
                train_transforms(reference_img).to(accelerator.device).flatten()
            )
            flat_reference_img /= flat_reference_img.norm(dim=-1, keepdim=True)

            # Compute pixel cosine similarity with all training images
            pixel_similarity_array = []
            for batch in train_dataloader:
                imgs = batch["pixel_values"].to(accelerator.device)
                flat_imgs = imgs.flatten(start_dim=1)
                flat_imgs /= flat_imgs.norm(dim=-1, keepdim=True)
                pixel_similarity = torch.matmul(flat_imgs, flat_reference_img)
                pixel_similarity_array.append(pixel_similarity.cpu().numpy())
            pixel_similarity_mat.append(np.concatenate(pixel_similarity_array))

    pixel_similarity_mat = np.stack(pixel_similarity_mat, axis=1, dtype=np.float32)

    num_groups = len(group_indices_dict.keys())
    group_max_pixel_similarity = np.zeros(shape=(num_groups, args.num_images))
    group_avg_pixel_similarity = np.zeros(shape=(num_groups, args.num_images))

    for i, group_indices in group_indices_dict.items():
        group_pixel_similarity = pixel_similarity_mat[group_indices, :]
        group_max_pixel_similarity[i, :] = group_pixel_similarity.max(axis=0)
        group_avg_pixel_similarity[i, :] = group_pixel_similarity.mean(axis=0)

    output_dict = {
        "max_pixel_similarity": group_max_pixel_similarity,
        "avg_pixel_similarity": group_avg_pixel_similarity,
    }

    # Save results.
    output_dir = os.path.join(args.output_dir, "baselines")
    os.makedirs(output_dir, exist_ok=True)
    for name, output in output_dict.items():
        with open(os.path.join(output_dir, f"{args.group}_{name}.npy"), "wb") as handle:
            np.save(handle, output)

    # Rank groups and save ranked group indices.
    for name, output in output_dict.items():
        for i in range(args.num_images):
            file_prefix = f"generated_image_{i}_{args.group}_rank"
            rank = np.argsort(
                -output[:, i],  # Flip sign for descending rankings.
                kind="stable",
            )
            with open(
                os.path.join(output_dir, f"{file_prefix}_{name}.npy"), "wb"
            ) as handle:
                np.save(handle, rank)
        global_rank = np.argsort(-output.mean(axis=-1), kind="stable")
        with open(
            os.path.join(
                output_dir, f"all_generated_images_{args.group}_rank_{name}.npy"
            ),
            "wb",
        ) as handle:
            np.save(handle, global_rank)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
