"""Run the CLIP similarity baseline for data attribution."""
import argparse
import os

import clip
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers import FluxPipeline
from tqdm import tqdm

from src.constants import DATASET_DIR
from src.datasets import create_dataset
from src.utils import print_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the CLIP similarity baseline for data attribution."
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
        default=2048,
        help="batch size for the training data loader",
    )
    return parser.parse_args()


def main(args):
    """Main function."""
    # Initialize accelerator
    accelerator = Accelerator()
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=accelerator.device)

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

    # Load training dataset for computing CLIP similarity
    train_dataset, _ = create_dataset(
        "fashion",
        train=True,
        removal_dist=args.removal_dist,
        removal_idx=args.removal_seed,
        datamodel_alpha=args.datamodel_alpha,
    )

    # Create custom collate function for CLIP preprocessing
    def collate_fn(examples):
        # Use CLIP preprocessing on PIL images directly
        images = [clip_preprocess(example["image"]) for example in examples]
        pixel_values = torch.stack(images)
        return {"pixel_values": pixel_values}

    clip_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
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

    # Generate images and compute CLIP similarity
    clip_similarity_mat = []
    print(f"Generating {args.num_images} images and computing CLIP similarities...")

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

            # Compute CLIP embedding for reference image
            reference_clip_embedding = clip_model.encode_image(
                clip_preprocess(reference_img).unsqueeze(0).to(accelerator.device)
            ).flatten()
            reference_clip_embedding /= reference_clip_embedding.norm(
                dim=-1, keepdim=True
            )

            # Compute CLIP cosine similarity with all training images
            clip_similarity_array = []
            for batch in clip_dataloader:
                imgs = batch["pixel_values"].to(accelerator.device)
                clip_embeddings = clip_model.encode_image(imgs)
                clip_embeddings /= clip_embeddings.norm(dim=-1, keepdim=True)
                clip_similarity = torch.matmul(
                    clip_embeddings, reference_clip_embedding
                )
                clip_similarity_array.append(clip_similarity.cpu().numpy())
            clip_similarity_mat.append(np.concatenate(clip_similarity_array))

    clip_similarity_mat = np.stack(clip_similarity_mat, axis=1, dtype=np.float32)

    num_groups = len(group_indices_dict.keys())
    group_max_clip_similarity = np.zeros(shape=(num_groups, args.num_images))
    group_avg_clip_similarity = np.zeros(shape=(num_groups, args.num_images))

    for i, group_indices in group_indices_dict.items():
        group_clip_similarity = clip_similarity_mat[group_indices, :]
        group_max_clip_similarity[i, :] = group_clip_similarity.max(axis=0)
        group_avg_clip_similarity[i, :] = group_clip_similarity.mean(axis=0)

    output_dict = {
        "max_clip_similarity": group_max_clip_similarity,
        "avg_clip_similarity": group_avg_clip_similarity,
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
