"""Main function for group attribution"""
import argparse
import os
import pickle
import random

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, FluxPipeline, UNet2DConditionModel
from lightning.pytorch import seed_everything
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from configs.ddim_config import DDPMConfig
from src import constants
from src.constants import DATASET_DIR
from src.datasets import FashionDatasetWrapper, create_dataset
from src.models.diffusion import GaussianDiffusion
from src.models.embedding import ConditionalEmbedding
from src.models.unet import Unet
from src.models.utils import get_named_beta_schedule
from src.utils import print_args


def parse_args():
    """Parser function"""
    parser = argparse.ArgumentParser(
        description="Compute DAS loss for diffusion models"
    )

    # Basic experiment settings
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for reproducibility",
        default=42,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar20",
        choices=["cifar20", "artbench", "fashion"],
        help="dataset name",
    )

    # Model checkpoint
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="path to model checkpoint (cifar20) or LoRA weights dir (artbench)",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/miniSD-diffusers",
        help="Path to pretrained Stable Diffusion model (artbench only)",
    )
    # Model settings
    parser.add_argument(
        "--discrete_label",
        action="store_true",
        help="use discrete labels instead of continuous embeddings",
    )
    parser.add_argument(
        "--dtype", default=torch.float32, help="data type for computation"
    )

    # Diffusion model parameters
    parser.add_argument(
        "--T", type=int, default=1000, help="total number of diffusion timesteps"
    )
    parser.add_argument(
        "--w",
        type=float,
        default=1.2,
        help="classifier-free guidance strength",
    )
    parser.add_argument(
        "--v",
        type=float,
        default=0.3,
        help="variance of posterior distribution",
    )

    # DAS loss computation parameters
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=100,
        help="number of timesteps to sample for loss computation",
    )
    parser.add_argument(
        "--e_seed",
        type=int,
        default=42,
        help="random seed for noise generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="batch size for loss computation (defaults to config value)",
    )

    # Output settings
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="path to save output pickle file (auto-generated if not provided)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of dataloader workers",
    )

    # Artbench-specific settings
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="image resolution for artbench (default: 256)",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="use center crop for artbench images",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="use random horizontal flip for artbench images",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="guidance scale for FLUX model (fashion dataset)",
    )

    args = parser.parse_args()

    return args


def _load_model(ckpt_path, device, unet_config, clsnum, discrete_label, train_dataset):
    """Load model from checkpoint"""
    # Initialize model
    net = Unet(**unet_config)

    cemblayer = ConditionalEmbedding(
        clsnum, unet_config.get("cdim"), unet_config.get("cdim")
    )

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)

    if "net" in checkpoint:
        net.load_state_dict(checkpoint["net"])
    else:
        net.load_state_dict(checkpoint)

    if "cemblayer" in checkpoint:
        cemblayer.load_state_dict(checkpoint["cemblayer"])

    net = net.to(device)
    cemblayer = cemblayer.to(device)
    net.eval()
    cemblayer.eval()

    return net, cemblayer


def _load_sd_model(lora_dir, pretrained_model_path, device):
    """Load Stable Diffusion model with LoRA for artbench"""
    # Load base models
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler"
    )

    # Load LoRA weights
    unet.load_lora_adapter(
        lora_dir,
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="default_0",
        prefix="unet",
    )
    unet.set_adapters("default_0")

    # Move to device and eval mode
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)

    text_encoder.eval()
    vae.eval()
    unet.eval()

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    return unet, vae, text_encoder, tokenizer, noise_scheduler


def _load_flux_model(lora_dir, pretrained_model_path, device):
    """Load FLUX model with LoRA for fashion dataset"""
    weight_dtype = torch.float16

    # Load FLUX pipeline
    pipeline = FluxPipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=weight_dtype,
    )

    # Load LoRA weights
    if lora_dir is not None:
        pipeline.load_lora_weights(
            lora_dir, weight_name="pytorch_lora_weights.safetensors"
        )
        print(f"Loaded LoRA from {lora_dir}")

    # Move to device and eval mode
    pipeline = pipeline.to(device)
    pipeline.transformer.eval()
    pipeline.transformer.requires_grad_(False)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline


def set_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def create_artbench_dataset(tokenizer, args):
    """Create artbench dataset with preprocessing transforms."""
    from datasets import load_dataset

    # Load artbench dataset - matching datasets.py approach
    split = "train"
    data_dir = os.path.join(DATASET_DIR, "artbench-10-imagefolder-split", split)
    data_files = {split: os.path.join(data_dir, "**")}

    dataset = load_dataset("imagefolder", data_files=data_files)

    # Filter to post_impressionism style
    cls_idx = np.where(np.array(dataset[split]["style"]) == "post_impressionism")[0]
    dataset = dataset[split].select(cls_idx)
    assert dataset.num_rows == 5000

    column_names = dataset.column_names
    print(f"Dataset columns: {column_names}")

    # Get column names - should have 'image' and 'text' columns now
    image_column, caption_column = column_names[0], column_names[1]

    # Define preprocessing transforms
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution)
            if args.center_crop
            else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip()
            if args.random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def tokenize_captions(examples, is_train=True):
        """Tokenize captions for text conditioning."""
        captions = []

        if caption_column is None:
            # Use default caption for post-impressionism
            captions = ["a Post-Impressionist painting"] * len(examples[image_column])
        else:
            for caption in examples[caption_column]:
                if isinstance(caption, str):
                    captions.append(caption)
                elif isinstance(caption, (list, np.ndarray)):
                    captions.append(random.choice(caption) if is_train else caption[0])
                else:
                    raise ValueError(
                        f"{caption_column} should contain strings or lists."
                    )

        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    def preprocess_train(examples):
        """Preprocess training examples."""
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    return dataset.with_transform(preprocess_train)


def create_fashion_dataset(args):
    """Create fashion dataset with preprocessing."""
    train_dataset, _ = create_dataset(
        "fashion",
        train=True,
    )
    fashion_dataset = FashionDatasetWrapper(
        hf_dataset=train_dataset,
        size=args.resolution,
        pad_to_square=True,
        custom_instance_prompts=True,
    )
    return fashion_dataset


def create_artbench_collate_fn():
    """Create collate function for artbench dataloader."""

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    return collate_fn


def create_fashion_collate_fn():
    """Create collate function for fashion dataloader."""

    def collate_fn(examples):
        pixel_values = [example["instance_images"] for example in examples]
        prompts = [example["instance_prompt"] for example in examples]
        filenames = [example["filename"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        return {
            "pixel_values": pixel_values,
            "prompts": prompts,
            "filenames": filenames,
        }

    return collate_fn


def compute_losses_for_model(net, cemblayer, diffusion, dataloader, args):
    """Compute losses for all batches and timesteps"""
    print(f"Computing losses across {args.num_timesteps} timesteps")

    batch_loss_list = []
    timestep_interval = max(1, args.T // args.num_timesteps)

    for step, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Handle data based on discrete_label flag, matching main.py
        if args.discrete_label:
            if isinstance(batch, (list, tuple)):
                images, labels = batch
                emb = None
            else:
                images = batch["image"] if "image" in batch else batch["input"]
                labels = (
                    batch["label"] if "label" in batch else batch.get("class_label")
                )
                emb = None
        else:
            if isinstance(batch, (list, tuple)):
                images, labels, emb = batch
            else:
                images = batch["image"] if "image" in batch else batch["input"]
                labels = (
                    batch["label"] if "label" in batch else batch.get("class_label")
                )
                emb = batch.get("embedding")

        images = images.to(args.device)
        labels = labels.to(args.device) if labels is not None else None
        emb = emb.to(args.device) if emb is not None else None

        bsz = images.shape[0]

        # Prepare model_kwargs for CFG, matching trak_compute.py
        model_kwargs = {}
        if args.discrete_label:
            model_kwargs["cemb"] = cemblayer(labels.long(), drop_prob=0.0)
        else:
            model_kwargs["cemb"] = cemblayer(emb, drop_prob=0.0)

        # Compute losses across selected timesteps
        time_loss_list = []

        for index_t, t in enumerate(range(0, args.T, timestep_interval)):
            if index_t >= args.num_timesteps:
                break

            # Create timesteps tensor
            timesteps = torch.tensor([t] * bsz, device=images.device).long()

            # Set seed for reproducible noise generation
            set_seeds(args.e_seed * 1000 + t)

            # Add noise using the diffusion forward process, matching trak_compute.py
            noisy_images, noise = diffusion.q_sample(images, timesteps)

            # Compute loss
            with torch.no_grad():
                # Predict noise using model_kwargs
                predicted_noise = net(noisy_images, timesteps, **model_kwargs)

                # Compute MSE loss
                loss = F.mse_loss(
                    predicted_noise.float(), noise.float(), reduction="none"
                )

                # Average over spatial dimensions, keep batch dimension
                loss = loss.mean(dim=list(range(1, len(loss.shape))))
                time_loss_list.append(loss.detach().cpu().numpy())

        # Stack timestep losses: shape (num_timesteps, batch_size)
        batch_loss_array = np.stack(time_loss_list, axis=0)
        batch_loss_list.append(batch_loss_array)

    return batch_loss_list


def compute_losses_for_sd_model(
    unet, vae, text_encoder, tokenizer, noise_scheduler, dataloader, args
):
    """Compute losses for Stable Diffusion model (artbench)"""
    print(
        f"Computing losses across {args.num_timesteps} timesteps for Stable Diffusion"
    )

    batch_loss_list = []
    timestep_interval = max(1, args.T // args.num_timesteps)

    for step, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        images = batch["pixel_values"]
        text_input_ids = batch["input_ids"]

        images = images.to(args.device)
        text_input_ids = text_input_ids.to(args.device)
        bsz = images.shape[0]

        with torch.no_grad():
            encoder_hidden_states = text_encoder(text_input_ids)[0]
            latents = vae.encode(images).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Compute losses across selected timesteps
        time_loss_list = []

        for index_t, t in enumerate(range(0, args.T, timestep_interval)):
            if index_t >= args.num_timesteps:
                break

            timesteps = torch.tensor([t] * bsz, device=latents.device).long()

            # Set seed for reproducible noise generation
            set_seeds(args.e_seed * 1000 + t)
            noise = torch.randn_like(latents)

            # Add noise using the scheduler
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get target based on prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                )

            # Compute loss
            with torch.no_grad():
                # Predict noise
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                # Compute MSE loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

                # Average over spatial dimensions, keep batch dimension
                loss = loss.mean(dim=list(range(1, len(loss.shape))))
                time_loss_list.append(loss.detach().cpu().numpy())

        # Stack timestep losses: shape (num_timesteps, batch_size)
        batch_loss_array = np.stack(time_loss_list, axis=0)
        batch_loss_list.append(batch_loss_array)

    return batch_loss_list


def compute_losses_for_flux_model(pipeline, dataloader, args):
    """Compute losses for FLUX model (fashion)"""
    print(f"Computing losses across {args.num_timesteps} timesteps for FLUX model")

    # Load precomputed embeddings
    vqvae_latent_dir = os.path.join(
        DATASET_DIR,
        "fashion-product",
        "precomputed_emb",
    )
    latents_cache = torch.load(
        os.path.join(vqvae_latent_dir, "vqvae_output.pt"),
    )
    print("VQVAE output loaded.")

    text_embeddings_dir = os.path.join(
        DATASET_DIR,
        "fashion-product",
        "precomputed_text_emb",
    )
    text_embeddings_cache = torch.load(
        os.path.join(text_embeddings_dir, "text_embeddings.pt"),
    )
    print("Text embeddings loaded.")

    batch_loss_list = []
    timestep_interval = max(1, args.T // args.num_timesteps)
    noise_scheduler = pipeline.scheduler
    transformer = pipeline.transformer
    weight_dtype = torch.float16

    for step, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        filenames = batch["filenames"]

        # Get precomputed embeddings
        prompt_embeds = torch.stack(
            [text_embeddings_cache[filename]["prompt_embeds"] for filename in filenames]
        ).to(args.device, dtype=weight_dtype)
        pooled_prompt_embeds = torch.stack(
            [
                text_embeddings_cache[filename]["pooled_prompt_embeds"]
                for filename in filenames
            ]
        ).to(args.device, dtype=weight_dtype)

        latents = torch.stack([latents_cache[filename] for filename in filenames]).to(
            args.device, dtype=weight_dtype
        )

        bsz = latents.shape[0]

        # Create text_ids and img_ids
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, dtype=weight_dtype).to(
            args.device
        )
        latent_image_ids = FluxPipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            args.device,
            weight_dtype,
        )

        # Compute losses across selected timesteps
        time_loss_list = []

        for index_t, t in enumerate(range(0, args.T, timestep_interval)):
            if index_t >= args.num_timesteps:
                break

            timesteps = torch.tensor([t] * bsz, device=latents.device).long()

            # Set seed for reproducible noise generation
            set_seeds(args.e_seed * 1000 + t)
            noise = torch.randn_like(latents, dtype=weight_dtype)

            # For FLUX flow matching:
            # noisy_latents = (1 - sigma) * latents + sigma * noise
            # Convert timesteps to sigmas
            sigmas = timesteps.float() / noise_scheduler.config.num_train_timesteps
            sigmas = sigmas.to(weight_dtype).view(
                -1, 1, 1, 1
            )  # Reshape for broadcasting
            noisy_latents = (1 - sigmas) * latents + sigmas * noise

            # Pack latents
            packed_noisy_latents = FluxPipeline._pack_latents(
                noisy_latents,
                batch_size=bsz,
                num_channels_latents=noisy_latents.shape[1],
                height=noisy_latents.shape[2],
                width=noisy_latents.shape[3],
            )

            # For flow matching, target is the velocity: noise - latents
            target = noise - latents

            # Compute loss
            with torch.no_grad():
                # Create guidance tensor
                guidance = torch.full(
                    [bsz], args.guidance_scale, device=args.device, dtype=weight_dtype
                )

                # Predict noise
                model_pred = transformer(
                    hidden_states=packed_noisy_latents,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                ).sample

                # Unpack predictions
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=int(latents.shape[2] * pipeline.vae_scale_factor),
                    width=int(latents.shape[3] * pipeline.vae_scale_factor),
                    vae_scale_factor=pipeline.vae_scale_factor,
                )

                # Compute MSE loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")

                # Average over spatial dimensions, keep batch dimension
                loss = loss.mean(dim=list(range(1, len(loss.shape))))
                time_loss_list.append(loss.detach().cpu().numpy())

        # Stack timestep losses: shape (num_timesteps, batch_size)
        batch_loss_array = np.stack(time_loss_list, axis=0)
        batch_loss_list.append(batch_loss_array)

    return batch_loss_list


def main(args):
    """Main function to compute residual for DAS loss"""
    seed_everything(args.opt_seed, workers=True)  # Seed for model optimization.

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device

    if args.dataset == "cifar20":
        # DDPM-based model
        config = {**DDPMConfig.cifar20_config}

        train_dataset, clsnum = create_dataset(
            dataset_name=args.dataset,
            train=True,
            removal_dist="all",
            discrete_label=True,
        )

        # Determine batch size
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else config.get("batch_size", 256)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"Dataset size: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")

        # Load DDPM model
        local_cfg = config["unet_config"]
        net, cemblayer = _load_model(
            args.ckpt_path,
            device,
            local_cfg,
            clsnum,
            args.discrete_label,
            train_dataset,
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

        print("\nComputing DAS losses for DDPM")
        print(f"Checkpoint: {args.ckpt_path}")
        print(f"Dataset: {args.dataset}")
        print(f"Number of timesteps: {args.num_timesteps}")

        # Compute losses
        batch_loss_list = compute_losses_for_model(
            net, cemblayer, diffusion, train_loader, args
        )

    elif args.dataset == "artbench":
        # Load Stable Diffusion model with LoRA
        unet, vae, text_encoder, tokenizer, noise_scheduler = _load_sd_model(
            args.ckpt_path, args.pretrained_model_name_or_path, device
        )

        # Create artbench dataset with preprocessing
        train_dataset = create_artbench_dataset(tokenizer, args)

        # Determine batch size
        batch_size = args.batch_size if args.batch_size is not None else 32

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=create_artbench_collate_fn(),
        )

        print(f"Dataset size: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        print("\nComputing DAS losses for Stable Diffusion")
        print(f"LoRA dir: {args.ckpt_path}")
        print(f"Base model: {args.pretrained_model_name_or_path}")
        print(f"Dataset: {args.dataset}")
        print(f"Number of timesteps: {args.num_timesteps}")

        # Compute losses
        batch_loss_list = compute_losses_for_sd_model(
            unet, vae, text_encoder, tokenizer, noise_scheduler, train_loader, args
        )

    elif args.dataset == "fashion":
        # Load FLUX model with LoRA
        pipeline = _load_flux_model(
            args.ckpt_path, args.pretrained_model_name_or_path, device
        )

        # Create fashion dataset with preprocessing
        train_dataset = create_fashion_dataset(args)

        # Determine batch size
        batch_size = args.batch_size if args.batch_size is not None else 8

        # Create dataloader
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=create_fashion_collate_fn(),
        )

        print(f"Dataset size: {len(train_dataset)}")
        print(f"Number of batches: {len(train_loader)}")
        print("\nComputing DAS losses for FLUX")
        print(f"LoRA dir: {args.ckpt_path}")
        print(f"Base model: {args.pretrained_model_name_or_path}")
        print(f"Dataset: {args.dataset}")
        print(f"Number of timesteps: {args.num_timesteps}")

        # Compute losses
        batch_loss_list = compute_losses_for_flux_model(pipeline, train_loader, args)

    else:
        raise ValueError(f"Unsupported dataset={args.dataset}")

    # Concatenate all batches: list of arrays with shape (num_timesteps, batch_size)
    # Result: (num_timesteps, total_samples)
    all_losses = np.concatenate(batch_loss_list, axis=1)

    print(f"\nComputed losses shape: {all_losses.shape}")
    print(f"  Timesteps: {all_losses.shape[0]}")
    print(f"  Samples: {all_losses.shape[1]}")

    # Determine output file path
    if args.output_file is None:
        # Create default output path
        output_dir = os.path.join(
            args.outdir,
            f"seed{args.opt_seed}",
            args.dataset,
            "das_losses",
        )
        os.makedirs(output_dir, exist_ok=True)

        args.output_file = os.path.join(output_dir, "das_loss.pkl")

    # Save results
    results = {
        "losses": all_losses,
        "args": vars(args),
        "num_samples": all_losses.shape[1],
        "num_timesteps": all_losses.shape[0],
        "dataset": args.dataset,
    }

    with open(args.output_file, "wb") as f:
        pickle.dump(results, f)

    print(f"\nResults saved to: {args.output_file}")

    return True


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    success = main(args)

    if success:
        print("\nDAS loss computation completed successfully!")
    else:
        print("\nDAS loss computation failed!")
