"""Class for TRAK score calculation."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline
from lightning.pytorch import seed_everything
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from trak.projectors import CudaProjector, ProjectionType
from trak.utils import is_not_buffer

import src.constants as constants
from configs.ddim_config import DDPMConfig
from src.datasets import ImageDataset, TensorDataset, create_dataset
from src.models.diffusion import GaussianDiffusion
from src.models.diffusion_utils import _load_model
from src.models.utils import get_named_beta_schedule


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Calculating gradient for D-TRAK and TRAK."
    )
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="directory path for loading pre-trained model",
        default=None,
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar20", help="name for dataset"
    )
    parser.add_argument(
        "--device", type=str, help="device of training", default="cuda:0"
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
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
        "--model_behavior",
        type=str,
        choices=[
            "loss",  # TRAK
            "mean",
            "mean-squared-l2-norm",  # D-TRAK
            "l1-norm",
            "l2-norm",
            "linf-norm",
            "ssim",
            "fid",
            "nrmse",
            "is",
        ],
        default=None,
        required=True,
        help="Specification for D-TRAK model behavior.",
    )
    parser.add_argument(
        "--model_behavior_value",
        type=float,
        default=None,
        help="Model output for a pre-calculated model behavior e.g. FID, SSIM, IS.",
    )
    parser.add_argument(
        "--t_strategy",
        type=str,
        choices=["uniform", "cumulative"],
        help="strategy for sampling time steps",
    )
    parser.add_argument(
        "--k_partition",
        type=int,
        default=None,
        help="Partition for embeddings across time steps.",
    )
    parser.add_argument(
        "--projector_dim",
        type=int,
        default=1024,
        help="Dimension for TRAK projector",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default=None,
        help="filepath of sample (generated) images ",
    )
    parser.add_argument(
        "--calculate_gen_grad",
        help="whether to generate validation set and calculate phi",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="number of generated images to consider for local model behaviors",
        default=None,
    )
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--w", type=float, default=0.0)
    parser.add_argument("--v", type=float, default=0.0)
    parser.add_argument("--discrete_label", action="store_true", default=True)
    parser.add_argument("--dtype", default=torch.float32)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per device for computing TRAK scores",
    )

    return parser.parse_args()


def count_parameters(model):
    """Helper function that return the sum of parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vectorize_and_ignore_buffers(g, params_dict=None):
    """
    Flattens and concatenates gradients from multiple weight matrices into a single tensor.

    Args:
    -------
        g (tuple of torch.Tensor):
            Gradients for each weight matrix, each with shape [batch_size, ...].
        params_dict (dict, optional):
            Dictionary to identify non-buffer gradients in 'g'.

    Returns
    -------
    torch.Tensor:
        Tensor with shape [batch_size, num_params], where each row represents
        flattened and concatenated gradients for a single batch instance.
        'num_params' is the total count of flattened parameters across all weight matrices.

    Note:
    - If 'params_dict' is provided, only non-buffer gradients are processed.
    - The output tensor is formed by flattening each gradient tensor and concatenating them.
    """
    batch_size = len(g[0])
    out = []
    if params_dict is not None:
        for b in range(batch_size):
            out.append(
                torch.cat(
                    [
                        x[b].flatten()
                        for i, x in enumerate(g)
                        if is_not_buffer(i, params_dict)
                    ]
                )
            )
    else:
        for b in range(batch_size):
            out.append(torch.cat([x[b].flatten() for x in g]))
    return torch.stack(out)


def main(args):
    """Main function for computing project@gradient for D-TRAK and TRAK."""

    device = torch.device(args.device)

    if args.dataset == "cifar20":
        config = {**DDPMConfig.cifar20_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 64, 64).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    else:
        raise ValueError(
            (f"dataset={args.dataset} is not one of " f"{constants.DATASET}")
        )

    train_dataset, clsnum = create_dataset(
        dataset_name=args.dataset,
        train=True,
        discrete_label=args.discrete_label,
    )

    local_cfg = config["unet_config"]
    model, cemblayer = _load_model(
        args.ckpt_path, device, local_cfg, clsnum, args.discrete_label
    )

    model.eval()
    cemblayer.eval()

    betas = get_named_beta_schedule(num_diffusion_timesteps=args.T)
    diffusion = GaussianDiffusion(
        dtype=args.dtype,
        model=model,
        betas=betas,
        w=args.w,
        v=args.v,
        device=device,
    )

    if args.sample_dir is None:
        if not args.calculate_gen_grad:
            n_samples = len(train_dataset)
            print(
                "Calculating D-TRAK / TRAK on training dataset with %d samples"
                % n_samples
            )
            sample_dataloader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
            )

            save_dir = os.path.join(
                args.outdir,
                args.dataset,
                "d_trak",
                f"train_f={args.model_behavior}_t={args.t_strategy}_k={args.k_partition}_d={args.projector_dim}",
            )
        else:
            save_dir = os.path.join(
                args.outdir,
                args.dataset,
                "d_trak",
                f"gen_f={args.model_behavior}_t={args.t_strategy}_k={args.k_partition}_d={args.projector_dim}",
            )

    else:
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),  # Normalize to [0,1].
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1].
            ]
        )
        print("Calculating D-TRAK / TRAK on sample dataset at %s" % args.sample_dir)
        sample_dataset = ImageDataset(args.sample_dir, preprocess)
        n_samples = len(sample_dataset)
        sample_dataloader = DataLoader(
            sample_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        save_dir = os.path.join(
            args.outdir,
            args.dataset,
            "d_trak",
            f"reference_f={args.model_behavior}_t={args.t_strategy}_k={args.k_partition}_d={args.projector_dim}",
        )

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # Try to load the model class from the `diffusers` package if available.

    if args.dataset == "celeba":
        # The pipeline is of class LDMPipeline.
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        pipeline.unet = model

        pipeline.scheduler.set_timesteps(
            config["scheduler_config"]["num_train_timesteps"]
        )
        vqvae = pipeline.vqvae
        pipeline.vqvae.config.scaling_factor = 1
        vqvae.requires_grad_(False)

        if args.sample_dir is None:
            # Load the precomputed output, avoiding GPU memory usage by the VQ-VAE model
            pipeline.vqvae = None
            vqvae_latent_dir = os.path.join(
                args.outdir,
                args.dataset,
                "precomputed_emb",
            )
            vqvae_latent_dict = torch.load(
                os.path.join(
                    vqvae_latent_dir,
                    "vqvae_output.pt",
                ),
                map_location="cpu",
            )
        else:
            vqvae = vqvae.to(device)

        pipeline.to(device)
    elif args.dataset == "cifar20":
        pipeline = None
        pipeline_scheduler = None
    else:
        pipeline = DDPMPipeline(
            unet=model, scheduler=DDPMScheduler(**config["scheduler_config"])
        ).to(device)

        pipeline_scheduler = pipeline.scheduler

    # Init a memory-mapped array stored on disk directly for D-TRAK results.

    if args.calculate_gen_grad:
        # Generate samples for Journey TRAK
        generated_samples = []
        n_samples = args.n_samples

        if pipeline is not None:
            pipeline.scheduler.num_train_steps = 1000
            pipeline.scheduler.num_inference_steps = 100

        for random_seed in tqdm(range(n_samples)):
            noise_latents = []
            noise_generator = torch.Generator(device=args.device).manual_seed(
                random_seed
            )

            with torch.no_grad():
                noises = torch.randn(
                    example_inputs["sample"].shape,
                    generator=noise_generator,
                    device=args.device,
                )
                input = noises

                # Prepare model_kwargs for CFG
                if args.dataset == "cifar20":
                    model_kwargs = {}
                    random_label = torch.randint(
                        low=0, high=clsnum, size=(1,), device=args.device
                    )

                    model_kwargs["cemb"] = cemblayer(random_label, drop_prob=0.0)

                for t in range(999, -1, -1000 // args.k_partition):
                    noise_latents.append(input.squeeze(0).detach().cpu())

                    t_tensor = torch.tensor([t], device=input.device).long()

                    if args.dataset == "cifar20":
                        input = diffusion.p_sample(input, t_tensor, **model_kwargs)
                #  flip the order so noise_latents[0] gives us the final image
                noise_latents = torch.stack(noise_latents[::-1])

            generated_samples.append(noise_latents)
        generated_samples = torch.stack(generated_samples)
        bogus_labels = torch.zeros(n_samples, dtype=torch.int)
        images_dataset = TensorDataset(
            generated_samples, transform=None, label=bogus_labels
        )

        sample_dataloader = DataLoader(
            images_dataset,
            batch_size=args.batch_size,
            num_workers=1,
            pin_memory=True,
        )

    dstore_keys = np.memmap(
        save_dir,
        dtype=np.float32,
        mode="w+",
        shape=(n_samples, args.projector_dim),
    )

    # Initialize random matrix projector from trak
    projector = CudaProjector(
        grad_dim=count_parameters(model),
        proj_dim=args.projector_dim,
        seed=args.opt_seed,
        proj_type=ProjectionType.normal,  # proj_type=ProjectionType.rademacher,
        device=device,
        max_batch_size=args.batch_size,
    )

    params = {
        k: v.detach() for k, v in model.named_parameters() if v.requires_grad is True
    }
    buffers = {
        k: v.detach() for k, v in model.named_buffers() if v.requires_grad is True
    }
    starttime = time.time()
    if args.model_behavior == "mean-squared-l2-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets, cemb):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
            cemb = cemb.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=(noisy_latents, timesteps),  # Pass both x and t as args
                kwargs={"cemb": cemb},
            )
            ####
            # predictions = predictions.reshape(1, -1)
            # f = torch.norm(predictions.float(), p=2.0, dim=-1)**2 # squared
            # f = f/predictions.size(1) # mean
            # f = f.mean()
            ####
            f = F.mse_loss(
                predictions.float(), torch.zeros_like(targets).float(), reduction="none"
            )
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####

            return f

    elif args.model_behavior == "mean":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets, cemb):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
            cemb = cemb.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=(noisy_latents, timesteps),  # Pass both x and t as args
                kwargs={"cemb": cemb},
            )
            ####
            f = predictions.float()
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####

            return f

    elif args.model_behavior == "l1-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets, cemb):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
            cemb = cemb.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=(noisy_latents, timesteps),  # Pass both x and t as args
                kwargs={"cemb": cemb},
            )
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=1.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    elif args.model_behavior == "l2-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets, cemb):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
            cemb = cemb.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=(noisy_latents, timesteps),  # Pass both x and t as args
                kwargs={"cemb": cemb},
            )
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=2.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    elif args.model_behavior == "linf-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets, cemb):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
            cemb = cemb.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=(noisy_latents, timesteps),  # Pass both x and t as args
                kwargs={"cemb": cemb},
            )
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=float("inf"), dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    else:
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets, cemb):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)
            cemb = cemb.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=(noisy_latents, timesteps),  # Pass both x and t as args
                kwargs={"cemb": cemb},
            )
            ####
            f = F.mse_loss(predictions.float(), targets.float(), reduction="none")
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            return f

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(
        ft_compute_grad,
        in_dims=(None, None, 0, 0, 0, 0),
    )
    for step, batch in enumerate(sample_dataloader):

        seed_everything(args.opt_seed, workers=True)

        image, _ = batch[0], batch[1]
        image = image.to(device)
        bsz = image.shape[0]

        if args.dataset == "celeba" and not args.calculate_gen_grad:
            if args.sample_dir is None:  # Compute TRAK with pre-computed embeddings.
                imageid = batch[2]
                image = torch.stack(
                    [vqvae_latent_dict[imageid[i]] for i in range(len(image))]
                ).to(device)
            else:  # Directly encode the images if there's no precomputation
                image = vqvae.encode(image, False)[0]
            image = image * vqvae.config.scaling_factor

        if args.t_strategy == "uniform":
            selected_timesteps = range(0, 1000, 1000 // args.k_partition)
        elif args.t_strategy == "cumulative":
            selected_timesteps = range(0, args.k_partition)

        for index_t, t in enumerate(selected_timesteps):
            # Sample a random timestep for each image
            timesteps = torch.tensor([t] * bsz, device=image.device)
            timesteps = timesteps.long()
            seed_everything(args.opt_seed * 1000 + t)  # !!!!

            if args.calculate_gen_grad:
                noisy_latents = image[:, index_t, :, :, :]
                noise = torch.randn_like(noisy_latents)
                target = noise  # For epsilon prediction
            else:
                if args.dataset == "cifar20":
                    # Use GaussianDiffusion's q_sample method
                    noisy_latents, noise = diffusion.q_sample(image, timesteps)
                    target = noise  # Epsilon prediction
                else:
                    # Use pipeline scheduler for other datasets (celeba)
                    noise = torch.randn_like(image)
                    noisy_latents = pipeline_scheduler.add_noise(
                        image, noise, timesteps
                    )

                    # Get the target for loss depending on the prediction type
                    if pipeline_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif pipeline_scheduler.config.prediction_type == "v_prediction":
                        target = pipeline_scheduler.get_velocity(
                            image, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {pipeline_scheduler.config.prediction_type}"
                        )

            # Generate labels and embeddings outside vmap
            if args.dataset == "cifar20":
                random_labels = torch.randint(
                    low=0, high=clsnum, size=(bsz,), device=device
                )
                cemb_batch = cemblayer(random_labels, drop_prob=0.0)
            else:
                # Dummy embeddings for other cases
                dummy_cemb = torch.zeros(bsz, local_cfg["cdim"], device=device)
                cemb_batch = dummy_cemb

            ft_per_sample_grads = ft_compute_sample_grad(
                params,
                buffers,
                noisy_latents,
                timesteps,
                target,
                cemb_batch,
            )

            # if len(keys) == 0:
            #     keys = ft_per_sample_grads.keys()

            ft_per_sample_grads = vectorize_and_ignore_buffers(
                list(ft_per_sample_grads.values())
            )

            # print(ft_per_sample_grads.size())
            # print(ft_per_sample_grads.dtype)

            if index_t == 0:
                emb = ft_per_sample_grads
            else:
                emb += ft_per_sample_grads
            # break

        emb = emb / args.k_partition
        print(emb.size())

        # If is_grads_dict == True, then turn emb into a dict.
        # emb_dict = {k: v for k, v in zip(keys, emb)}

        emb = projector.project(emb, model_id=0)
        print(emb.size())
        print(emb.dtype)

        while (
            np.abs(
                dstore_keys[
                    step * args.batch_size : step * args.batch_size + bsz,
                    0:32,
                ]
            ).sum()
            == 0
        ):
            print("saving")
            dstore_keys[step * args.batch_size : step * args.batch_size + bsz] = (
                emb.detach().cpu().numpy()
            )
        print(f"{step} / {len(sample_dataloader)}, {t}")
        print(step * args.batch_size, step * args.batch_size + bsz)

    print("total_time", time.time() - starttime)


if __name__ == "__main__":
    args = parse_args()
    main(args)
