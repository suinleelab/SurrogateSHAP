#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "diffusers @ git+https://github.com/huggingface/diffusers.git",
#     "torch>=2.0.0",
#     "accelerate>=0.31.0",
#     "transformers>=4.41.2",
#     "ftfy",
#     "tensorboard",
#     "Jinja2",
#     "peft>=0.11.1",
#     "sentencepiece",
#     "torchvision",
#     "datasets",
#     "bitsandbytes",
#     "prodigyopt",
# ]
# ///

import argparse
import copy
import itertools
import json
import math
import os
import pandas as pd
import shutil
import sys
import time
import warnings
from contextlib import nullcontext

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import convert_unet_state_dict_to_peft, is_wandb_available
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
from huggingface_hub import login
from src import constants
from src.datasets import FashionDatasetWrapper, create_dataset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.37.0.dev0")

logger = get_logger(__name__)
login("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")

def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    return text_encoder_one, text_encoder_two


def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    is_final_validation=False,
):
    accelerator.print(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )
    autocast_ctx = (
        torch.autocast(accelerator.device.type)
        if not is_final_validation
        else nullcontext()
    )

    # pre-calculate  prompt embeds, pooled prompt embeds, text ids because t5 does not support autocast
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            pipeline_args["prompt"], prompt_2=pipeline_args["prompt"]
        )
    images = []
    for _ in range(args.num_validation_images):
        with autocast_ctx:
            image = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    free_memory()

    return images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        default="all",
        choices=["all", "uniform", "shapley", "datamodel", "loo", "aoi", "percentile"],
    )
    parser.add_argument(
        "--removal_seed",
        type=int,
        help="random seed for sampling from the removal distribution",
        default=0,
    )
    parser.add_argument(
        "--removal_unit",
        type=str,
        help="unit of data for removal",
        choices=["brand"],
        default=None,
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="alpha value for the datamodel removal distribution",
        default=None,
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
        choices=[1, 2, 3, 4, 5, 10, 15, 20, 30],
        default=10,
        help="Percentile value to use",
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
            "Mode for percentile removal: 'remove' to remove indices, 'keep' to keep only those indices",
        )
    )
    parser.add_argument(
        "--method",
        type=str,
        default="retrain",
        choices=["retrain", "pruned_ft", "sparse_gd", "gd"],
        help="training or unlearning method",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to repeat the training data.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=False,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'. Not required for fashion dataset.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=4,
        help="LoRA alpha to be used for additional scaling.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="Dropout probability for LoRA layers",
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=None,
        help="Pruning ratio for pruned_ft method",
    )

    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=(
            'We default to the "none" weighting scheme for uniform sampling and uniform loss'
        ),
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_decouple",
        type=bool,
        default=True,
        help="Use AdamW style decoupled weight decay",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )
    parser.add_argument(
        "--adam_weight_decay_text_encoder",
        type=float,
        default=1e-03,
        help="Weight decay to use for text_encoder",
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        type=str,
        default=None,
        help=(
            "Whether to reuse or compute cached latents. Choose between ['reuse', 'compute']. "
            "'reuse' will load precomputed latents if available, 'compute' will always compute the latents afresh."
        ),
    )
    parser.add_argument(
        "--cache_text_embeddings",
        type=str,
        default=None,
        help=(
            "Whether to reuse or compute cached latents. Choose between ['reuse', 'compute']. "
            "'reuse' will load precomputed latents if available, 'compute' will always compute the latents afresh."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_npu_flash_attention",
        action="store_true",
        help="Enabla Flash Attention for NPU",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Validate instance_prompt requirement based on dataset type
    # if args.dataset_type == "standard" and args.instance_prompt is None:
    #     raise ValueError("You must specify --instance_prompt for standard dataset type.")

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    prompts = [example["instance_prompt"] for example in examples]
    filenames = [example["filename"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    batch = {"pixel_values": pixel_values, "prompts": prompts, "filenames": filenames}
    return batch


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError(
                "text_input_ids must be provided when the tokenizer is not specified"
            )

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    if hasattr(text_encoder, "module"):
        dtype = text_encoder.module.dtype
    else:
        dtype = text_encoder.dtype
    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    if hasattr(text_encoders[0], "module"):
        dtype = text_encoders[0].module.dtype
    else:
        dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids

def _infer_rank_pattern_from_peft_state_dict(peft_sd: dict) -> dict:
    """
    peft_sd keys look like:
      '...<module_path>....lora_A.weight' / '...lora_B.weight'
    rank = lora_A.weight.shape[0]
    Returns rank_pattern mapping module_key -> rank.
    """
    rank_pattern = {}
    for k, v in peft_sd.items():
        if not (isinstance(v, torch.Tensor) and k.endswith("lora_A.weight")):
            continue
        r = int(v.shape[0])
        if r <= 0:
            # If this happens, your pruned file truly has rank-0 tensors.
            # Skip so we don't create invalid config.
            continue
        module_key = k[: -len(".lora_A.weight")]
        rank_pattern[module_key] = r

    if not rank_pattern:
        raise ValueError("Could not infer any positive ranks from the LoRA weights (rank_pattern empty).")
    return rank_pattern


def _load_pruned_lora_into_transformer(
    transformer,
    lora_dir: str,
    target_modules: list[str],
    lora_alpha: int,
    lora_dropout: float,
):
    # 1) Read LoRA weights (diffusers format)
    lora_state_dict = FluxPipeline.lora_state_dict(lora_dir)

    # 2) Transformer-only keys + convert to PEFT format
    transformer_state_dict = {
        k.replace("transformer.", ""): v
        for k, v in lora_state_dict.items()
        if k.startswith("transformer.")
    }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)

    # 3) Infer per-module ranks
    rank_pattern = _infer_rank_pattern_from_peft_state_dict(transformer_state_dict)
    r_max = max(rank_pattern.values())
    if r_max <= 0:
        raise ValueError(f"Inferred r_max={r_max}, refusing to create adapter with non-positive rank.")

    # 4) Create adapter named 'default'
    cfg = LoraConfig(
        r=r_max,                      # must be > 0
        rank_pattern=rank_pattern,     # enables per-module ranks
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(cfg)  # adapter_name defaults to "default"

    # 5) Filter checkpoint keys to those that exist in the model (avoids strict arg)
    model_keys = set(get_peft_model_state_dict(transformer, adapter_name="default").keys())
    filtered = {k: v for k, v in transformer_state_dict.items() if k in model_keys}

    missing = len(model_keys) - len(filtered)
    extra = len(transformer_state_dict) - len(filtered)
    print(
        f"[LoRA load] filtered keys: {len(filtered)} | "
        f"missing in ckpt: {missing} | extra in ckpt: {extra}"
    )

    # 6) Load into adapter (PEFT will only see matching keys)
    set_peft_model_state_dict(transformer, filtered, adapter_name="default")

    return "default"


def main(args):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    removal_dir = "full"
    if args.removal_dist is not None:
        if args.removal_unit is None:
            raise ValueError("--removal_unit is not specified")

        removal_dir_dist = args.removal_dist
        if args.removal_dist == "datamodel":
            removal_dir_dist += f"_alpha={args.datamodel_alpha}"

        # For "all" distribution, use simple naming without unit prefix
        if args.removal_dist == "all":
            removal_dir = f"{removal_dir_dist}_seed={args.removal_seed}"
        else:
            removal_dir = f"{args.removal_unit}_{removal_dir_dist}"
            
        if args.removal_dist == "loo":
            removal_dir += f"/{removal_dir_dist}_idx={args.removal_seed}"
        elif args.removal_dist == "aoi":
            removal_dir += f"/{removal_dir_dist}_idx={args.aoi_idx}"
        elif args.removal_dist == "percentile":
            # Parse filename: brand_{method}_{model_behavior}_percentiles.json
            # Example: brand_avg_clip_similarity_lpips_mean_percentiles.json
            percentile_filename = os.path.basename(args.percentile_file).replace("_percentiles.json", "")
            if percentile_filename.startswith("brand_"):
                percentile_filename = percentile_filename[6:]  # Remove 'brand_' prefix
            # percentile_filename now contains: method_model_behavior (e.g., avg_clip_similarity_lpips_mean)
            removal_dir += f"/{removal_dir_dist}"
            removal_dir += f"/{percentile_filename}"
            removal_dir += f"/{args.percentile_type}{args.percentile_value}"
            if args.percentile_mode == "keep":
                removal_dir += "_keep"
        elif args.removal_dist != "all":
            # Only add seed suffix for non-all distributions
            removal_dir += f"/{removal_dir_dist}_seed={args.removal_seed}"

    # Handle different training methods
    expected_num_lora_params = None
    lora_dir = None
    if args.method == "pruned_ft":
        if args.pruning_ratio is None:
            raise ValueError("--pruning_ratio must be specified for pruned_ft method")
        args.method = f"pruned_ft_ratio={args.pruning_ratio}_lr={args.learning_rate}"
        # Pruned LoRAs are stored under retrain/models/pruned_ratio={ratio}/{removal_dir}/
        lora_dir = os.path.join(
            args.output_dir,
            args.dataset,
            "retrain",
            "models",
            f"pruned_ratio={args.pruning_ratio}",
            removal_dir,
        )
        # Look for info.csv to get expected parameter count
        info_csv_path = os.path.join(lora_dir, "info.csv")
        if os.path.exists(info_csv_path):
            info_df = pd.read_csv(info_csv_path)
            expected_num_lora_params = info_df[info_df["metric"] == "pruned_lora_params"]
            if not expected_num_lora_params.empty:
                expected_num_lora_params = expected_num_lora_params["value"].item()
                accelerator.print(f"Expected pruned LoRA parameters: {expected_num_lora_params:,}")
    elif args.method in ["sparse_gd", "gd"]:
        # For sparse_gd and gd methods, load from a pre-trained LoRA
        # You can add specific config here similar to SD version
        if args.resume_from_checkpoint:
            lora_dir = args.resume_from_checkpoint
        else:
            raise ValueError(f"--resume_from_checkpoint must be specified for {args.method} method")

    if args.output_dir is not None:
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.method)
        args.model_outdir = os.path.join(
            args.output_dir,
            "models",
            removal_dir,
            f"lr={args.learning_rate}_r={args.rank}_epochs={args.num_train_epochs}",
        )
        args.sample_outdir = os.path.join(
            args.output_dir,
            "samples",
            removal_dir,
            f"lr={args.learning_rate}_r={args.rank}_epochs={args.num_train_epochs}",
        )

        # If trained weights already exist, skip the script.
        lora_weight_path = os.path.join(
            args.model_outdir, "pytorch_lora_weights.safetensors"
        )
        if os.path.exists(lora_weight_path):
            accelerator.print(
                f"Found trained LoRA weights at {lora_weight_path}. Process cancelled."
            )
            sys.exit(0)  # Exit without raising an error.

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(args.model_outdir, exist_ok=True)
            os.makedirs(args.sample_outdir, exist_ok=True)

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Only load text encoders if we're not reusing cached embeddings
    if args.cache_text_embeddings != "reuse":
        text_encoder_one, text_encoder_two = load_text_encoders(
            text_encoder_cls_one, text_encoder_cls_two
        )
    else:
        text_encoder_one, text_encoder_two = None, None

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    if text_encoder_one is not None:
        text_encoder_one.requires_grad_(False)
    if text_encoder_two is not None:
        text_encoder_two.requires_grad_(False)

    if args.enable_npu_flash_attention:
        if is_torch_npu_available():
            print("npu flash attention enabled.")
            transformer.set_attention_backend("_native_npu")
        else:
            raise ValueError(
                "npu flash attention requires torch_npu extensions and is supported only on npu device "
            )

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    transformer.to(accelerator.device, dtype=weight_dtype)
    # Only move VAE to GPU if we're not reusing cached latents
    if args.cache_latents != "reuse":
        vae.to(accelerator.device, dtype=weight_dtype)
    # Only move text encoders to GPU if we're not reusing cached text embeddings
    if args.cache_text_embeddings != "reuse":
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # Check if we need to load a pre-trained LoRA (for pruned_ft, sparse_gd, gd methods)
    pruned_lora_path = None
    
    # For pruned_ft method, lora_dir was already set earlier - verify it exists and has weights
    if lora_dir is not None:
        if os.path.exists(os.path.join(lora_dir, "pytorch_lora_weights.safetensors")):
            pruned_lora_path = lora_dir
            accelerator.print(f"Loading LoRA from {pruned_lora_path} for {args.method} method.")
        else:
            # Try to find the actual LoRA directory by looking for subdirectories
            if os.path.exists(lora_dir):
                lora_subdirs = [d for d in os.listdir(lora_dir) if os.path.isdir(os.path.join(lora_dir, d)) and d.startswith("lr=")]
                if lora_subdirs:
                    potential_lora_dir = os.path.join(lora_dir, lora_subdirs[0])
                    if os.path.exists(os.path.join(potential_lora_dir, "pytorch_lora_weights.safetensors")):
                        pruned_lora_path = potential_lora_dir
                        accelerator.print(f"Found pruned LoRA at: {pruned_lora_path}")
            
            if pruned_lora_path is None:
                raise ValueError(f"Could not find pytorch_lora_weights.safetensors in {lora_dir} or its subdirectories")
    # Check resume_from_checkpoint for standard checkpoint loading
    elif args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            checkpoint_path = args.resume_from_checkpoint
        else:
            checkpoint_path = args.model_outdir
        
        # Check if this is a pruned LoRA (has pytorch_lora_weights.safetensors directly)
        potential_pruned_path = os.path.join(checkpoint_path, "pytorch_lora_weights.safetensors")
        if os.path.exists(potential_pruned_path):
            pruned_lora_path = checkpoint_path
            accelerator.print(f"Detected pruned LoRA at {pruned_lora_path}. Will load with dynamic rank configuration.")

    # If not loading pre-trained LoRA, create standard LoRA config
    if pruned_lora_path is None:
        # now we will add new LoRA weights the transformer layers
        transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(transformer_lora_config)
        loaded_adapter_name = "default"  # Standard adapter name for new LoRA
        if args.train_text_encoder:
            text_lora_config = LoraConfig(
                r=args.rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                init_lora_weights="gaussian",
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            )
            text_encoder_one.add_adapter(text_lora_config)
    else:
        # Load pruned LoRA - will be loaded later with pipeline.load_lora_weights
        # This allows PEFT to auto-detect the rank from the saved weights
        loaded_adapter_name = _load_pruned_lora_into_transformer(
            transformer=transformer,
            lora_dir=pruned_lora_path,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
        accelerator.print(
            f"Loaded pruned LoRA into transformer as adapter '{loaded_adapter_name}' from {pruned_lora_path}"
        )
    # Estimate number of tunable LoRA parameters
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    transformer_trainable_params = count_trainable_parameters(transformer)
    accelerator.print(f"Trainable LoRA parameters in transformer: {transformer_trainable_params:,}")
    
    if args.train_text_encoder:
        text_encoder_trainable_params = count_trainable_parameters(text_encoder_one)
        accelerator.print(f"Trainable LoRA parameters in text_encoder: {text_encoder_trainable_params:,}")
        accelerator.print(f"Total trainable LoRA parameters: {transformer_trainable_params + text_encoder_trainable_params:,}")
        total_lora_params = transformer_trainable_params + text_encoder_trainable_params
    else:
        accelerator.print(f"Total trainable LoRA parameters: {transformer_trainable_params:,}")
        total_lora_params = transformer_trainable_params
    
    # Validate expected parameters for pruned_ft method
    if expected_num_lora_params is not None:
        if abs(total_lora_params - expected_num_lora_params) > 1:
            raise ValueError(
                f"Loaded LoRA parameters ({total_lora_params:,}) do not match expected "
                f"pruned parameters ({expected_num_lora_params:,})"
            )
        accelerator.print(f"✓ Loaded LoRA parameter count matches expected: {expected_num_lora_params:,}")
    
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None
            modules_to_save = {}
            unwrapped_modules_to_save = {}
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model, adapter_name="default")
                    modules_to_save["transformer"] = model
                    unwrapped_modules_to_save["transformer"] = unwrap_model(model)
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(
                        model, adapter_name="default"
                    )
                    modules_to_save["text_encoder"] = model
                    unwrapped_modules_to_save["text_encoder"] = unwrap_model(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            FluxPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
                **_collate_lora_metadata(unwrapped_modules_to_save),
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f"{k.replace('transformer.', '')}": v
            for k, v in lora_state_dict.items()
            if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        
        # Use strict=False to handle pruned LoRA with different ranks
        incompatible_keys = set_peft_model_state_dict(
            transformer_, transformer_state_dict, adapter_name="default"
        )
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
            if missing_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict missing keys: "
                    f" {missing_keys}. This is expected for pruned LoRA."
                )
        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(
                lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_
            )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    if args.train_text_encoder:
        text_lora_parameters_one = list(
            filter(lambda p: p.requires_grad, text_encoder_one.parameters())
        )

    # Optimization parameters
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.learning_rate,
    }
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [
            transformer_parameters_with_lr,
            text_parameters_one_with_lr,
        ]
    else:
        params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError(
                "To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`"
            )

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # Dataset and DataLoaders creation:
    # Handle percentile-based removal: load indices from JSON file
    removal_idx_to_use = args.removal_seed
    if args.removal_dist == "percentile":
        if args.percentile_file is None:
            raise ValueError("--percentile_file must be specified when using removal_dist='percentile'")
        
        accelerator.print(f"Loading percentile data from {args.percentile_file}")
        with open(args.percentile_file, 'r') as f:
            percentile_data = json.load(f)
        
        # Get indices based on percentile type and value
        key = f"{args.percentile_type}_{args.percentile_value}pct_indices"
        percentile_indices = percentile_data.get(key)
        
        if percentile_indices is None:
            raise ValueError(f"Key '{key}' not found in {args.percentile_file}")
        
        accelerator.print(f"Loaded {len(percentile_indices)} indices from {args.percentile_type} {args.percentile_value}% percentile")
        
        # Use percentile indices as removal_idx
        # If mode is "remove", these are the indices to remove
        # If mode is "keep", we need to invert the selection (handled in create_dataset)
        removal_idx_to_use = percentile_indices
    
    if args.dataset == "fashion":
        # Use custom fashion dataset loader
        # Build kwargs based on removal_dist
        create_dataset_kwargs = {
            "dataset_name": args.dataset,
            "train": True,
            "removal_dist": args.removal_dist,
            "removal_idx": removal_idx_to_use,
            "datamodel_alpha": args.datamodel_alpha,
        }
        
        # Add percentile-specific parameters
        if args.removal_dist == "percentile":
            create_dataset_kwargs["percentile_indices"] = removal_idx_to_use
            create_dataset_kwargs["percentile_mode"] = args.percentile_mode
        
        train_dataset, _ = create_dataset(**create_dataset_kwargs)
        train_dataset = FashionDatasetWrapper(
            hf_dataset=train_dataset,
            size=args.resolution,
            pad_to_square=True,
            custom_instance_prompts=True,
        )
    else:
        # Use standard DreamBoothDataset
        raise NotImplementedError(
            "Only 'fashion' dataset is implemented in this context."
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        (
            instance_prompt_hidden_states,
            instance_pooled_prompt_embeds,
            instance_text_ids,
        ) = compute_text_embeddings(args.instance_prompt, text_encoders, tokenizers)

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            (
                class_prompt_hidden_states,
                class_pooled_prompt_embeds,
                class_text_ids,
            ) = compute_text_embeddings(args.class_prompt, text_encoders, tokenizers)

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat(
                    [prompt_embeds, class_prompt_hidden_states], dim=0
                )
                pooled_prompt_embeds = torch.cat(
                    [pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0
                )
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(
                tokenizer_one, args.instance_prompt, max_sequence_length=77
            )
            tokens_two = tokenize_prompt(
                tokenizer_two,
                args.instance_prompt,
                max_sequence_length=args.max_sequence_length,
            )
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(
                    tokenizer_one, args.class_prompt, max_sequence_length=77
                )
                class_tokens_two = tokenize_prompt(
                    tokenizer_two,
                    args.class_prompt,
                    max_sequence_length=args.max_sequence_length,
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    if args.cache_latents == "compute":
        latents_cache = {}

        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                filenames = batch["filenames"]
                # Encode and sample latents, then apply scaling
                latent_dist = vae.encode(batch["pixel_values"]).latent_dist
                latents = latent_dist.sample()
                latents = (
                    latents - vae_config_shift_factor
                ) * vae_config_scaling_factor

                # Store the entire batch as-is to maintain order
                for idx, filename in enumerate(filenames):
                    latents_cache[filename] = latents[idx].cpu()

        # Save the cache of latents to a file
        vqvae_latent_dir = os.path.join(
            constants.DATASET_DIR,
            "fashion-product",
            "precomputed_emb",
        )
        os.makedirs(vqvae_latent_dir, exist_ok=True)
        torch.save(
            latents_cache,
            os.path.join(vqvae_latent_dir, "vqvae_output.pt"),
        )

        accelerator.print(
            "VQVAE output saved. Set precompute_state=reuse to unload VQVAE model."
        )

        if args.validation_prompt is None:
            del vae
            free_memory()

        raise SystemExit("Latents precomputation completed. Exiting now.")

    elif args.cache_latents == "reuse":
        # Load the cache of latents from a file
        vqvae_latent_dir = os.path.join(
            constants.DATASET_DIR,
            "fashion-product",
            "precomputed_emb",
        )
        latents_cache = torch.load(
            os.path.join(vqvae_latent_dir, "vqvae_output.pt"),
        )
        accelerator.print("VQVAE output loaded. Unloading VQVAE model.")
        del vae
        free_memory()
    else:
        latents_cache = None

    if args.cache_text_embeddings == "compute":
        text_embeddings_cache = {}
        for batch in tqdm(train_dataloader, desc="Caching text embeddings"):
            prompts = batch["prompts"]
            filenames = batch["filenames"]

            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                    prompts, text_encoders, tokenizers
                )
                for idx, prompt in enumerate(prompts):
                    text_embeddings_cache[filenames[idx]] = {
                        "prompt_embeds": prompt_embeds[idx].cpu().half(),
                        "pooled_prompt_embeds": pooled_prompt_embeds[idx].cpu().half(),
                        "text_ids": text_ids.cpu().half(),
                    }

        # Save the cache of text embeddings to a file
        text_embeddings_dir = os.path.join(
            constants.DATASET_DIR,
            "fashion-product",
            "precomputed_text_emb",
        )
        os.makedirs(text_embeddings_dir, exist_ok=True)
        torch.save(
            text_embeddings_cache,
            os.path.join(text_embeddings_dir, "text_embeddings.pt"),
        )

        accelerator.print(
            "Text embeddings saved. Set cache_text_embeddings=reuse to unload text encoders."
        )

        raise SystemExit("Text embeddings precomputation completed. Exiting now.")
    elif args.cache_text_embeddings == "reuse":
        text_embeddings_dir = os.path.join(
            constants.DATASET_DIR,
            "fashion-product",
            "precomputed_text_emb",
        )
        text_embeddings_cache = torch.load(
            os.path.join(text_embeddings_dir, "text_embeddings.pt"),
        )
        accelerator.print("Text embeddings loaded.")
    else:
        text_embeddings_cache = None

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / args.gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            args.num_train_epochs
            * accelerator.num_processes
            * num_update_steps_per_epoch
        )
    else:
        num_training_steps_for_scheduler = (
            args.max_train_steps * accelerator.num_processes
        )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps:
            accelerator.print(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = "fashion-flux-dev-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    accelerator.print("***** Running training *****")
    accelerator.print(f"  Num examples = {len(train_dataset)}")
    accelerator.print(f"  Num batches each epoch = {len(train_dataloader)}")
    accelerator.print(f"  Num Epochs = {args.num_train_epochs}")
    accelerator.print(
        f"  Instantaneous batch size per device = {args.train_batch_size}"
    )
    accelerator.print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    accelerator.print(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
    )
    accelerator.print(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Create time recording file
    if accelerator.is_main_process:
        time_file = os.path.join(args.model_outdir, "time.csv")
        with open(time_file, "w") as f:
            if args.max_train_steps is None:
                f.write("epoch,time,gpu\n")
            else:
                f.write("step,time,gpu\n")

    # Potentially load in the weights and states from a previous save
    # For sparse_gd and gd methods, resume_from_checkpoint is used to load initial weights, not resume training
    if args.resume_from_checkpoint and args.method not in ["sparse_gd", "gd"]:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.model_outdir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.model_outdir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        if args.max_train_steps is None and accelerator.is_main_process:
            epoch_start_time = time.time()

        transformer.train()
        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            if args.max_train_steps is not None and accelerator.is_main_process:
                step_start_time = time.time()
            models_to_accumulate = [transformer]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]
                filenames = batch["filenames"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if not args.train_text_encoder:
                        if args.cache_text_embeddings == "reuse":
                            prompt_embeds = torch.stack(
                                [
                                    text_embeddings_cache[filename]["prompt_embeds"]
                                    for filename in filenames
                                ]
                            ).to(accelerator.device)
                            pooled_prompt_embeds = torch.stack(
                                [
                                    text_embeddings_cache[filename][
                                        "pooled_prompt_embeds"
                                    ]
                                    for filename in filenames
                                ]
                            ).to(accelerator.device)
                            text_ids = text_embeddings_cache[filenames[0]][
                                "text_ids"
                            ].to(accelerator.device)

                        else:
                            (
                                prompt_embeds,
                                pooled_prompt_embeds,
                                text_ids,
                            ) = compute_text_embeddings(
                                prompts, text_encoders, tokenizers
                            )
                    else:
                        tokens_one = tokenize_prompt(
                            tokenizer_one, prompts, max_sequence_length=77
                        )
                        tokens_two = tokenize_prompt(
                            tokenizer_two,
                            prompts,
                            max_sequence_length=args.max_sequence_length,
                        )
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=prompts,
                        )
                else:
                    elems_to_repeat = len(prompts)
                    if args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[
                                tokens_one.repeat(elems_to_repeat, 1),
                                tokens_two.repeat(elems_to_repeat, 1),
                            ],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=args.instance_prompt,
                        )
                    else:
                        (
                            prompt_embeds,
                            pooled_prompt_embeds,
                            text_ids,
                        ) = compute_text_embeddings(prompts, text_encoders, tokenizers)

                # Convert images to latent space
                if args.cache_latents == "reuse":
                    # Cached latents are already sampled and scaled
                    model_input = torch.stack(
                        [latents_cache[filename] for filename in filenames]
                    ).to(accelerator.device, dtype=weight_dtype)
                else:
                    pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                    model_input = vae.encode(pixel_values).latent_dist.sample()
                    model_input = (
                        model_input - vae_config_shift_factor
                    ) * vae_config_scaling_factor
                    model_input = model_input.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                latent_image_ids = FluxPipeline._prepare_latent_image_ids(
                    model_input.shape[0],
                    model_input.shape[2] // 2,
                    model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(
                    device=model_input.device
                )

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(
                    timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
                )
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                packed_noisy_model_input = FluxPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=model_input.shape[0],
                    num_channels_latents=model_input.shape[1],
                    height=model_input.shape[2],
                    width=model_input.shape[3],
                )

                # handle guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.tensor(
                        [args.guidance_scale], device=accelerator.device
                    )
                    guidance = guidance.expand(model_input.shape[0])
                else:
                    guidance = None

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model
                    # (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                model_pred = FluxPipeline._unpack_latents(
                    model_pred,
                    height=model_input.shape[2] * vae_scale_factor,
                    width=model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # flow matching loss
                target = noise - model_input

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (
                            weighting.float()
                            * (model_pred_prior.float() - target_prior.float()) ** 2
                        ).reshape(target_prior.shape[0], -1),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(
                            transformer.parameters(), text_encoder_one.parameters()
                        )
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.max_train_steps is not None and accelerator.is_main_process:
                    step_time = time.time() - step_start_time
                    time_record = f"{global_step},{step_time:.8f},{torch.cuda.get_device_name()}\n"
                    with open(time_file, "a") as f:
                        f.write(time_record)

                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                accelerator.print(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                accelerator.print(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.model_outdir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.model_outdir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        accelerator.print(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break  # Add this to exit epoch loop

        if args.max_train_steps is None and accelerator.is_main_process:
            epoch_time = time.time() - epoch_start_time
            time_record = f"{epoch},{epoch_time:.8f},{torch.cuda.get_device_name()}\n"
            with open(time_file, "a") as f:
                f.write(time_record)

        if accelerator.is_main_process:
            if (
                args.validation_prompt is not None
                and epoch % args.validation_epochs == 0
            ):
                # create pipeline
                if not args.train_text_encoder:
                    text_encoder_one, text_encoder_two = load_text_encoders(
                        text_encoder_cls_one, text_encoder_cls_two
                    )
                    text_encoder_one.to(weight_dtype)
                    text_encoder_two.to(weight_dtype)

                if args.cache_latents == "reuse":
                    # need to load VAE again for validation
                    vae = AutoencoderKL.from_pretrained(
                        args.pretrained_model_name_or_path,
                        subfolder="vae",
                        revision=args.revision,
                        variant=args.variant,
                    ).to(weight_dtype)
                    vae.to(accelerator.device)

                pipeline = FluxPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=unwrap_model(text_encoder_one),
                    text_encoder_2=unwrap_model(text_encoder_two),
                    transformer=unwrap_model(transformer),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline_args = {"prompt": args.validation_prompt}
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                )
                if not args.train_text_encoder:
                    del text_encoder_one, text_encoder_two
                    free_memory()

                images = None

                del pipeline

                if args.cache_latents == "reuse":
                    del vae

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        modules_to_save = {}
        transformer = unwrap_model(transformer)
        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer, adapter_name="default")
        modules_to_save["transformer"] = transformer

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(
                text_encoder_one.to(torch.float32), adapter_name="default"
            )
            modules_to_save["text_encoder"] = text_encoder_one
        else:
            text_encoder_lora_layers = None

        FluxPipeline.save_lora_weights(
            save_directory=args.model_outdir,
            transformer_lora_layers=transformer_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            **_collate_lora_metadata(modules_to_save),
        )

        # Final inference
        # Load previous pipeline
        pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        # load attention processors
        pipeline.load_lora_weights(args.model_outdir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompt": args.validation_prompt}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
                torch_dtype=weight_dtype,
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    """Main entry point of the script."""
    args = parse_args()
    main(args)
