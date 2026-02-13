"""Prune LoRA weights of a text-to-image diffusion model."""
import argparse
import os
from functools import reduce

import numpy as np
import torch
import torch_pruning as tp
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.utils import convert_state_dict_to_diffusers
from peft.tuners.lora.layer import LoraLayer
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm
from transformers import CLIPTextModel

from src.utils import print_args


def parse_args():
    """Parsing arguments"""
    parser = argparse.ArgumentParser()
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
        "--resolution",
        type=int,
        default=256,
        help="the resolution for input images",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="directory containing LoRA weights to load",
        required=True,
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        help="ratio for remaining parameters",
        default=0.3,
    )
    return parser.parse_args()


def main(args):
    """Main function for pruning."""
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.set_progress_bar_config(disable=True)
    unet = pipeline.unet.to("cuda")

    # Figure out the hidden state shape.
    if isinstance(pipeline.text_encoder, CLIPTextModel):
        position_embedding = (
            pipeline.text_encoder.text_model.embeddings.position_embedding
        )
        hidden_state_shape = position_embedding.weight.shape
    else:
        text_encoder_type = type(pipeline.text_encoder)
        raise NotImplementedError(
            f"hidden state shape retrieval not implemeted for {text_encoder_type}"
        )

    # Figure out the latent sample shape.
    with torch.no_grad():
        latent = pipeline.vae.encode(
            torch.randn(1, 3, args.resolution, args.resolution)
        ).latent_dist.sample()
        latent_shape = latent.shape

    example_inputs = {
        "sample": torch.randn(latent_shape).to("cuda"),
        "timestep": torch.ones((1,)).long().to("cuda"),
        "encoder_hidden_states": torch.randn(1, *hidden_state_shape).to("cuda"),
    }

    unet_macs, unet_params = tp.utils.count_ops_and_params(unet, example_inputs)

    unet.load_attn_procs(args.lora_dir)
    print(f"LoRA weights loaded from {args.lora_dir}")

    unet_lora_macs, unet_lora_params = tp.utils.count_ops_and_params(
        unet, example_inputs
    )
    unet.zero_grad()
    unet.eval()

    # Set up dependency graph and importance metric.
    dg = tp.DependencyGraph().build_dependency(unet, example_inputs=example_inputs)
    imp = tp.importance.MagnitudeImportance()

    # Retrieve the LoRA modules.
    lora_dict = {}

    # Iterate through all named modules to find LoRA layers
    for name, module in unet.named_modules():
        if isinstance(module, LoraLayer):
            lora_dict[name] = module
            print(f"Found LoRA module: {name}")

    print(f"Number of LoRA modules: {len(lora_dict)}")

    # Calculate importance scores.
    print("Calculating importance score for each LoRA node...")
    node_list, node_score_list, node_param_size_list = [], [], []

    for name, lora in tqdm(lora_dict.items()):
        assert isinstance(lora, LoraLayer)

        # We assume a single adapter (e.g. "default")
        adapter_name = next(iter(lora.lora_A.keys()))

        down_layer = lora.lora_A[adapter_name]  # Linear(in_features -> r)
        up_layer = lora.lora_B[adapter_name]  # Linear(r -> out_features)
        rank = lora.r[adapter_name]

        group_idxs = list(range(rank))

        # ---- DOWN (A) ----
        down_group = dg.get_pruning_group(
            down_layer, tp.prune_linear_out_channels, idxs=group_idxs
        )
        down_scores = imp(down_group).detach().cpu().tolist()
        down_nodes = [(name, i, "down") for i in range(rank)]
        node_list.extend(down_nodes)
        node_score_list.extend(down_scores)
        node_param_size_list.extend([down_layer.in_features] * rank)

        # ---- UP (B) ----
        up_group = dg.get_pruning_group(
            up_layer, tp.prune_linear_in_channels, idxs=group_idxs
        )
        up_scores = imp(up_group).detach().cpu().tolist()
        up_nodes = [(name, i, "up") for i in range(rank)]
        node_list.extend(up_nodes)
        node_score_list.extend(up_scores)
        node_param_size_list.extend([up_layer.out_features] * rank)

    # Identify the pairs of LoRA downsampling and upsampling nodes to remove. Pairs of
    # nodes, instead of individual nodes, are removed to ensure that the dependency
    # graph is still structurally sound.
    assert sum(node_param_size_list) == unet_lora_params - unet_params
    lora_param_size = sum(node_param_size_list)
    target_param_size = args.pruning_ratio * lora_param_size
    removed_pair_set = set()

    sorted_indices = np.argsort(node_score_list, kind="stable")
    for i in sorted_indices:
        node, node_param_size = node_list[i], node_param_size_list[i]
        pair = (node[0], node[1])  # (module name, group idx).
        pair_param_size = node_param_size * 2

        if pair not in removed_pair_set:
            removed_pair_set.add(pair)
            lora_param_size -= pair_param_size

        if lora_param_size <= target_param_size:
            break

    removed_module_dict = {}
    for pair in removed_pair_set:
        name, group_idx = pair
        removed_module_dict.setdefault(name, []).append(group_idx)

    # Prune the LoRA nodes.
    print("Pruning the LoRA weights...")
    for name, removed_idx_list in removed_module_dict.items():
        # Recover the LoraLayer from the UNet by dotted name
        lora = reduce(getattr, name.split("."), unet)
        assert isinstance(lora, LoraLayer)

        adapter_name = next(iter(lora.lora_A.keys()))
        down_layer = lora.lora_A[adapter_name]
        up_layer = lora.lora_B[adapter_name]
        rank = lora.r[adapter_name]

        if len(removed_idx_list) > 0:
            # Actually prune the LoRA A/B layers
            tp.prune_linear_out_channels(down_layer, idxs=removed_idx_list)
            tp.prune_linear_in_channels(up_layer, idxs=removed_idx_list)

            # Update rank tracking inside PEFT
            new_rank = rank - len(removed_idx_list)
            lora.r[adapter_name] = new_rank

            # Sanity checks: A: in_features -> r, B: r -> out_features
            assert down_layer.out_features == new_rank
            assert up_layer.in_features == new_rank

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(unet, example_inputs)
    lora_params = unet_lora_params - unet_params
    pruned_lora_params = pruned_params - unet_params
    actual_pruning_ratio = pruned_lora_params / lora_params

    # Save the pruned LoRA weights and pruning info.
    parsed_dir_list = args.lora_dir.split("/")
    if parsed_dir_list[-1] == "":
        parsed_dir_list = parsed_dir_list[:-1]

    # Replace the "method" part of the directory.
    parsed_dir_list[-3] = f"pruned_ratio={args.pruning_ratio}"
    outdir = "/".join(parsed_dir_list)

    unet_lora_state_dict = convert_state_dict_to_diffusers(
        get_peft_model_state_dict(unet, adapter_name="default_0")
    )
    StableDiffusionPipeline.save_lora_weights(
        save_directory=outdir,
        unet_lora_layers=unet_lora_state_dict,
        safe_serialization=True,
    )
    # unet.save_attn_procs(outdir)
    print(f"Pruned LoRA weights saved to {outdir}")

    info_file = os.path.join(outdir, "info.csv")
    with open(info_file, "w") as handle:
        handle.write("metric,value" + "\n")
        handle.write(f"unet_macs,{unet_macs:.0f}" + "\n")
        handle.write(f"unet_lora_macs,{unet_lora_macs:.0f}" + "\n")
        handle.write(f"pruned_unet_lora_macs,{pruned_macs:.0f}" + "\n")
        handle.write(f"unet_params,{unet_params:.0f}" + "\n")
        handle.write(f"lora_params,{lora_params:.0f}" + "\n")
        handle.write(f"pruned_lora_params,{pruned_lora_params:.0f}" + "\n")
        handle.write(f"target_pruning_ratio,{args.pruning_ratio}" + "\n")
        handle.write(f"actual_pruning_ratio,{actual_pruning_ratio:.5f}" + "\n")
    print(f"Pruning information saved to {info_file}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Pruning done!")
