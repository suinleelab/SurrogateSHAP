"""
Prune LoRA weights of a Flux text-to-image diffusion model.

This version addresses the persistent:
  ValueError: `r` should be a positive integer value but the value passed is 0

by doing ALL of the following:
1) Correct pair parameter accounting: pair_size = in_features(A) + out_features(B)
2) Enforce min_rank_per_layer >= 1 during selection (never prune a layer to rank 0)
3) Safety-net during pruning (truncate idx list if it would hit rank < min)
4) Post-prune validation:
    detect any layer that still ended up with rank 0 (prints names)
5) State-dict sanitization: drop any LoRA weight tensors that have a 0 dimension
   (so loaders never infer rank=0 from shapes)

Run:
  python prune_flux_lora.py --lora_dir /path/to/lora --pruning_ratio 0.3
"""

import argparse
import os
from functools import reduce

import numpy as np
import torch
import torch_pruning as tp
from diffusers import FluxPipeline
from peft.tuners.lora.layer import LoraLayer
from peft.utils import get_peft_model_state_dict
from tqdm import tqdm

from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.3,
        help="Ratio of LoRA parameters to REMAIN (e.g. 0.3 keeps 30%).",
    )
    parser.add_argument(
        "--min_rank_per_layer",
        type=int,
        default=1,
        help="Minimum LoRA rank to keep per layer. Must be >= 1.",
    )
    return parser.parse_args()


def _get_module_by_dotted_name(
    root_module: torch.nn.Module, dotted: str
) -> torch.nn.Module:
    return reduce(getattr, dotted.split("."), root_module)


def _sanitize_lora_state_dict(
    state_dict: dict,
) -> tuple[dict, list[tuple[str, tuple[int, ...]]]]:
    """
    Drops any tensor entries that have a 0 dimension.
    This prevents diffusers/peft from inferring rank=0 and throwing:
      ValueError: `r` should be a positive integer value but the value passed is 0
    """
    cleaned = {}
    dropped = []
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            shape = tuple(v.shape)
            if any(d == 0 for d in shape):
                dropped.append((k, shape))
                continue
        cleaned[k] = v
    return cleaned, dropped


def main(args):
    """Main pruning logic."""
    if args.min_rank_per_layer < 1:
        raise ValueError("--min_rank_per_layer must be >= 1 (PEFT requires r > 0).")

    device = torch.device("cuda")
    dtype = torch.bfloat16

    pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=dtype,
    )
    pipeline.set_progress_bar_config(disable=True)
    pipeline.to(device)

    transformer = pipeline.transformer
    transformer.eval()
    transformer.zero_grad(set_to_none=True)

    # Base params (no LoRA)
    transformer_params = sum(p.numel() for p in transformer.parameters())

    # Load LoRA (pipeline-level)
    pipeline.load_lora_weights(args.lora_dir)
    print(f"LoRA weights loaded from {args.lora_dir}")

    # Make sure newly loaded LoRA params are on GPU too
    pipeline.to(device)
    transformer = pipeline.transformer
    transformer.eval()
    transformer.zero_grad(set_to_none=True)

    transformer_lora_params = sum(p.numel() for p in transformer.parameters())

    # Gather LoRA layers
    lora_dict: dict[str, LoraLayer] = {}
    for name, module in transformer.named_modules():
        if isinstance(module, LoraLayer):
            lora_dict[name] = module

    if not lora_dict:
        raise RuntimeError(
            "No LoRA modules found after loading. Check your --lora_dir."
        )

    print(f"Number of LoRA modules: {len(lora_dict)}")

    # Choose adapter name (single-adapter assumption)
    some_lora = next(iter(lora_dict.values()))
    adapter_name = next(iter(some_lora.lora_A.keys()))
    print(f"Using adapter: {adapter_name}")

    # Pre-check ranks
    for name, lora in lora_dict.items():
        r = int(lora.r[adapter_name])
        if r < args.min_rank_per_layer:
            raise ValueError(f"Layer {name} has rank {r} < {args.min_rank_per_layer}.")

    # Build node scores + correct per-pair parameter sizes
    node_list: list[tuple[str, int, str]] = []  # (module_name, rank_idx, "down"/"up")
    node_score_list: list[float] = []
    node_param_size_list: list[int] = []  # for accounting sanity only
    pair_size: dict[
        tuple[str, int], int
    ] = {}  # (module_name, rank_idx) -> in(A)+out(B)

    print("Calculating importance scores...")
    for name, lora in tqdm(lora_dict.items()):
        down_layer = lora.lora_A[adapter_name]  # Linear(in -> r)
        up_layer = lora.lora_B[adapter_name]  # Linear(r -> out)
        rank = int(lora.r[adapter_name])

        # Correct per-rank pair size
        in_feat = int(down_layer.in_features)
        out_feat = int(up_layer.out_features)
        for i in range(rank):
            pair_size[(name, i)] = in_feat + out_feat

        # DOWN scores (rows of A)
        down_scores = [
            torch.norm(down_layer.weight[i, :], p=2).item() for i in range(rank)
        ]
        node_list.extend([(name, i, "down") for i in range(rank)])
        node_score_list.extend(down_scores)
        node_param_size_list.extend([in_feat] * rank)

        # UP scores (cols of B)
        up_scores = [torch.norm(up_layer.weight[:, i], p=2).item() for i in range(rank)]
        node_list.extend([(name, i, "up") for i in range(rank)])
        node_score_list.extend(up_scores)
        node_param_size_list.extend([out_feat] * rank)

    # Sanity: sum of node sizes equals LoRA params count (A + B)
    expected_lora_params = transformer_lora_params - transformer_params
    if sum(node_param_size_list) != expected_lora_params:
        raise AssertionError(
            "LoRA param accounting mismatch:\n"
            f"sum(node_param_size_list)={sum(node_param_size_list)} vs "
            f"(transformer_lora_params - transformer_params)={expected_lora_params}\n"
            "If your LoRA has extra parameters beyond A/B linears, adjust accounting."
        )

    total_lora_params = expected_lora_params
    target_remaining = int(args.pruning_ratio * total_lora_params)

    # Track how many ranks remain per module to enforce min rank
    rank_remaining = {
        name: int(lora.r[adapter_name]) for name, lora in lora_dict.items()
    }

    remaining = total_lora_params
    removed_pair_set: set[tuple[str, int]] = set()

    # Sort by ascending importance
    sorted_indices = np.argsort(node_score_list, kind="stable")

    print("Selecting pairs to remove (with min-rank constraint)...")
    for idx in sorted_indices:
        mod_name, r_idx, _which = node_list[idx]
        pair = (mod_name, r_idx)

        if pair in removed_pair_set:
            continue

        # IMPORTANT: don't drop below min rank
        if rank_remaining[mod_name] <= args.min_rank_per_layer:
            continue

        removed_pair_set.add(pair)
        rank_remaining[mod_name] -= 1
        remaining -= pair_size[pair]

        if remaining <= target_remaining:
            break

    print(
        f"Target remaining LoRA params: {target_remaining:,} / {total_lora_params:,} "
        f"({args.pruning_ratio:.3f})"
    )
    print(f"Estimated remaining after selection: {max(remaining, 0):,}")
    print(f"Pairs to remove: {len(removed_pair_set):,}")

    # Group removals per module
    removed_module_dict: dict[str, list[int]] = {}
    for (name, idx) in removed_pair_set:
        removed_module_dict.setdefault(name, []).append(idx)

    # Apply pruning
    print("Pruning the LoRA weights...")
    for name, removed_idx_list in tqdm(removed_module_dict.items()):
        lora = _get_module_by_dotted_name(transformer, name)
        assert isinstance(lora, LoraLayer)

        down_layer = lora.lora_A[adapter_name]
        up_layer = lora.lora_B[adapter_name]
        rank = int(lora.r[adapter_name])

        removed_idx_list = sorted(set(int(i) for i in removed_idx_list))
        removed_idx_list = [i for i in removed_idx_list if 0 <= i < rank]
        if not removed_idx_list:
            continue

        # SAFETY NET: never prune below min_rank_per_layer
        new_rank = rank - len(removed_idx_list)
        if new_rank < args.min_rank_per_layer:
            max_remove = rank - args.min_rank_per_layer
            if max_remove <= 0:
                continue
            removed_idx_list = removed_idx_list[:max_remove]
            new_rank = rank - len(removed_idx_list)

        # Apply pruning to A and B
        tp.prune_linear_out_channels(down_layer, idxs=removed_idx_list)  # rows of A
        tp.prune_linear_in_channels(up_layer, idxs=removed_idx_list)  # inputs/cols of B

        # Update rank tracking
        lora.r[adapter_name] = int(new_rank)

        # Sanity: these must be > 0
        assert down_layer.out_features == new_rank and new_rank >= 1
        assert up_layer.in_features == new_rank and new_rank >= 1

    # Post-prune validation: detect any rank-0 layer (should not happen, but we check)
    zero_rank_layers = []
    for name, lora in lora_dict.items():
        down_layer = lora.lora_A[adapter_name]
        up_layer = lora.lora_B[adapter_name]
        rA = int(getattr(down_layer, "out_features", -1))
        rB = int(getattr(up_layer, "in_features", -1))
        if rA <= 0 or rB <= 0:
            zero_rank_layers.append((name, rA, rB))

    if zero_rank_layers:
        print("[ERROR] Some layers ended with rank <= 0 (these WILL break loading):")
        for name, rA, rB in zero_rank_layers:
            print(f"  - {name}: down.out_features={rA}, up.in_features={rB}")
        print(
            "Continuing anyway, "
            " but we will SANITIZE the saved state dict by dropping any "
            "zero-dimension tensors so the loader won't infer rank=0."
        )

    # Count parameters after pruning
    pruned_params = sum(p.numel() for p in transformer.parameters())
    lora_params_before = total_lora_params
    lora_params_after = pruned_params - transformer_params
    actual_remaining_ratio = lora_params_after / max(lora_params_before, 1)

    print(f"LoRA params before: {lora_params_before:,}")
    print(f"LoRA params after : {lora_params_after:,}")
    print(f"Actual remaining ratio: {actual_remaining_ratio:.8f}")

    # Output directory
    parsed_dir_list = args.lora_dir.split("/")
    if parsed_dir_list and parsed_dir_list[-1] == "":
        parsed_dir_list = parsed_dir_list[:-1]

    if len(parsed_dir_list) >= 3:
        parsed_dir_list[-3] = f"pruned_ratio={args.pruning_ratio}"
        outdir = "/".join(parsed_dir_list)
    else:
        outdir = args.lora_dir.rstrip("/") + f"_pruned_ratio={args.pruning_ratio}"

    os.makedirs(outdir, exist_ok=True)

    # Save pruned LoRA weights
    transformer_lora_state_dict = get_peft_model_state_dict(
        transformer, adapter_name=adapter_name
    )

    # SANITIZE: remove any zero-dim tensors to prevent rank=0 inference on load
    cleaned_state_dict, dropped = _sanitize_lora_state_dict(transformer_lora_state_dict)
    if dropped:
        print(f"[WARN] Dropping {len(dropped)} zero-dimension tensors from saved LoRA:")
        for k, shape in dropped[:50]:
            print(f"  - {k}: shape={shape}")
        if len(dropped) > 50:
            print(f"  ... and {len(dropped) - 50} more.")

    FluxPipeline.save_lora_weights(
        save_directory=outdir,
        transformer_lora_layers=cleaned_state_dict,
        text_encoder_lora_layers=None,
    )
    print(f"Pruned LoRA weights saved to {outdir}")

    # Save info
    info_file = os.path.join(outdir, "info.csv")
    with open(info_file, "w", encoding="utf-8") as handle:
        handle.write("metric,value\n")
        handle.write(f"transformer_params,{transformer_params}\n")
        handle.write(f"lora_params_before,{lora_params_before}\n")
        handle.write(f"lora_params_after,{lora_params_after}\n")
        handle.write(f"target_remaining_ratio,{args.pruning_ratio}\n")
        handle.write(f"min_rank_per_layer,{args.min_rank_per_layer}\n")
        handle.write(f"actual_remaining_ratio,{actual_remaining_ratio:.8f}\n")
        handle.write(f"dropped_zero_dim_tensors,{len(dropped)}\n")

    print(f"Pruning information saved to {info_file}")


if __name__ == "__main__":
    """Main entry point of the script."""
    args = parse_args()
    print_args(args)
    main(args)
    print("Pruning done!")
