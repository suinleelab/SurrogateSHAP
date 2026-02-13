"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse
import json
import os
from ast import literal_eval
from multiprocessing import Pool
from typing import Any, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.datasets import create_dataset
from src.utils import print_args


def save_top_bottom_percentiles(
    coeff, dataset_name, method_name, model_behavior_key, output_dir
):
    """Save top and bottom percentile players based on attribution scores."""
    n_classes = len(coeff)
    percentiles = [1, 2, 3, 4, 5, 10, 15, 20, 30]

    # Sort by coefficient (descending)
    sorted_indices = np.argsort(-coeff)  # Negative for descending order

    results = {
        "method": method_name,
        "dataset": dataset_name,
        "model_behavior": model_behavior_key,
        "n_classes": n_classes,
    }

    for pct in percentiles:
        n_top = max(1, int(n_classes * pct / 100))

        top_indices = sorted_indices[:n_top].tolist()
        bottom_indices = sorted_indices[-n_top:].tolist()

        results[f"top_{pct}pct_indices"] = top_indices
        results[f"top_{pct}pct_scores"] = coeff[top_indices].tolist()
        results[f"bottom_{pct}pct_indices"] = bottom_indices
        results[f"bottom_{pct}pct_scores"] = coeff[bottom_indices].tolist()

    # Save full ranking
    results["full_ranking"] = sorted_indices.tolist()
    results["full_scores"] = coeff[sorted_indices].tolist()

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{method_name}_{model_behavior_key}_percentiles.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved top/bottom percentiles to {output_file}")

    # Print summary
    print(f"\nTop/Bottom Players Summary for {method_name} ({model_behavior_key}):")
    print("=" * 60)
    for pct in percentiles:
        n_top = max(1, int(n_classes * pct / 100))
        top_avg = np.mean(coeff[sorted_indices[:n_top]])
        bottom_avg = np.mean(coeff[sorted_indices[-n_top:]])
        print(
            f"{pct:3d}%: Top {n_top:3d} avg={top_avg:+.4f}, "
            f"Bottom {n_top:3d} avg={bottom_avg:+.4f}"
        )

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="evaluate data attribution methods using the linear model score"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
        choices=["fashion"],
        default="fashion",
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="datamodel alpha for the test set",
        default=0.5,
    )
    parser.add_argument(
        "--group",
        type=str,
        default="brand",
        choices=["brand"],
        help="unit for how to group images",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        help="directory containing baseline attribution values",
        required=True,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="number of subsets used for evaluating data attributions",
        default=100,
    )
    parser.add_argument(
        "--model_behavior_key",
        type=str,
        help="key to query model behavior in the test database",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="number of generated images to consider for local model behaviors",
        default=None,
    )
    parser.add_argument(
        "--output_percentiles",
        action="store_true",
        help="Save top/bottom percentile players",
    )
    parser.add_argument(
        "--percentiles_dir",
        type=str,
        default="results/fashion100/percentiles",
        help="Directory to save percentile results",
    )
    return parser.parse_args()


def parse_listish(x: Any):
    """Parse list-like strings from CSV; pass through non-strings."""
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                return literal_eval(s)
            except Exception:
                return x
    return x


def get_targets(ds) -> List[int]:
    """Return targets from a dataset, supporting common attribute names."""
    if hasattr(ds, "targets"):
        return list(ds.targets)
    if hasattr(ds, "labels"):
        return list(ds.labels)
    if hasattr(ds, "brand"):
        return list(ds.brand)
    raise AttributeError("Dataset has neither `.targets` nor `.labels`. nor `.brand`")


def num_classes_from(clsnum: Any) -> int:
    """Normalize `clsnum` to an integer count."""
    if isinstance(clsnum, int):
        return clsnum
    try:
        return len(clsnum)
    except Exception:
        raise TypeError(f"Unexpected clsnum type: {type(clsnum)}")


def feature_row_for(dataset, removal_dist, removal_idx, datamodel_alpha) -> List[int]:
    """Build a binary per-class presence feature row."""
    train_dataset, clsnum = create_dataset(
        dataset_name=dataset,
        train=True,
        removal_dist=removal_dist,
        removal_idx=removal_idx,
        datamodel_alpha=datamodel_alpha,
    )
    C = num_classes_from(clsnum)
    remained_targets = train_dataset.remaining_idx
    return [1 if c in remained_targets else 0 for c in range(C)]


def build_matrix(
    df: pd.DataFrame, model_behavior, dataset, datamodel_alpha
) -> (np.ndarray, np.ndarray):
    """Construct X (binary per-class) and y (FID)."""
    rdist = df["removal_dist"].map(parse_listish)
    # Handle both removal_idx (old format) and removal_seed (new format)
    if "removal_seed" in df.columns:
        ridx = df["removal_seed"].map(parse_listish)
    else:
        ridx = df["removal_idx"].map(parse_listish)
    y = df[model_behavior].to_numpy(dtype=float)

    args_list = [(dataset, rd, ri, datamodel_alpha) for rd, ri in zip(rdist, ridx)]

    # Parallel computation
    with Pool(processes=8) as pool:
        X_rows = pool.starmap(feature_row_for, args_list)
    X = np.asarray(X_rows, dtype=float)
    return X, y


def evaluate_lds(attrs_all, test_data_list, num_model_behaviors):
    """Evaluate LDS mean and CI across a list of test data."""
    lds_list = []
    for (x_test, y_test) in test_data_list:
        model_behavior_lds_list = []
        for k in range(num_model_behaviors):
            model_behavior_lds_list.append(
                spearmanr(x_test @ attrs_all[:, k], y_test[:, k]).statistic * 100
            )
        lds_list.append(np.mean(model_behavior_lds_list))
    lds_mean = np.mean(lds_list)
    lds_ci = np.std(lds_list) / np.sqrt(len(lds_list)) * 1.96
    return lds_mean, lds_ci


def main(args):
    """Main function."""
    # Collect test data.
    test_paths = [
        f"results/fashion100/datamodel/datamodel_training_0_5_seed{seed}.json"
        for seed in range(42, 45)
    ]
    test_dfs = [pd.read_json(p, lines=True) for p in test_paths]
    test_data_list = []
    for test_df in test_dfs:
        X_test, y_test = build_matrix(
            test_df, args.model_behavior_key, args.dataset, args.datamodel_alpha
        )
        y_test = y_test.reshape(-1, 1)
        test_data_list.append((X_test, y_test))
    num_model_behaviors = y_test.shape[-1]

    _, num_groups = create_dataset(
        dataset_name=args.dataset,
        train=True,
    )
    # Specify the baseline attributions.
    baseline_list = [
        "avg_pixel_similarity",
        "max_pixel_similarity",
        "avg_clip_similarity",
        "max_clip_similarity",
        "avg_grad_sim",
        "max_grad_sim",
        "relative_influence",
        "renorm_influence",
        "trak",
        "journey_trak",
        "dtrak",
        "das",
    ]
    baseline_list = [f"{args.group}_{baseline}" for baseline in baseline_list]

    for baseline in baseline_list:
        baseline_file = os.path.join(args.baseline_dir, f"{baseline}.npy")
        with open(baseline_file, "rb") as handle:
            attrs_all = np.load(handle)
            assert attrs_all.shape[0] == num_groups
            assert attrs_all.shape[-1] >= num_model_behaviors
            if num_model_behaviors == 1:
                attrs_all = np.mean(attrs_all, axis=-1, keepdims=True)

        lds_mean, lds_ci = evaluate_lds(
            attrs_all=attrs_all,
            test_data_list=test_data_list,
            num_model_behaviors=num_model_behaviors,
        )
        print(f"{baseline}")
        print(f"\tLDS: {lds_mean:.2f} ({lds_ci:.2f})")
        # Save top/bottom percentiles if requested
        if args.output_percentiles:
            percentiles_dir = os.path.join(args.percentiles_dir)
            # Extract attributions for all classes (first model behavior)
            # attrs_all has shape (n_classes, num_model_behaviors)
            coeff = attrs_all[:, 0] if num_model_behaviors > 0 else attrs_all.flatten()
            save_top_bottom_percentiles(
                coeff, args.dataset, baseline, args.model_behavior_key, percentiles_dir
            )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
