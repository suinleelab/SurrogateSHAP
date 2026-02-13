"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from configs.sd_config import DatasetStats
from src.constants import DATASET_DIR
from src.datasets import remove_index_by_datamodel, remove_index_by_shapley
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

    # Save to JSON
    import json

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
        choices=["artbench_post_impressionism"],
        default="artbench_post_impressionism",
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
        default="artist",
        choices=["artist", "filename"],
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
        default="results/artBench/percentiles",
        help="Directory to save percentile results",
    )
    return parser.parse_args()


def feature_row_for(dataset_name, removal_dist, removal_seed, datamodel_alpha):
    """Build binary per-class presence feature row."""
    removal_unit_file = os.path.join(
        DATASET_DIR,
        "artbench-10-imagefolder-split/train/post_impressionism_artists.csv",
    )
    removal_unit_df = pd.read_csv(removal_unit_file)
    total_num_classes = len(removal_unit_df["artist"].unique())
    # Get remaining indices based on removal distribution
    if removal_dist == "datamodel":
        remaining_idx, _ = remove_index_by_datamodel(
            removal_unit_df, alpha=datamodel_alpha, seed=removal_seed
        )
    elif removal_dist == "shapley":
        remaining_idx, _ = remove_index_by_shapley(removal_unit_df, seed=removal_seed)
    else:
        raise ValueError(f"Unknown removal_dist: {removal_dist}")

    return [1 if i in remaining_idx else 0 for i in range(total_num_classes)]


def collect_data(
    df,
    num_groups,
    model_behavior_key,
    n_samples,
    dataset_name,
    datamodel_alpha,
    collect_remaining_masks=True,
):
    """Build X (feature matrix) and y (behavior vector/matrix) from a dataframe."""

    rdist = df["removal_dist"].tolist()
    ridx = df["removal_seed"].tolist()

    # Extract y (model behavior). Support single or multi-key behaviors.
    if isinstance(model_behavior_key, (list, tuple)):
        y = df[list(model_behavior_key)].to_numpy(dtype=float)
    else:
        y = df[model_behavior_key].to_numpy(dtype=float)

    # Build X rows
    X_rows = []
    C_ref = None
    seen_id = set()
    keep_ids = []
    for idx, (rd, ri) in enumerate(zip(rdist, ridx)):
        if ri not in seen_id:
            x = feature_row_for(dataset_name, rd, ri, datamodel_alpha)
            if C_ref is None:
                C_ref = len(x)
            elif len(x) != C_ref:
                raise ValueError(
                    f"Inconsistent class dimension: expected {C_ref}, got {len(x)}"
                )
            X_rows.append(x)
            keep_ids.append(idx)
            seen_id.add(ri)

    X = np.asarray(X_rows, dtype=float)
    # Ensure y is 2D: (N, M)
    y = [y[i] for i in keep_ids]
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
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
    if args.dataset == "artbench_post_impressionism":
        dataset_stats = DatasetStats.artbench_post_impressionism_stats
        num_groups = dataset_stats["num_groups"]
        test_db_list = [
            (
                f"results/artBench/datamodel/retrain_datamodel_seed{seed}"
                f"_alpha={args.datamodel_alpha}_w=7.5.json"
            )
            for seed in [42, 43, 44]
        ]
        # test_db_list = [args.test_db]
    else:
        raise ValueError

    # Collect test data.
    test_data_list = []
    for test_db in test_db_list:
        test_df = pd.read_json(test_db, lines=True)
        # For test data, check if removal_seed column exists
        test_df = test_df.sort_values(by="removal_seed")
        test_subset_seeds = [i for i in range(args.test_size)]
        test_df = test_df[test_df["removal_seed"].isin(test_subset_seeds)]

        # assert len(test_df) == args.test_size
        x_test, y_test = collect_data(
            df=test_df,
            num_groups=num_groups,
            model_behavior_key=args.model_behavior_key,
            n_samples=args.n_samples,
            dataset_name=args.dataset,
            datamodel_alpha=args.datamodel_alpha,
        )
        test_data_list.append((x_test, y_test))
    num_model_behaviors = y_test.shape[-1]

    # Specify the baseline attributions.
    baseline_list = [
        "avg_pixel_similarity",
        "max_pixel_similarity",
        "avg_clip_similarity",
        "max_clip_similarity",
    ]
    if "aesthetic_score" in args.model_behavior_key:
        baseline_list.extend(
            [
                "avg_grad_sim",
                "max_grad_sim",
                "avg_aesthetic_score",
                "max_aesthetic_score",
                "relative_influence",
                "renorm_influence",
                "trak",
                "journey_trak",
                "dtrak",
                "das",
            ]
        )
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
