"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from configs.sd_config import DatasetStats
from src.attributions.methods.attribution_utils import gmv_topk_distance
from src.attributions.methods.shapley import data_shapley, surrogateshap
from src.constants import DATASET_DIR
from src.datasets import (
    remove_data_by_loo,
    remove_index_by_datamodel,
    remove_index_by_shapley,
    remove_index_by_shapley_uniform,
)
from src.utils import print_args


def metrics(y_true, y_pred):
    """Compute regression metrics between true and predicted values."""
    rho = spearmanr(y_true, y_pred).statistic * 100
    r = np.corrcoef(y_true, y_pred)[0, 1] if len(np.unique(y_pred)) > 1 else np.nan

    f_mean = np.mean(y_true)
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - f_mean) ** 2)
    r2 = 1 - ss_res / ss_tot

    rmse = np.mean((y_true - y_pred) ** 2) ** 0.5

    return dict(spearman=rho, pearson=r, rmse=rmse, r2=r2)


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
    elif removal_dist == "shapley_uniform":
        remaining_idx, _ = remove_index_by_shapley_uniform(
            removal_unit_df, seed=removal_seed
        )
    elif removal_dist == "loo":
        remaining_idx, _ = remove_data_by_loo(removal_unit_df, removal_seed)
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
        description="evaluate Shapley values using the linear model score"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
        choices=["artbench"],
        default="artbench",
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="datamodel alpha for the test set",
        default=0.5,
    )
    parser.add_argument(
        "--fit_db",
        type=str,
        help="database with model behaviors for fitting Shapley values",
        default=("results/artBench/shapley/cfg_shapley.json"),
    )
    parser.add_argument(
        "--fit_size_factor",
        type=float,
        help="factor for scaling the baseline fitting size",
        default=1.0,
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        help="aggregation method for GMV",
        choices=["mean", "max"],
        default="mean",
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        help="directory containing generated images",
        default=None,
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        help="directory containing training images",
        default=None,
    )
    parser.add_argument(
        "--null_db",
        type=str,
        help="database with model behaviors for the null model",
        default=("results/artBench/null_model.json"),
    )
    parser.add_argument(
        "--full_db",
        type=str,
        help="database with model behaviors for the fully trained model",
        default=("results/artBench/full_model.json"),
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="number of subsets used for evaluating data attributions",
        default=100,
    )
    parser.add_argument(
        "--fit_size",
        type=int,
        nargs="*",
        help="number of subsets used for fitting baseline data attributions",
        default=[300],
    )
    parser.add_argument(
        "--method",
        type=str,
        help="attribution method",
        choices=["kernelshap", "surrogateshap", "gmv", "loo"],
        default="kernelshap",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        help="maximum depth for surrogateshap",
        default=5,
    )
    parser.add_argument(
        "--model_behavior_key",
        type=str,
        help="key to query model behavior in the databases",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--test_db",
        type=str,
        help="test database (retrain datamodel)",
        default=("results/artBench/retrain_datamodel_epochs_w=7.5.json"),
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="number of generated images to consider for local model behaviors",
        default=150,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="directory to save the Shapley values",
        default=None,
    )
    parser.add_argument(
        "--outfile_prefix",
        type=str,
        help="output file prefix for saving the Shapley values",
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


def main(args):
    """Main function."""
    dataset_stats = DatasetStats.artbench_post_impressionism_stats
    num_groups = dataset_stats["num_groups"]

    seeds = [42, 43, 44]
    test_db_list = [
        os.path.join(
            "results/artBench/datamodel",
            f"retrain_datamodel_seed{seed}_alpha={args.datamodel_alpha}_w=7.5.json",
        )
        for seed in seeds
    ]

    v1 = pd.read_json(args.full_db, lines=True)[args.model_behavior_key].values
    v0 = pd.read_json(args.null_db, lines=True)[args.model_behavior_key].values

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

    fit_df = pd.read_json(args.fit_db, lines=True)
    fit_df = fit_df.sort_values(by="removal_seed")

    # Evaluate Shapley values with varying fitting sizes.
    lds_mean_list, lds_ci_list = [], []
    fit_size_list = []

    retrain_df = pd.read_json(
        "results/artBench/shapley/retrain_shapley_w=7.5.json", lines=True
    )

    X, Y = collect_data(
        df=retrain_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
        dataset_name=args.dataset,
        datamodel_alpha=args.datamodel_alpha,
    )
    # kernel_shap = data_shapley(X.shape[-1], X, Y, v0, v1).flatten()

    kernel_shap, inter, _ = surrogateshap(
        X,
        Y[:, 0],
        v0,
        v1,
    )
    kernel_shap = kernel_shap.flatten()

    for baseline_fit_size in args.fit_size:
        fit_size = math.floor(baseline_fit_size * args.fit_size_factor)
        fit_size_list.append(fit_size)

        X, Y = collect_data(
            df=fit_df,
            num_groups=num_groups,
            model_behavior_key=args.model_behavior_key,
            n_samples=args.n_samples,
            dataset_name=args.dataset,
            datamodel_alpha=args.datamodel_alpha,
        )
        x_fit = X[:fit_size]
        y_fit = Y[:fit_size]

        attrs_all = []

        for k in range(num_model_behaviors):

            if args.method == "kernelshap":
                attrs = data_shapley(x_fit.shape[-1], x_fit, y_fit[:, k], v0, v1)
                attrs_all.append(attrs.flatten())
                corr = spearmanr(attrs.flatten(), kernel_shap).statistic
                print("Correlation with full kernel SHAP:", corr)
            elif args.method == "loo":
                attrs = np.zeros(x_fit.shape[-1])

                for i in range(x_fit.shape[-1]):
                    index = np.where(x_fit[:, i] == 0)[0]
                    attrs[index] = v1 - y_fit[index, k]

                corr = spearmanr(attrs.flatten(), kernel_shap).statistic
                print("Correlation with full kernel SHAP:", corr)
                attrs_all.append(attrs.flatten())
            elif args.method == "gmv":
                coeffs = gmv_topk_distance(
                    args.aggregation,
                    args.dataset,
                    args.n_samples,
                    args.generated_dir,
                    args.training_dir,
                    distance="lpips",
                    topk=50,
                )
                attrs_all.append(coeffs.flatten())
            elif args.method == "surrogateshap":
                main, _, _ = surrogateshap(
                    x_fit,
                    y_fit[:, k],
                    v0,
                    v1,
                    args.max_depth,
                )
                phi = main[0]  # (n,)
                attrs_all.append(phi.flatten())
                corr = spearmanr(phi.flatten(), kernel_shap).statistic
                print("Correlation with full kernel SHAP:", corr)

        attrs_all = np.stack(attrs_all, axis=1)

        lds_mean, lds_ci = evaluate_lds(
            attrs_all=attrs_all,
            test_data_list=test_data_list,
            num_model_behaviors=num_model_behaviors,
        )
        lds_mean_list.append(lds_mean)
        lds_ci_list.append(lds_ci)

        print(f"fit size: {fit_size}")
        print(f"\tLDS: {lds_mean:.3f} ({lds_ci:.3f})")

        if args.output_dir is not None:
            outfile = f"artist_{args.outfile_prefix}_fit_size={fit_size}.npy"
            with open(os.path.join(args.output_dir, outfile), "wb") as handle:
                np.save(handle, attrs_all)

            global_rank = np.argsort(-attrs_all.mean(axis=-1), kind="stable")
            rank_file = "all_generated_images_artist_rank"
            rank_file += f"_{args.outfile_prefix}_fit_size={fit_size}.npy"
            with open(os.path.join(args.output_dir, rank_file), "wb") as handle:
                np.save(handle, global_rank)

        # Save top/bottom percentiles if requested
        if args.output_percentiles:
            percentiles_dir = os.path.join(args.percentiles_dir)
            # Extract attributions for all classes (first model behavior)
            # attrs_all has shape (n_classes, num_model_behaviors)
            coeff = attrs_all[:, 0] if num_model_behaviors > 0 else attrs_all.flatten()
            save_top_bottom_percentiles(
                coeff,
                args.dataset,
                args.method,
                args.model_behavior_key,
                percentiles_dir,
            )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
