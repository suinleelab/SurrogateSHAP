"""Scripts to evaluate Linear Datamodel Score (LDS)"""
import argparse
import json
import os
from ast import literal_eval
from multiprocessing import Pool
from typing import Any, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import constants
from src.attributions.methods.attribution_utils import gmv_topk_distance
from src.attributions.methods.shapley import data_shapley, surrogateshap
from src.datasets import create_dataset


def save_top_bottom_percentiles(
    coeff, dataset_name, method_name, model_behavior, output_dir
):
    """Save top and bottom percentile players based on attribution scores."""
    n_classes = len(coeff)
    percentiles = [1, 2, 3, 4, 5, 10, 15, 20, 30]

    # Sort by coefficient (descending)
    sorted_indices = np.argsort(-coeff)  # Negative for descending order

    results = {
        "method": method_name,
        "dataset": dataset_name,
        "model_behavior": model_behavior,
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
        output_dir, f"{method_name}_{model_behavior}_percentiles.json"
    )
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved top/bottom percentiles to {output_file}")

    # Print summary
    print(f"\nTop/Bottom Players Summary for {method_name} ({model_behavior}):")
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


def parse_args():
    """Parser function"""
    parser = argparse.ArgumentParser(description="test for CFG diffusion model")
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--fit_db", type=str, help="path to the fit database", default=None
    )
    parser.add_argument(
        "--dataset", type=str, default="fashion", help="name for dataset"
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        default=0.5,
        help="alpha value for datamodel removal",
    )
    parser.add_argument(
        "--method",
        type=str,
        help="attribution method",
        default="kernel_shap",
    )
    parser.add_argument(
        "--model_behavior",
        type=str,
        help="Model behavior to be attributed",
        default="msssim_mean",
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        help="directory containing training images",
        default=None,
    )
    parser.add_argument(
        "--generated_dir",
        type=str,
        help="directory containing generated images",
        default=None,
    )
    # scalar for computing Shapley value
    parser.add_argument(
        "--v0",
        type=float,
        default=0.5,
        help="Null model behavior",
    )
    parser.add_argument(
        "--v1",
        type=float,
        default=0.5,
        help="Full model behavior",
    )
    # Params for gradient-based approaches, e.g. D-TRAK, TRAK, etc.
    parser.add_argument(
        "--gradient_type",
        type=str,
        choices=[
            "vanilla_gradient",
            "trak",
            "relative_if",
            "renormalized_if",
            "journey_trak",
            "d_trak",
        ],
        default=None,
        help="Specification for gradient-based model behavior.",
    )
    parser.add_argument(
        "--projector_dim",
        type=int,
        default=4096,
        help="Dimension for TRAK projector",
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
        default=100,
        help="Partition for embeddings across time steps.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=300,
        help="number of samples for computing gradient scores",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "max"],
        default="mean",
        help="aggregation method for gradient scores",
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
    args = parser.parse_args()

    return args


def main(args):
    """Main function to compute Shapley values and evaluate."""

    test_seeds = [42, 43, 44]
    test_paths = [
        f"results/fashion100/datamodel/datamodel_training_0_5_seed{seed}.json"
        for seed in test_seeds
    ]
    test_dfs = [pd.read_json(p, lines=True) for p in test_paths]

    # Build test matrices for all three seeds
    test_data = []
    for test_df in test_dfs:
        X_test, y_test = build_matrix(
            test_df, args.model_behavior, args.dataset, args.datamodel_alpha
        )
        test_data.append((X_test, y_test))

    # train_path = os.path.join("results/fashion100/shapley/cfg_shapley_300.json")
    train_df = pd.read_json(args.fit_db, lines=True)
    X_train, y_train = build_matrix(
        train_df, args.model_behavior, args.dataset, args.datamodel_alpha
    )

    # Compute kernel SHAP with all retraing instances,
    # We consider this as oracle Shapley values

    retrain_train_path = os.path.join("results/fashion100/shapley/cfg_shapley.json")
    retrain_df = pd.read_json(retrain_train_path, lines=True)

    X_retrain, y_retrain = build_matrix(
        retrain_df, args.model_behavior, args.dataset, args.datamodel_alpha
    )

    null_db = pd.read_json(
        os.path.join("results/fashion100", "null_model.json"), lines=True
    )
    full_db = pd.read_json(
        os.path.join("results/fashion100", "full_model.json"), lines=True
    )

    v0 = null_db[args.model_behavior].values[0]
    v1 = full_db[args.model_behavior].values[0]
    print(f"Metric: {args.model_behavior}, v0 (null): {v0:.4f}, v1 (full): {v1:.4f}")

    kernel_shap = data_shapley(
        dataset_size=X_retrain.shape[1],
        x_train=X_retrain,
        y_train=y_retrain,
        v1=v1,
        v0=v0,
    )
    kernel_shap = np.asarray(kernel_shap, dtype=float).reshape(-1)

    if args.method in ["kernel_shap", "surrogateshap"]:
        subsets = [100, 200, 300, 400, 500, 600, 700, 800]
    else:
        subsets = [0]

    for s in subsets:
        X_train_sub = X_train[:s, :]
        y_train_sub = y_train[:s]

        # Store results for all three test sets
        all_metrics = []
        # all_corrs = []

        if args.method == "kernel_shap":
            shapley_values = data_shapley(
                dataset_size=X_train.shape[1],
                x_train=X_train_sub,
                y_train=y_train_sub,
                v1=v1,
                v0=v0,
            )
            shapley_values = np.asarray(shapley_values, dtype=float).reshape(-1)
            coeff = shapley_values
            # corr = spearmanr(shapley_values, kernel_shap).statistic
        elif args.method == "loo":
            coeff = np.zeros(X_train.shape[-1])

            for i in range(X_train.shape[-1]):
                index = np.where(X_train[:, i] == 0)[0]
                coeff[index] = v1 - y_train[index]

            corr = spearmanr(coeff.flatten(), kernel_shap).statistic
            print("Correlation with full kernel SHAP:", corr)
        elif args.method == "gmv":
            coeff = gmv_topk_distance(
                args.aggregation,
                args.dataset,
                args.sample_size,
                args.generated_dir,
                args.training_dir,
                distance="lpips",
                topk=50,
            )
        elif args.method == "surrogateshap":
            v0 = v0
            v1 = v1
            main, _, _ = surrogateshap(X_train_sub, y_train_sub, v0, v1)
            coeff = main[0]  # (n,)

        for X_test, y_test in test_data:
            pred_y = (X_test @ coeff).reshape(-1)
            quad = metrics(y_test, pred_y)
            all_metrics.append(quad)
            # all_corrs.append(corr)

        # Compute mean and std across the three test sets
        mean_spearman = np.mean([m["spearman"] for m in all_metrics])
        std_spearman = np.std([m["spearman"] for m in all_metrics])
        mean_pearson = np.mean([m["pearson"] for m in all_metrics])
        std_pearson = np.std([m["pearson"] for m in all_metrics])
        mean_rmse = np.mean([m["rmse"] for m in all_metrics])
        std_rmse = np.std([m["rmse"] for m in all_metrics])
        mean_r2 = np.mean([m["r2"] for m in all_metrics])
        std_r2 = np.std([m["r2"] for m in all_metrics])
        # mean_corr = np.mean(all_corrs)
        # std_corr = np.std(all_corrs)

        print(
            f"Number of subsets : {s} - "
            f"Faithfulness of {args.method}: "
            f"LDS={mean_spearman:.3f}±{std_spearman:.3f}, "
            f"Pearson={mean_pearson:.3f}±{std_pearson:.3f}, "
            f"RMSE={mean_rmse:.3f}±{std_rmse:.3f}, "
            f"R2={mean_r2:.3f}±{std_r2:.3f}, "
            # f"Correlation with KernelSHAP: {mean_corr:.3f}±{std_corr:.3f}"
        )
        # Save top/bottom percentiles if requested
        if args.output_percentiles:
            percentiles_dir = os.path.join(args.percentiles_dir)
            # Extract attributions for all classes (first model behavior)
            # attrs_all has shape (n_classes, num_model_behaviors)
            save_top_bottom_percentiles(
                coeff,
                args.dataset,
                args.method,
                args.model_behavior,
                percentiles_dir,
            )


if __name__ == "__main__":
    args = parse_args()
    main(args)
