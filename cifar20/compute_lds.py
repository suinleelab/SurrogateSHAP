"""Scripts to evaluate Linear Datamodel Score (LDS)"""
import argparse
import os
from ast import literal_eval
from multiprocessing import Pool
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src import constants
from src.attributions.methods.attribution_utils import (
    CLIPScore,
    gmv_topk_distance,
    pixel_distance,
)
from src.attributions.methods.compute_gradient_score import compute_gradient_scores
from src.attributions.methods.shapley import (
    data_shapley,
    predict_quadratic,
    surrogateshap,
)
from src.datasets import create_dataset


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
    raise AttributeError("Dataset has neither `.targets` nor `.labels`.")


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
    remained_targets = set(get_targets(train_dataset))
    return [1 if c in remained_targets else 0 for c in range(C)]


def build_matrix(
    df: pd.DataFrame, model_behavior, dataset, datamodel_alpha
) -> (np.ndarray, np.ndarray):
    """Construct X (binary per-class) and y (FID)."""
    rdist = df["removal_dist"].map(parse_listish)
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


def save_top_bottom_percentiles(
    coeff, dataset_name, method_name, model_behavior, output_dir
):
    """Save top and bottom percentile players based on attribution scores."""
    n_classes = len(coeff)
    percentiles = [5, 10, 20, 30, 40]

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

    # Save to JSON
    import json

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
        "--fit_db",
        type=str,
        help="Path to fit dataset for computign LDS",
        required=None,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--generated_dir", type=str, help="directory of generated images", default=None
    )
    parser.add_argument(
        "--training_dir", type=str, help="directory of training images", default=None
    )
    parser.add_argument(
        "--loo_db", type=str, help="path to LOO results CSV file", default=None
    )
    parser.add_argument(
        "--dataset", type=str, default="cifar20", help="name for dataset"
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
        default=None,
    )
    parser.add_argument(
        "--model_behavior",
        type=str,
        help="Model behavior to be attributed",
        default="fid",
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
    parser.add_argument(
        "--max_depth",
        type=int,
        help="maximum depth for surrogateshap",
        default=5,
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
            "das",
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
        default=10240,
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
        default="results/cifar20/percentiles",
        help="Directory to save percentile results",
    )
    args = parser.parse_args()

    return args


def main(args):
    """Main function to compute Shapley values and evaluate."""

    test_seeds = [42, 43, 44, 45, 46]
    test_paths = [
        os.path.join(
            "results",
            args.dataset,
            f"datamodel/retrain_datamodel_{args.datamodel_alpha}_seed={s}.csv",
        )
        for s in test_seeds
    ]
    test_dfs = [pd.read_csv(p) for p in test_paths]

    # Build test matrices for all three seeds
    test_data = []
    for test_df in test_dfs:
        X_test, y_test = build_matrix(
            test_df, args.model_behavior, args.dataset, args.datamodel_alpha
        )
        test_data.append((X_test, y_test))

    # Only load training data for methods that require it
    if args.method in ["kernel_shap", "surrogateshap"]:
        if args.fit_db is None:
            raise ValueError(f"Method '{args.method}' requires --fit_db argument")
        train_df = pd.read_csv(args.fit_db)
        X_train, y_train = build_matrix(
            train_df, args.model_behavior, args.dataset, args.datamodel_alpha
        )

    if args.method in ["kernel_shap", "surrogateshap"]:
        subsets = [
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
            150,
            200,
            300,
            400,
            500,
            600,
            700,
        ]
    else:
        subsets = [0]

    for s in subsets:
        if args.method in ["kernel_shap", "surrogateshap"]:
            X_train_sub = X_train[:s, :]
            y_train_sub = y_train[:s]

        if args.method == "kernel_shap":
            shapley_values = data_shapley(
                dataset_size=X_train.shape[1],
                x_train=X_train_sub,
                y_train=y_train_sub,
                v1=args.v1,
                v0=args.v0,
            )
            coeff = np.asarray(shapley_values, dtype=float).reshape(-1)
        elif args.method == "trak":
            coeff = compute_gradient_scores(
                args.dataset,
                args.gradient_type,
                args.k_partition,
                args.projector_dim,
                args.sample_size,
                args.aggregation,
            )
        elif args.method == "pixel_dist":
            coeff = pixel_distance(
                aggregation=args.aggregation,
                dataset_name=args.dataset,
                sample_size=args.sample_size,
                generated_dir=args.generated_dir,
                training_dir=args.training_dir,
            )
        elif args.method == "clip_score":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            clip = CLIPScore(device)
            coeff = clip.clip_score(
                args.aggregation,
                args.dataset,
                args.n_samples if args.sample_size is None else args.sample_size,
                args.generated_dir,
                args.training_dir,
            )
        elif args.method == "gmv":
            coeff = gmv_topk_distance(
                args.aggregation,
                args.dataset,
                args.n_samples if args.sample_size is None else args.sample_size,
                args.generated_dir,
                args.training_dir,
                distance="lpips",
                topk=50,
            )
        elif args.method == "loo":
            loo_results = pd.read_csv(args.fit_db)
            full_model_behavior = args.v1
            coeff = np.zeros(X_test.shape[1], dtype=float)
            for idx, row in loo_results.iterrows():
                removal_idx = int(row["removal_idx"])
                coeff[removal_idx] = full_model_behavior - row[args.model_behavior]
        elif args.method == "surrogateshap":
            v0 = args.v0
            v1 = args.v1
            main, inter, ev = surrogateshap(
                X_train_sub,
                y_train_sub,
                v0,
                v1,
                args.max_depth,
            )
            coeff = main[0]  # (n,)
            Phi = inter[0]  # (n,n) symmetric; diag == phi

        # Store results for all three test sets
        all_metrics = []
        # all_corrs = []

        for X_test, y_test in test_data:
            if args.method == "surrogateshap":
                # additive prediction
                pred_y = v0 + X_test @ coeff
                quad = metrics(y_test, pred_y)
                # corr = spearmanr(phi, kernel_shap).statistic

                # quadratic prediction
                y_pred_quad = predict_quadratic(X_test, v0, coeff, Phi)
                quad = metrics(y_test, y_pred_quad)
            else:
                pred_y = (X_test @ coeff).reshape(-1)
                quad = metrics(y_test, pred_y)
                # corr = spearmanr(coeff, kernel_shap).statistic
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
            f"Spearman={mean_spearman:.4f}±{std_spearman:.4f}, "
            f"Pearson={mean_pearson:.4f}±{std_pearson:.4f}, "
            f"RMSE={mean_rmse:.4f}±{std_rmse:.4f}, "
            f"R2={mean_r2:.4f}±{std_r2:.4f}, "
        )

    # Save top/bottom percentiles if requested
    if args.output_percentiles:
        percentiles_dir = os.path.join(args.percentiles_dir)
        save_top_bottom_percentiles(
            coeff, args.dataset, args.method, args.model_behavior, percentiles_dir
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
