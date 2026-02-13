"""Scripts that create synthetic data generation for Shapley value"""
import argparse
import itertools
import math

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.tree import DecisionTreeRegressor

# from xgboost import XGBRegressor


# Method configurations for plotting
METHOD_CONFIGS = {
    "kernelshap": {"marker": "o", "color_idx": 0, "label": "KernelSHAP", "alpha": 0.8},
    "mc_permutation": {
        "marker": "x",
        "color_idx": 1,
        "label": "MC Permutation",
        "alpha": 0.8,
    },
    "surrogate_xgboost": {
        "marker": "^",
        "color_idx": 2,
        "label": "SurrogateSHAP",
        "linewidth": 2.5,
    },
    "surrogate_xgboost_kernelmasks": {
        "marker": "v",
        "color_idx": 5,
        "label": "SurrogateSHAP (Shapley kernel)",
        "linewidth": 2.5,
    },
    "proxyspex_gbt_order1": {
        "marker": "D",
        "color_idx": 3,
        "label": "ProxySPEX (order=1)",
        "linestyle": "--",
        "alpha": 0.4,
    },
    "proxyspex_gbt_order2": {
        "marker": "v",
        "color_idx": 3,
        "label": "ProxySPEX (order=2)",
        "linestyle": "-.",
        "alpha": 0.7,
    },
    "proxyspex_gbt_order3": {
        "marker": "s",
        "color_idx": 3,
        "label": "ProxySPEX (order=3)",
        "linestyle": ":",
        "alpha": 1.0,
    },
}


def plot_method_on_axis(ax, summary_df, method_key, metric_col, colors, **plot_kwargs):
    """Helper function to plot a method on an axis with consistent styling."""
    if metric_col not in summary_df.columns:
        return False

    config = METHOD_CONFIGS.get(method_key, {})
    color = colors[config.get("color_idx", 0)]

    plot_params = {
        "marker": config.get("marker", "o"),
        "color": color,
        "linewidth": config.get("linewidth", 2),
        "label": config.get("label", method_key),
    }

    # Add optional parameters
    if "alpha" in config:
        plot_params["alpha"] = config["alpha"]
    if "linestyle" in config:
        plot_params["linestyle"] = config["linestyle"]

    # Override with any passed kwargs
    plot_params.update(plot_kwargs)

    # Check if this is an errorbar plot (has std column)
    std_col = metric_col.replace("_mean", "_std")
    if "_mean" in metric_col and std_col in summary_df.columns:
        plot_params["capsize"] = 4
        ax.errorbar(
            summary_df["n_subsets"],
            summary_df[metric_col],
            yerr=summary_df[std_col],
            **plot_params,
        )
    else:
        ax.plot(summary_df["n_subsets"], summary_df[metric_col], **plot_params)

    return True


# Experiment helper classes and functions
class MethodResults:
    """Container for tracking results of a single method across trials."""

    def __init__(self, method_name):
        self.method_name = method_name
        self.errors = []
        self.recall1 = []
        self.recall5 = []
        self.all_phi = []

    def add_trial(self, phi, phi_oracle):
        """Add results from a single trial."""
        self.all_phi.append(phi)
        denom = np.linalg.norm(phi_oracle) + 1e-12
        self.errors.append(np.linalg.norm(phi - phi_oracle) / denom)
        self.recall1.append(recall_at_k(phi, phi_oracle, 1))
        self.recall5.append(recall_at_k(phi, phi_oracle, 5))

    def compute_summary(self, phi_oracle):
        """Compute mean/std and bias-variance decomposition."""
        bv = phi_bias_var_decomp(self.all_phi, phi_oracle)
        return {
            f"{self.method_name}_rel_L2_mean": float(np.mean(self.errors)),
            f"{self.method_name}_rel_L2_std": float(np.std(self.errors)),
            f"{self.method_name}_rel_bias2": float(bv["rel_bias2"]),
            f"{self.method_name}_rel_var": float(bv["rel_var"]),
            f"{self.method_name}_rel_mse": float(bv["rel_mse"]),
            f"{self.method_name}_recall1_mean": float(np.mean(self.recall1)),
            f"{self.method_name}_recall1_std": float(np.std(self.recall1)),
            f"{self.method_name}_recall5_mean": float(np.mean(self.recall5)),
            f"{self.method_name}_recall5_std": float(np.std(self.recall5)),
        }


def phi_bias_var_decomp(phis_list, phi_oracle):
    """
    Given a list/array of phi vectors from repeated trials, compute:
      bias^2 = ||E[phi_hat] - phi*||^2
      var    = E ||phi_hat - E[phi_hat]||^2
      mse    = bias^2 + var
    Also returns relative versions dividing by ||phi*||^2.

    Returns a dict of scalar summaries + optional per-feature variance stats.
    """
    phis = np.asarray(phis_list, dtype=float)  # (R, M)
    phi_oracle = np.asarray(phi_oracle, dtype=float).reshape(-1)
    assert phis.ndim == 2 and phis.shape[1] == phi_oracle.shape[0]

    mean_phi = phis.mean(axis=0)
    diff_mean = mean_phi - phi_oracle

    bias2 = float(np.sum(diff_mean ** 2))
    var = float(np.mean(np.sum((phis - mean_phi) ** 2, axis=1)))
    mse = bias2 + var

    denom = float(np.sum(phi_oracle ** 2)) + 1e-12
    rel_bias2 = bias2 / denom
    rel_var = var / denom
    rel_mse = mse / denom

    # per-feature variance (useful diagnostics)
    per_feat_var = np.var(phis, axis=0, ddof=0)
    return {
        "bias2": bias2,
        "var": var,
        "mse": mse,
        "rel_bias2": rel_bias2,
        "rel_var": rel_var,
        "rel_mse": rel_mse,
        "per_feat_var_mean": float(np.mean(per_feat_var)),
        "per_feat_var_max": float(np.max(per_feat_var)),
    }


def recall_at_k(estimated_phis, oracle_phis, k):
    """Compute Recall@k for top-k feature identification."""
    # Get indices of top-k features according to oracle
    oracle_top_k = set(np.argsort(np.abs(oracle_phis))[-k:])
    # Get indices of top-k features according to estimate
    estimated_top_k = set(np.argsort(np.abs(estimated_phis))[-k:])
    # Compute recall
    return len(oracle_top_k & estimated_top_k) / k


def plot_recall_comparison(summary_df, func_type="Nonlinear"):
    """Plot Recall@k comparison across methods."""
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("colorblind", n_colors=7)

    # Methods to plot in order
    methods_to_plot = [
        "kernelshap",
        "mc_permutation",
        "surrogate_xgboost",
        "surrogate_xgboost_kernelmasks",
        "proxyspex_gbt_order3",
    ]

    # Recall@1
    for method in methods_to_plot:
        plot_method_on_axis(
            axes[0], summary_df, method, f"{method}_recall1_mean", colors
        )

    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("A. Recall@1", fontsize=18, fontweight="bold")
    axes[0].set_xlabel("Number of Coalition Samples ($M$)", fontsize=16)
    axes[0].set_ylabel("Recall@1", fontsize=16)
    axes[0].tick_params(axis="both", which="major", labelsize=14)
    axes[0].legend(fontsize=13, loc="lower right")
    axes[0].grid(True, alpha=0.4)

    # Recall@5
    for method in methods_to_plot:
        plot_method_on_axis(
            axes[1], summary_df, method, f"{method}_recall5_mean", colors
        )

    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("B. Recall@5", fontsize=18, fontweight="bold")
    axes[1].set_xlabel("Number of Coalition Samples ($M$)", fontsize=16)
    axes[1].set_ylabel("Recall@5", fontsize=16)
    axes[1].tick_params(axis="both", which="major", labelsize=14)
    axes[1].legend(fontsize=13, loc="lower right")
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        f"shapley_exp/recall_comparison_{func_type}.pdf", bbox_inches="tight", dpi=300
    )
    plt.show()


def plot_validation_results(summary_df, func):
    """Generates a 3-panel validation plot for the Surrogate SHAP method."""
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = sns.color_palette("colorblind", n_colors=7)

    # Methods for validation plots (no error bars)
    val_methods = [
        "mc_permutation",
        "surrogate_xgboost",
        "proxyspex_gbt_order1",
        "proxyspex_gbt_order2",
        "proxyspex_gbt_order3",
    ]

    # 1. Oracle Fidelity (Rel L2 Error)
    for method in val_methods:
        col = (
            f"{method.replace('surrogate_', '')}_rel_L2"
            if method == "surrogate_xgboost"
            else f"{method}_rel_L2"
        )
        if method == "surrogate_xgboost":
            col = "xgb_rel_L2"
        elif method == "mc_permutation":
            col = "mc_perm_rel_L2"
        else:
            col = f"{method}_rel_L2"
        plot_method_on_axis(axes[0], summary_df, method, col, colors)

    axes[0].set_yscale("log")
    axes[0].legend(fontsize=13, loc="best")
    axes[0].set_title("A. Oracle Fidelity", fontsize=18, fontweight="bold")
    axes[0].set_xlabel("Sampled Coalitions (M)", fontsize=16)
    axes[0].set_ylabel("Relative $L_2$ Error (Log Scale)", fontsize=16)
    axes[0].tick_params(axis="both", which="major", labelsize=14)
    axes[0].grid(True, which="both", ls="-", alpha=0.5)

    # 2. Null Player Test
    for method in val_methods:
        if method == "surrogate_xgboost":
            col = "xgb_null_player_val"
        elif method == "mc_permutation":
            col = "mc_perm_null_player_val"
        else:
            col = f"{method}_null_player_val"
        plot_method_on_axis(axes[1], summary_df, method, col, colors)

    # Compute max for y-limit
    null_cols = [c for c in summary_df.columns if "null_player_val" in c]
    max_null = max([summary_df[c].max() for c in null_cols])
    axes[1].set_ylim(0, max_null * 1.2)
    axes[1].set_title("B. Null Player Test", fontsize=18, fontweight="bold")
    axes[1].legend(fontsize=13, loc="best")
    axes[1].set_xlabel("Sampled Coalitions (M)", fontsize=16)
    axes[1].set_ylabel(r"Attribution Magnitude ($\phi_{null}$)", fontsize=16)
    axes[1].tick_params(axis="both", which="major", labelsize=14)

    # 3. Efficiency Gap
    for method in val_methods:
        if method == "surrogate_xgboost":
            col = "xgb_efficiency_gap"
        elif method == "mc_permutation":
            col = "mc_perm_efficiency_gap"
        else:
            col = f"{method}_efficiency_gap"
        plot_method_on_axis(axes[2], summary_df, method, col, colors)

    # Compute max for y-limit
    gap_cols = [c for c in summary_df.columns if "efficiency_gap" in c]
    max_gap = max([summary_df[c].max() for c in gap_cols])
    axes[2].set_ylim(0, max(0.01, max_gap * 1.2))
    axes[2].set_title("C. Efficiency Gap", fontsize=18, fontweight="bold")
    axes[2].legend(fontsize=13, loc="best")
    axes[2].set_xlabel("Sampled Coalitions (M)", fontsize=16)
    axes[2].set_ylabel(r"Abs. Gap |$\sum \phi - \Delta v$|", fontsize=16)
    axes[2].tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    plt.savefig(f"shapley_exp/surrogate_shap_validation_{func}.png", dpi=300)
    plt.show()


def plot_efficiency_comparison(summary_df, func_type="Nonlinear"):
    """Paper-quality plot comparing KernelSHAP vs. Tree-based Surrogates."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.figure(figsize=(9, 5))
    colors = sns.color_palette("colorblind", n_colors=7)

    # Methods to plot
    methods_to_plot = [
        "kernelshap",
        "mc_permutation",
        "surrogate_xgboost",
        "proxyspex_gbt_order1",
        "proxyspex_gbt_order2",
        "proxyspex_gbt_order3",
    ]

    for method in methods_to_plot:
        plot_method_on_axis(
            plt.gca(), summary_df, method, f"{method}_rel_L2_mean", colors
        )

    # Styling for Academic Journals
    plt.yscale("log")
    plt.xlabel("Number of Coalition Samples ($M$)", fontsize=16)
    plt.ylabel(r"Relative $L_2$ Error", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)

    plt.legend(
        fontsize=13,
        frameon=True,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=3,
        columnspacing=1.0,
        borderaxespad=0,
    )
    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        f"shapley_exp/efficiency_comparison_{func_type}.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

    plt.tight_layout()
    plt.savefig(f"shapley_exp/shapley_efficiency_{func_type}.pdf", bbox_inches="tight")
    plt.show()


def plot_bias_variance_decomposition(summary_df, func_type="Nonlinear"):
    """Plot bias and variance decomposition for all methods."""
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = sns.color_palette("colorblind", n_colors=7)

    methods_to_plot = [
        "kernelshap",
        "mc_permutation",
        "surrogate_xgboost",
        "surrogate_xgboost_kernelmasks",
        "proxyspex_gbt_order3",
    ]

    # Plot 1: Bias²
    for method in methods_to_plot:
        plot_method_on_axis(axes[0], summary_df, method, f"{method}_rel_bias2", colors)

    axes[0].set_yscale("log")
    axes[0].set_title("A. Bias² (Systematic Error)", fontsize=18, fontweight="bold")
    axes[0].set_xlabel("Number of Coalition Samples ($M$)", fontsize=16)
    axes[0].set_ylabel(
        r"Relative Bias²",
        fontsize=14,
    )
    axes[0].tick_params(axis="both", which="major", labelsize=14)
    axes[0].grid(True, which="both", ls="--", alpha=0.4)

    # Plot 2: Variance
    for method in methods_to_plot:
        plot_method_on_axis(axes[1], summary_df, method, f"{method}_rel_var", colors)

    axes[1].set_yscale("log")
    axes[1].set_title("B. Variance (Estimation Spread)", fontsize=18, fontweight="bold")
    axes[1].set_xlabel("Number of Coalition Samples ($M$)", fontsize=16)
    axes[1].set_ylabel(r"Relative Variance", fontsize=14)
    axes[1].tick_params(axis="both", which="major", labelsize=14)
    axes[1].grid(True, which="both", ls="--", alpha=0.4)

    # Add single legend outside the figure to the right
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=13
    )

    plt.tight_layout()
    plt.savefig(
        f"shapley_exp/bias_variance_decomp_{func_type}.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def plot_mse_decomposition(summary_df, func_type="Nonlinear"):
    """Plot MSE decomposition showing bias² + variance = MSE."""
    sns.set_theme(style="whitegrid", context="paper")

    # Check if kernelmasks data exists
    has_kernelmasks = "surrogate_xgboost_kernelmasks_rel_mse" in summary_df.columns
    num_methods = 5 if has_kernelmasks else 4
    fig, axes = plt.subplots(1, num_methods, figsize=(5.5 * num_methods, 5))

    colors = sns.color_palette("colorblind", n_colors=7)
    methods = [
        ("KernelSHAP", "kernelshap", "o", colors[0]),
        ("MC Permutation", "mc_permutation", "x", colors[1]),
        ("SurrogateSHAP (uniform)", "surrogate_xgboost", "^", colors[2]),
        ("ProxySPEX (order=3)", "proxyspex_gbt_order3", "s", colors[3]),
    ]

    if has_kernelmasks:
        methods.insert(
            3,
            (
                "SurrogateSHAP (Shapley kernel)",
                "surrogate_xgboost_kernelmasks",
                "v",
                colors[5],
            ),
        )

    for idx, (name, prefix, marker, color) in enumerate(methods):
        ax = axes[idx]

        # Plot bias², variance, and MSE
        ax.plot(
            summary_df["n_subsets"],
            summary_df[f"{prefix}_rel_bias2"],
            marker=marker,
            color=color,
            linewidth=2,
            label="Bias²",
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(
            summary_df["n_subsets"],
            summary_df[f"{prefix}_rel_var"],
            marker=marker,
            color=color,
            linewidth=2,
            label="Variance",
            linestyle="-.",
            alpha=0.7,
        )
        ax.plot(
            summary_df["n_subsets"],
            summary_df[f"{prefix}_rel_mse"],
            marker=marker,
            color=color,
            linewidth=2.5,
            label="MSE (Bias² + Var)",
            linestyle="-",
        )

        ax.set_yscale("log")
        ax.set_title(f"{chr(65+idx)}. {name}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Coalition Samples ($M$)", fontsize=14)
        ax.set_ylabel("Relative Error", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.legend(fontsize=12, loc="best")
        ax.grid(True, which="both", ls="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(
        f"shapley_exp/mse_decomposition_{func_type}.pdf", bbox_inches="tight", dpi=300
    )
    plt.show()


def _int_to_mask(x: int, M: int) -> np.ndarray:
    m = np.zeros(M, dtype=int)
    for i in range(M):
        if x & (1 << i):
            m[i] = 1
    return m


def mobius_from_values(f: np.ndarray, M: int) -> np.ndarray:
    """
    Subset Möbius transform.
    Input: f[S] for all S in {0,1}^M, indexed by int bitmask (0..2^M-1).
    Output: mu[S] such that f[S] = sum_{T subseteq S} mu[T].
    """
    mu = f.astype(float).copy()
    for i in range(M):
        bit = 1 << i
        for s in range(1 << M):
            if s & bit:
                mu[s] -= mu[s ^ bit]
    return mu


def powerset_masks(M):
    """All 2^M binary masks as (2^M, M)."""
    return np.array(list(itertools.product([0, 1], repeat=M)), dtype=int)


def shapley_from_value_fn(M, v_of_mask):
    """Exact Shapley values for v:{0,1}^M->R via full enumeration."""
    factorial = math.factorial
    M_fact = factorial(M)
    phis = np.zeros(M, dtype=float)

    masks = powerset_masks(M)
    v_cache = {tuple(m.tolist()): float(v_of_mask(m)) for m in masks}

    for i in range(M):
        s = 0.0
        for m in masks:
            if m[i] == 1:
                continue
            k = int(m.sum())
            m_with = m.copy()
            m_with[i] = 1
            w = factorial(k) * factorial(M - k - 1) / M_fact
            s += w * (v_cache[tuple(m_with.tolist())] - v_cache[tuple(m.tolist())])
        phis[i] = s
    return phis


def mc_on_proxy_shapley_permutation(
    v_of_mask,
    M: int,
    *,
    budget_calls: int,
    seed: int = 0,
    reuse_empty: bool = True,
    cache: bool = False,
):
    """
    Monte Carlo Shapley baseline on the *proxy game* v:{0,1}^M -> R using
    permutation sampling (a.k.a. Shapley permutation estimator).

    For each random permutation pi, we evaluate v on the growing prefix coalitions:
      S0 = empty
      St = {pi[0],...,pi[t-1]}
    and accumulate marginal contributions v(St ∪ {i}) - v(St).

    Cost:
      - If reuse_empty=True:  M calls per permutation + 1 initial empty call
      - If reuse_empty=False: (M+1) calls per permutation
    We stop when the next permutation would exceed budget_calls.

    Returns
    -------
    phis : (M,) np.ndarray
        Estimated Shapley values.
    info : dict
        n_permutations, n_calls, used_cache
    """
    rng = np.random.default_rng(seed)
    phis = np.zeros(M, dtype=float)
    n_calls = 0
    n_perms = 0

    # Optional memoization (can reduce calls if v is deterministic)
    memo = {}

    def eval_mask(mask: np.ndarray) -> float:
        nonlocal n_calls
        if cache:
            key = mask.tobytes()
            if key in memo:
                return memo[key]
        val = float(v_of_mask(mask))
        n_calls += 1
        if cache:
            memo[key] = val
        return val

    empty = np.zeros(M, dtype=int)

    v_empty = None
    if reuse_empty:
        if budget_calls < 1:
            return phis, {"n_permutations": 0, "n_calls": 0, "used_cache": cache}
        v_empty = eval_mask(empty)

    # Each permutation needs M calls if we reuse v(empty), else M+1 calls
    calls_per_perm = M if reuse_empty else (M + 1)

    while n_calls + calls_per_perm <= budget_calls:
        perm = rng.permutation(M)
        mask = np.zeros(M, dtype=int)

        prev = v_empty if reuse_empty else eval_mask(mask)

        for j in perm:
            mask[j] = 1
            cur = eval_mask(mask)
            phis[j] += cur - prev
            prev = cur

        n_perms += 1

    if n_perms > 0:
        phis /= n_perms

    return phis, {"n_permutations": n_perms, "n_calls": n_calls, "used_cache": cache}


def shapley_kernel_weight(M, k):
    """Kernel SHAP weight for subset size k=|S| (excluding endpoints)."""
    if k == 0 or k == M:
        return 0.0
    return (M - 1) / (math.comb(M, k) * k * (M - k))


def kernelshap_estimate(masks, values, M):
    """
    KernelSHAP-style weighted linear regression:
      values ≈ phi0 + sum_i phi_i * z_i
    We strongly enforce endpoints by huge weights (so include them in masks!).
    """
    X = np.c_[np.ones(len(masks)), masks]  # intercept + features
    y = values.astype(float)

    w = np.array([shapley_kernel_weight(M, int(m.sum())) for m in masks], dtype=float)

    # enforce empty and full with huge weights
    big = 1e6
    empty = np.zeros(M, dtype=int)
    full = np.ones(M, dtype=int)
    is_empty = (masks == empty).all(axis=1)
    is_full = (masks == full).all(axis=1)
    w[is_empty] = big
    w[is_full] = big

    Xw = X * np.sqrt(w)[:, None]
    yw = y * np.sqrt(w)
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    phi0 = beta[0]
    phis = beta[1:]
    return phi0, phis


def surrogate_tree_shap_estimate(masks, values, M, max_depth=None):
    """
    Fit a DecisionTree surrogate g(z) to v(S), then compute main-effect Shapley
    values using TreeSHAP (via shap.TreeExplainer).

    Return:
    ------
      phi0: expected value under background (here all-zeros)
      phis: SHAP values for z = ones(M)
    """

    reg = DecisionTreeRegressor(random_state=0, max_depth=max_depth)
    reg.fit(masks, values)

    background = np.zeros((1, M), dtype=float)  # all-zeros baseline
    x = np.ones((1, M), dtype=float)  # grand coalition

    explainer = shap.TreeExplainer(
        reg, data=background, feature_perturbation="interventional"
    )
    phis = explainer.shap_values(x)

    # For single-output regression, shap_values is (1, M)
    phis = np.array(phis).reshape(-1)

    # Expected value under the background distribution
    phi0 = float(explainer.expected_value)
    return phi0, phis


def surrogate_xgboost_shap_estimate(
    masks, values, M, n_estimators=500, max_depth=-1, learning_rate=0.05, num_leaves=31
):
    """Fit a GBDT surrogate g(z) to v(S), then compute SHAP using TreeSHAP."""
    # Use same model as ProxySPEX
    n_samples = masks.shape[0]
    adaptive_num_leaves = min(num_leaves, max(2, n_samples // 4))
    adaptive_n_estimators = min(n_estimators, max(10, n_samples))

    reg = lgb.LGBMRegressor(
        objective="regression",
        random_state=0,
        n_estimators=adaptive_n_estimators,
        learning_rate=learning_rate,
        num_leaves=adaptive_num_leaves,
        max_depth=max_depth,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_samples=1,
        min_split_gain=0.0,
        verbosity=-1,
    )
    reg.fit(masks, values)

    background = np.zeros((1, M), dtype=float)  # all-zeros baseline
    x = np.ones((1, M), dtype=float)  # grand coalition

    explainer = shap.TreeExplainer(
        reg, data=background, feature_perturbation="interventional"
    )
    phis = explainer.shap_values(x)

    # For single-output regression, shap_values is (1, M)
    phis = np.array(phis).reshape(-1)

    # Expected value under the background distribution
    phi0 = float(explainer.expected_value)
    return phi0, phis


def shapley_from_fourier(fourier_dict, d):
    """
    Proxyspex closed-form Shapley from Fourier coefficients:
      phi_i = (-2) * sum_{S ⊇ {i}, |S| odd} F(S)/|S|
    """
    phis = np.zeros(d, dtype=float)

    for S_tuple, coef in fourier_dict.items():
        s = int(sum(S_tuple))
        if s == 0:
            continue
        if (s % 2) == 0:
            continue  # ONLY odd-sized subsets

        contrib = float(np.real(coef)) / s  # Shapley is real-valued here
        for i, bit in enumerate(S_tuple):
            if bit:
                phis[i] += contrib

    phis *= -2.0
    return phis


def lgboost_tree_to_fourier(tree_info):
    """
    Strips the Fourier coefficients from an LGBoost tree
    Code adapted from:
        Gorji, Ali, Andisheh Amrollahi, and Andreas Krause.
        "Amortized SHAP values via sparse Fourier function approximation."
        arXiv preprint arXiv:2410.06300 (2024).
    """

    def fourier_tree_sum(left_fourier, right_fourier, feature):
        final_fourier = {}
        all_freqs_tuples = set(left_fourier.keys()).union(right_fourier.keys())
        for freq_tuple in all_freqs_tuples:
            final_fourier[freq_tuple] = (
                left_fourier.get(freq_tuple, 0) + right_fourier.get(freq_tuple, 0)
            ) / 2
            current_freq_set = set(freq_tuple)
            feature_set = {feature}
            united_set = current_freq_set.union(feature_set)
            final_fourier[tuple(sorted(united_set))] = 0.5 * left_fourier.get(
                freq_tuple, 0
            ) - 0.5 * right_fourier.get(freq_tuple, 0)
        return final_fourier

    def process_node(node):
        if "leaf_value" in node:
            # Leaf node: return constant Fourier coefficient
            return {tuple(): node["leaf_value"]}
        else:
            # Internal node: recursively process children
            feature = node["split_feature"]
            left_fourier = process_node(node["left_child"])
            right_fourier = process_node(node["right_child"])
            return fourier_tree_sum(left_fourier, right_fourier, feature)

    return process_node(tree_info["tree_structure"])


def lgboost_to_fourier(model):
    """Convert a trained LightGBM model to its Fourier representation."""
    final_fourier = []
    dumped_model = model.booster_.dump_model()
    for tree_info in dumped_model["tree_info"]:
        final_fourier.append(lgboost_tree_to_fourier(tree_info))

    combined_fourier = {}
    for fourier in final_fourier:
        for k, v in fourier.items():
            tuple_k = [0] * model.n_features_
            for feature in k:
                tuple_k[feature] = 1
            tuple_k = tuple(tuple_k)
            if tuple_k in combined_fourier:
                combined_fourier[tuple_k] += v
            else:
                combined_fourier[tuple_k] = v
    return combined_fourier


def surrogate_gbdt_proxyspex_shap_estimate(
    masks: np.ndarray,
    values: np.ndarray,
    d: int,
    *,
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    max_depth: int = -1,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    energy_keep: float | None = None,
    max_order: int | None = None,
):
    """Surrogate GBDT with ProxySPEX Shapley estimation."""
    masks = np.asarray(masks, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    assert masks.shape[1] == d, "masks must have shape (n_samples, d)"

    n_samples = masks.shape[0]
    adaptive_num_leaves = min(num_leaves, max(2, n_samples // 4))
    adaptive_n_estimators = min(n_estimators, max(10, n_samples))

    reg = lgb.LGBMRegressor(
        objective="regression",
        random_state=0,
        n_estimators=adaptive_n_estimators,
        learning_rate=learning_rate,
        num_leaves=adaptive_num_leaves,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_samples=1,
        min_split_gain=0.0,
        verbosity=-1,
    )
    reg.fit(masks, values)

    phi0 = float(reg.predict(np.zeros((1, d), dtype=float), validate_features=False)[0])

    four_dict = lgboost_to_fourier(reg)  # {0/1 tuple -> F(S)}

    # 1) Order filter (interaction degree cap)
    if max_order is not None:
        four_dict = {k: v for k, v in four_dict.items() if sum(k) <= max_order}

    # 2) Energy truncation on remaining terms
    if energy_keep is not None and 0.0 < energy_keep < 1.0 and len(four_dict) > 0:
        keys = list(four_dict.keys())
        coefs = np.array([complex(four_dict[k]) for k in keys])

        empty = tuple([0] * d)
        if empty in four_dict:
            coefs[keys.index(empty)] = 0.0  # don't spend budget on constant term

        energy = np.abs(coefs) ** 2
        total = float(energy.sum())
        if total > 0:
            order_idx = np.argsort(energy)[::-1]
            cum = np.cumsum(energy[order_idx]) / total
            kk = int(np.searchsorted(cum, energy_keep, side="left")) + 1
            keep_idx = set(order_idx[:kk].tolist())
            four_dict = {
                keys[i]: four_dict[keys[i]] for i in range(len(keys)) if i in keep_idx
            }

    phis = shapley_from_fourier(four_dict, d)
    return phi0, phis


def run_experiment(
    func="nonlinear",
    M=10,
    sample_sizes=(16, 32, 64, 128, 256, 512, 600, 700, 800, 900, 1024),
    trials=50,
    seed=0,
):
    """Run experiment comparing sample efficiency of Shapley estimation methods."""

    rng = np.random.default_rng(seed)

    if func == "linear":
        v = make_oracle_v_no_interaction_linear(M, seed=seed)
    elif func == "nonlinear":
        v = make_oracle_v_no_interaction_tanh(M, seed=seed)
    else:
        v = make_oracle_v(M, seed=seed)

    # Oracle Shapley for true v
    phi_oracle = shapley_from_value_fn(M, v)

    all_masks = powerset_masks(M)
    empty = np.zeros(M, dtype=int)
    full = np.ones(M, dtype=int)
    interior = all_masks[
        ~((all_masks == empty).all(axis=1) | (all_masks == full).all(axis=1))
    ]

    rows = []
    for n in sample_sizes:
        n = max(n, 2)

        # Initialize results containers for all methods
        results = {
            "kernelshap": MethodResults("kernelshap"),
            "surrogate_tree": MethodResults("surrogate_tree"),
            "surrogate_xgboost": MethodResults("surrogate_xgboost"),
            "proxyspex_gbt_order1": MethodResults("proxyspex_gbt_order1"),
            "proxyspex_gbt_order2": MethodResults("proxyspex_gbt_order2"),
            "proxyspex_gbt_order3": MethodResults("proxyspex_gbt_order3"),
            "mc_permutation": MethodResults("mc_permutation"),
        }

        for _ in range(trials):
            k = n - 2
            idx = rng.choice(len(interior), size=min(k, len(interior)), replace=False)
            masks = np.vstack([empty, full, interior[idx]])
            values = np.array([v(m) for m in masks], dtype=float)

            _, phi_k = kernelshap_estimate(masks, values, M)
            results["kernelshap"].add_trial(phi_k, phi_oracle)

            _, phi_s = surrogate_tree_shap_estimate(masks, values, M)
            results["surrogate_tree"].add_trial(phi_s, phi_oracle)

            _, phi_x = surrogate_xgboost_shap_estimate(masks, values, M)
            results["surrogate_xgboost"].add_trial(phi_x, phi_oracle)

            _, phi_ps1 = surrogate_gbdt_proxyspex_shap_estimate(
                masks=masks,
                values=values,
                d=M,
                n_estimators=500,
                max_depth=9,
                max_order=1,
            )
            results["proxyspex_gbt_order1"].add_trial(phi_ps1, phi_oracle)

            _, phi_ps2 = surrogate_gbdt_proxyspex_shap_estimate(
                masks=masks,
                values=values,
                d=M,
                n_estimators=500,
                max_depth=9,
                max_order=2,
            )
            results["proxyspex_gbt_order2"].add_trial(phi_ps2, phi_oracle)

            _, phi_ps3 = surrogate_gbdt_proxyspex_shap_estimate(
                masks=masks,
                values=values,
                d=M,
                n_estimators=500,
                max_depth=9,
                max_order=3,
            )
            results["proxyspex_gbt_order3"].add_trial(phi_ps3, phi_oracle)

            # MC permutation sampling
            phi_mc, _ = mc_on_proxy_shapley_permutation(
                v, M, budget_calls=n, seed=rng.integers(0, 1000000)
            )
            results["mc_permutation"].add_trial(phi_mc, phi_oracle)

        # Aggregate results into row
        row_data = {"M": M, "n_subsets": n}
        for method_results in results.values():
            row_data.update(method_results.compute_summary(phi_oracle))

        rows.append(row_data)

    return pd.DataFrame(rows)


def make_oracle_v(M, seed=0):
    """Define an oracle coalition function v(S)"""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, size=M)
    pairs = [(0, 1, 1.5), (2, 3, -1.0), (4, 5, 0.8)]
    triple = (0, 2, 4, 2.0)
    base = 0.25

    def v(mask):
        mask = np.asarray(mask, dtype=int)
        val = base + (w * mask).sum()
        for a, b, coef in pairs:
            val += coef * (mask[a] * mask[b])
        a, b, c, coef = triple
        val += coef * (mask[a] * mask[b] * mask[c])
        # nonlinearity so it's not close to linear
        return float(3.0 * np.tanh(val / 3.0))

    return v


def make_oracle_v_no_interaction_tanh(M, seed=0, base=0.25, scale=3.0):
    """Simple additive experiment with nonlinear (tanh) output"""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, size=M)

    def v(mask):
        mask = np.asarray(mask, dtype=float)
        s = base + (w * mask).sum()
        return float(scale * np.tanh(s / scale))

    return v


def make_oracle_v_no_interaction_linear(M, seed=0, base=0.25):
    """Simple additive experiment"""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, size=M)

    def v(mask):
        mask = np.asarray(mask, dtype=float)  # allow {0,1}
        return float(base + (w * mask).sum())

    return v


def run_enhanced_experiment(
    func="nonlinear",
    M=10,
    sample_sizes=(16, 32, 64, 128, 256, 512, 1024),
    trials=10,
    seed=7,
):
    """Run enhanced experiment with Null Player and Efficiency checks."""
    rng = np.random.default_rng(seed)

    # 1. Define Oracle
    if func == "linear":
        v_base = make_oracle_v_no_interaction_linear(M, seed=seed)
    elif func == "nonlinear":
        v_base = make_oracle_v_no_interaction_tanh(M, seed=seed)
    else:
        v_base = make_oracle_v(M, seed=seed)
    # 2. Wrap Oracle to include a "Null Player" at the last index
    # A Null Player has no effect on the output regardless of inclusion

    def v_with_null(mask):
        # We ignore the last element of the mask (index M)
        return v_base(mask[:-1])

    M_plus_null = M + 1
    phi_oracle = shapley_from_value_fn(M_plus_null, v_with_null)

    # Values for Efficiency Check: v(Grand Coalition) - v(Empty Set)
    v_grand = v_with_null(np.ones(M_plus_null))
    v_empty = v_with_null(np.zeros(M_plus_null))
    total_utility = v_grand - v_empty

    # all_masks = powerset_masks(M_plus_null)
    empty = np.zeros(M_plus_null, dtype=int)
    full = np.ones(M_plus_null, dtype=int)
    # interior = all_masks[
    #     ~((all_masks == empty).all(axis=1) | (all_masks == full).all(axis=1))
    # ]

    rows = []
    for n in sample_sizes:
        res = {
            "kernel": [],
            "tree": [],
            "xgb": [],
            "mc_perm": [],
            "ps1": [],
            "ps2": [],
            "ps3": [],
        }
        null_vals = {
            "kernel": [],
            "tree": [],
            "xgb": [],
            "mc_perm": [],
            "ps1": [],
            "ps2": [],
            "ps3": [],
        }
        eff_gaps = {
            "kernel": [],
            "tree": [],
            "xgb": [],
            "mc_perm": [],
            "ps1": [],
            "ps2": [],
            "ps3": [],
        }

        for _ in range(trials):
            # One-Step Uniform Size Sampling
            # Sample sizes k uniformly, then pick subset
            sizes = rng.integers(1, M_plus_null, size=n - 2)
            sampled_masks = []
            for k in sizes:
                m = np.zeros(M_plus_null, dtype=int)
                m[rng.choice(M_plus_null, size=k, replace=False)] = 1
                sampled_masks.append(m)

            masks = np.vstack([empty, full, sampled_masks])
            values = np.array([v_with_null(m) for m in masks], dtype=float)

            # Estimates
            # MC permutation sampling
            budget_calls = n  # Use same budget as number of samples
            phi_mc, _ = mc_on_proxy_shapley_permutation(
                v_with_null,
                M_plus_null,
                budget_calls=budget_calls,
                seed=rng.integers(0, 1000000),
            )

            methods = {
                "kernel": kernelshap_estimate(masks, values, M_plus_null),
                "tree": surrogate_tree_shap_estimate(masks, values, M_plus_null),
                "xgb": surrogate_xgboost_shap_estimate(masks, values, M_plus_null),
                "mc_perm": (0.0, phi_mc),  # phi0 not used, only phis
                "ps1": surrogate_gbdt_proxyspex_shap_estimate(
                    masks=masks,
                    values=values,
                    d=M_plus_null,
                    n_estimators=500,
                    max_depth=9,
                    max_order=1,
                ),
                "ps2": surrogate_gbdt_proxyspex_shap_estimate(
                    masks=masks,
                    values=values,
                    d=M_plus_null,
                    n_estimators=500,
                    max_depth=9,
                    max_order=2,
                ),
                "ps3": surrogate_gbdt_proxyspex_shap_estimate(
                    masks=masks,
                    values=values,
                    d=M_plus_null,
                    n_estimators=500,
                    max_depth=9,
                    max_order=3,
                ),
            }

            for name, (phi0, phis) in methods.items():
                # L2 Error
                res[name].append(
                    np.linalg.norm(phis - phi_oracle) / np.linalg.norm(phi_oracle)
                )
                # Null Player Test (Value assigned to the last player)
                null_vals[name].append(abs(phis[-1]))
                # Efficiency Gap: |Sum(phis) - (v_grand - v_empty)|
                eff_gaps[name].append(abs(np.sum(phis) - total_utility))

        rows.append(
            {
                "n_subsets": n,
                "xgb_rel_L2": np.mean(res["xgb"]),
                "xgb_null_player_val": np.mean(null_vals["xgb"]),
                "xgb_efficiency_gap": np.mean(eff_gaps["xgb"]),
                "kernel_rel_L2": np.mean(res["kernel"]),
                "kernel_efficiency_gap": np.mean(eff_gaps["kernel"]),
                # "mc_perm_rel_L2": np.mean(res["mc_perm"]),
                # "mc_perm_null_player_val": np.mean(null_vals["mc_perm"]),
                # "mc_perm_efficiency_gap": np.mean(eff_gaps["mc_perm"]),
                "proxyspex_gbt_order1_rel_L2": np.mean(res["ps1"]),
                "proxyspex_gbt_order1_null_player_val": np.mean(null_vals["ps1"]),
                "proxyspex_gbt_order1_efficiency_gap": np.mean(eff_gaps["ps1"]),
                "proxyspex_gbt_order2_rel_L2": np.mean(res["ps2"]),
                "proxyspex_gbt_order2_null_player_val": np.mean(null_vals["ps2"]),
                "proxyspex_gbt_order2_efficiency_gap": np.mean(eff_gaps["ps2"]),
                "proxyspex_gbt_order3_rel_L2": np.mean(res["ps3"]),
                "proxyspex_gbt_order3_null_player_val": np.mean(null_vals["ps3"]),
                "proxyspex_gbt_order3_efficiency_gap": np.mean(eff_gaps["ps3"]),
            }
        )

    return pd.DataFrame(rows)


def _mask_to_bytes(m: np.ndarray) -> bytes:
    return np.asarray(m, dtype=np.uint8).tobytes()


def sample_masks_kernelshap(
    M: int, n: int, rng: np.random.Generator, *, max_tries: int = 10000
):
    """
    Protocol B sampler for KernelSHAP (typical): sample sizes k with
      p(k) ∝ C(M,k) * w(k) = (M-1)/(k(M-k))  for k=1..M-1,
    then sample a subset uniformly among size-k subsets.

    Always includes empty and full if n>=2.
    Tries to avoid duplicates; falls back to allowing duplicates if needed.
    """
    n = int(n)
    empty = np.zeros(M, dtype=int)
    full = np.ones(M, dtype=int)
    if n <= 0:
        return np.zeros((0, M), dtype=int)
    if n == 1:
        return empty.reshape(1, -1)

    # size distribution
    ks = np.arange(1, M, dtype=int)  # 1..M-1
    p = 1.0 / (ks * (M - ks))
    p = p / p.sum()

    masks = [empty, full]
    seen = {_mask_to_bytes(empty), _mask_to_bytes(full)}

    remaining = n - 2
    tries = 0
    while remaining > 0 and tries < max_tries:
        tries += 1
        k = int(rng.choice(ks, p=p))
        m = np.zeros(M, dtype=int)
        m[rng.choice(M, size=k, replace=False)] = 1
        key = _mask_to_bytes(m)
        if key in seen:
            continue
        seen.add(key)
        masks.append(m)
        remaining -= 1

    # If we couldn't fill uniquely, allow duplicates (still respects call budget)
    while remaining > 0:
        k = int(rng.choice(ks, p=p))
        m = np.zeros(M, dtype=int)
        m[rng.choice(M, size=k, replace=False)] = 1
        masks.append(m)
        remaining -= 1

    return np.vstack(masks)


def sample_masks_size_uniform(
    M: int, n: int, rng: np.random.Generator, *, max_tries: int = 10000
):
    """
    Protocol B sampler for surrogate training (reasonable 'best-practice' default):
    sample k uniformly from 1..M-1, then subset uniformly among size-k subsets.

    Always includes empty and full if n>=2.
    Tries to avoid duplicates; falls back to allowing duplicates if needed.
    """
    n = int(n)
    empty = np.zeros(M, dtype=int)
    full = np.ones(M, dtype=int)
    if n <= 0:
        return np.zeros((0, M), dtype=int)
    if n == 1:
        return empty.reshape(1, -1)

    masks = [empty, full]
    seen = {_mask_to_bytes(empty), _mask_to_bytes(full)}

    remaining = n - 2
    tries = 0
    while remaining > 0 and tries < max_tries:
        tries += 1
        k = int(rng.integers(1, M))  # 1..M-1
        m = np.zeros(M, dtype=int)
        m[rng.choice(M, size=k, replace=False)] = 1
        key = _mask_to_bytes(m)
        if key in seen:
            continue
        seen.add(key)
        masks.append(m)
        remaining -= 1

    while remaining > 0:
        k = int(rng.integers(1, M))
        m = np.zeros(M, dtype=int)
        m[rng.choice(M, size=k, replace=False)] = 1
        masks.append(m)
        remaining -= 1

    return np.vstack(masks)


def run_experiment_b(
    func="nonlinear",
    M=10,
    sample_sizes=(16, 32, 64, 128, 256, 512, 600, 700, 800, 900, 1024),
    trials=50,
    seed=0,
):
    """
    Protocol B:
      - Each method uses its *own* recommended / natural sampling procedure
        under the same oracle-call budget n.
      - KernelSHAP: kernel-style size sampling (p(k) ∝ 1/(k(M-k))) + WLS.
      - Surrogates + ProxySPEX: size-uniform sampling for training data (covers sizes).
      - MC permutation: permutation estimator with budget_calls=n.
    """
    rng = np.random.default_rng(seed)

    if func == "linear":
        v = make_oracle_v_no_interaction_linear(M, seed=seed)
    elif func == "nonlinear":
        v = make_oracle_v_no_interaction_tanh(M, seed=seed)
    else:
        v = make_oracle_v(M, seed=seed)

    phi_oracle = shapley_from_value_fn(M, v)

    rows = []
    for n in sample_sizes:
        n = max(int(n), 2)

        # Initialize results containers for all methods
        results = {
            "kernelshap": MethodResults("kernelshap"),
            "surrogate_tree": MethodResults("surrogate_tree"),
            "surrogate_xgboost": MethodResults("surrogate_xgboost"),
            "surrogate_xgboost_kernelmasks": MethodResults(
                "surrogate_xgboost_kernelmasks"
            ),
            "proxyspex_gbt_order1": MethodResults("proxyspex_gbt_order1"),
            "proxyspex_gbt_order2": MethodResults("proxyspex_gbt_order2"),
            "proxyspex_gbt_order3": MethodResults("proxyspex_gbt_order3"),
            "mc_permutation": MethodResults("mc_permutation"),
        }

        for _ in range(trials):
            # --- KernelSHAP: kernel-style sampling ---
            masks_k = sample_masks_kernelshap(M, n, rng)
            values_k = np.array([v(m) for m in masks_k], dtype=float)
            _, phi_k = kernelshap_estimate(masks_k, values_k, M)
            results["kernelshap"].add_trial(phi_k, phi_oracle)

            # --- Surrogates / ProxySPEX: size-uniform training set ---
            masks_s = sample_masks_size_uniform(M, n, rng)
            values_s = np.array([v(m) for m in masks_s], dtype=float)

            _, phi_s = surrogate_tree_shap_estimate(masks_s, values_s, M)
            results["surrogate_tree"].add_trial(phi_s, phi_oracle)

            _, phi_x = surrogate_xgboost_shap_estimate(masks_s, values_s, M)
            results["surrogate_xgboost"].add_trial(phi_x, phi_oracle)

            # --- Use kernel-style masks for XGBoost surrogate ---
            _, phi_m = surrogate_xgboost_shap_estimate(masks_k, values_k, M)
            results["surrogate_xgboost_kernelmasks"].add_trial(phi_m, phi_oracle)

            _, phi_ps1 = surrogate_gbdt_proxyspex_shap_estimate(
                masks=masks_s,
                values=values_s,
                d=M,
                n_estimators=500,
                max_depth=9,
                max_order=1,
            )
            results["proxyspex_gbt_order1"].add_trial(phi_ps1, phi_oracle)

            _, phi_ps2 = surrogate_gbdt_proxyspex_shap_estimate(
                masks=masks_s,
                values=values_s,
                d=M,
                n_estimators=500,
                max_depth=9,
                max_order=2,
            )
            results["proxyspex_gbt_order2"].add_trial(phi_ps2, phi_oracle)

            _, phi_ps3 = surrogate_gbdt_proxyspex_shap_estimate(
                masks=masks_s,
                values=values_s,
                d=M,
                n_estimators=500,
                max_depth=9,
                max_order=3,
            )
            results["proxyspex_gbt_order3"].add_trial(phi_ps3, phi_oracle)

            # --- MC permutation baseline: budget_calls = n ---
            phi_mc, _ = mc_on_proxy_shapley_permutation(
                v, M, budget_calls=n, seed=int(rng.integers(0, 1_000_000))
            )
            results["mc_permutation"].add_trial(phi_mc, phi_oracle)

        # Aggregate results into row
        row_data = {"M": M, "n_subsets": n}
        for method_results in results.values():
            row_data.update(method_results.compute_summary(phi_oracle))

        rows.append(row_data)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SurrogateSHAP experiments with synthetic data"
    )
    parser.add_argument(
        "--func",
        type=str,
        default="nonlinear",
        choices=["linear", "nonlinear", "interaction"],
        help="Type of oracle function: linear, nonlinear (tanh), or interaction",
    )
    parser.add_argument(
        "--M", type=int, default=10, help="Number of features (default: 10)"
    )
    parser.add_argument(
        "--trials", type=int, default=30, help="Number of trials per sample size"
    )
    parser.add_argument(
        "--seed", type=int, default=7, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--sample_sizes",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256, 512, 600, 700, 800, 900],
        help="List of sample sizes to test",
    )

    args = parser.parse_args()

    func = args.func
    print(f"Running experiments with func={func}, M={args.M}, trials={args.trials}")

    summary = run_enhanced_experiment(
        func=func,
        M=args.M,
        sample_sizes=tuple(args.sample_sizes),
        trials=args.trials,
        seed=args.seed,
    )
    print("\n=== Enhanced Experiment Results ===")
    print(
        summary[
            [
                "n_subsets",
                "xgb_rel_L2",
                "proxyspex_gbt_order1_rel_L2",
                "proxyspex_gbt_order2_rel_L2",
                "proxyspex_gbt_order3_rel_L2",
            ]
        ]
    )
    plot_validation_results(summary, func=func)

    summary_b = run_experiment_b(
        func=func,
        M=args.M,
        sample_sizes=tuple(args.sample_sizes),
        trials=args.trials,
        seed=args.seed,
    )
    print(summary_b)
    plot_efficiency_comparison(summary_b, func_type=f"{func.capitalize()}")
    plot_recall_comparison(summary_b, func_type=f"{func.capitalize()}")
    plot_bias_variance_decomposition(summary_b, func_type=f"{func.capitalize()}")
    plot_mse_decomposition(summary_b, func_type=f"{func.capitalize()}")
