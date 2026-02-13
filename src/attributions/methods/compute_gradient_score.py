"""Calcuation D-TRAK, relative IF, randomized IF"""
import os
import pickle
import time

import numpy as np

import src.constants as constants
from src.datasets import create_dataset


def load_error(dataset_name, opt_seed=42):
    """Load training error for DAS method."""
    error_path = os.path.join(
        constants.OUTDIR,
        f"seed{opt_seed}",
        dataset_name,
        "das_losses",
        "das_loss.pkl",
    )

    print(f"Loading DAS training error from {error_path}...")
    with open(error_path, "rb") as handle:
        error_data = pickle.load(handle)

    # Extract losses from dict format (output from compute_das_loss.py)
    if isinstance(error_data, dict) and "losses" in error_data:
        error_train = error_data["losses"]
        print(
            f"  Loaded {error_data['num_timesteps']} timesteps, "
            f"{error_data['num_samples']} samples"
        )
    elif isinstance(error_data, list):
        error_train = np.concatenate(error_data, axis=-1)
    else:
        error_train = error_data

    # error_train shape: (num_timesteps, num_samples)
    # Transpose to (num_samples, num_timesteps)
    error_train = np.swapaxes(error_train, 0, 1)

    # Take square root of MSE to get RMSE
    error_train = np.sqrt(error_train)

    # Normalize across timesteps
    error_train_norm = np.linalg.norm(error_train, axis=-1, keepdims=True)
    error_train = error_train / (error_train_norm + 1e-8)

    # Average across timesteps to get per-sample error
    error_train = error_train.mean(axis=-1)

    print(f"Successfully loaded error of shape {error_train.shape}")
    return error_train


def compute_gradient_scores(
    dataset_name,
    gradient_type,
    k_partition,
    projector_dim,
    sample_size=10240,
    aggregation="mean",
    opt_seed=42,
):
    """Compute scores for D-TRAK, TRAK, and influence function."""
    dataset, _ = create_dataset(dataset_name=dataset_name, train=True)

    if gradient_type == "d_trak":
        model_behavior = "mean-squared-l2-norm"
        t_strategy = "uniform"
    else:
        model_behavior = "loss"
        t_strategy = "uniform"

    params = f"f={model_behavior}_t={t_strategy}_k={k_partition}_d={projector_dim}"
    reference_f_name = "reference_" + params
    train_f_name = "train_" + params

    if gradient_type == "journey_trak":
        gen_f_name = "gen_" + params

        val_grad_path = os.path.join(
            constants.OUTDIR,
            "seed42",
            dataset_name,
            "d_trak",
            gen_f_name,
        )
    else:
        val_grad_path = os.path.join(
            constants.OUTDIR,
            "seed42",
            dataset_name,
            "d_trak",
            reference_f_name,
        )
    print(f"Loading pre-calculated grads for validation set from {val_grad_path}...")

    val_phi = np.memmap(
        val_grad_path,
        dtype=np.float32,
        mode="r",
        shape=(sample_size, projector_dim),
    )

    # retraining free gradient methods

    train_grad_dir = os.path.join(constants.OUTDIR, "seed42", dataset_name, "d_trak")
    train_grad_path = os.path.join(
        train_grad_dir,
        train_f_name,
    )
    kernel_path = os.path.join(
        train_grad_dir,
        f"kernel_{train_f_name}.npy",
    )
    print(f"Loading pre-calculated grads for training set from {train_grad_path}...")
    train_phi = np.memmap(
        train_grad_path,
        dtype=np.float32,
        mode="r",
        shape=(len(dataset), projector_dim),
    )

    if os.path.isfile(kernel_path):
        # Check if the kernel file exists
        print("Kernel file exists. Loading...")
        kernel = np.load(kernel_path)
    else:
        starttime = time.time()
        kernel = train_phi.T @ train_phi
        kernel = kernel + 5e-1 * np.eye(kernel.shape[0])
        kernel = np.linalg.inv(kernel)
        np.save(kernel_path, kernel)
        print(time.time() - starttime)

    if gradient_type == "vanilla_gradient":
        train_phi = train_phi / np.linalg.norm(train_phi, axis=1, keepdims=True)
        val_phi = val_phi / np.linalg.norm(val_phi, axis=1, keepdims=True)
        scores = np.dot(val_phi, train_phi.T)
    else:
        if gradient_type == "relative_if":
            magnitude = np.linalg.norm((train_phi @ kernel).T, axis=0)
        elif gradient_type == "renormalized_if":
            magnitude = np.linalg.norm(train_phi.T, axis=0)
        else:
            magnitude = 1.0

        scores = val_phi @ ((train_phi @ kernel).T) / magnitude

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    # Using the generated sample average as coefficients
    scores = np.mean(scores, axis=0)

    # Apply DAS training error weighting
    if gradient_type == "das":
        training_error = load_error(dataset_name, opt_seed)
        scores = scores * training_error
    labels = np.array(dataset.targets)
    unique_values = sorted(set(labels))
    # Map original labels to contiguous indices 0, 1, 2, ...
    value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    label_indices = np.array([value_to_index[label] for label in labels])
    num_labels = len(unique_values)

    # Compute average/max score per class
    result = np.zeros(num_labels)
    for i in range(num_labels):
        label_mask = label_indices == i
        if aggregation == "max":
            result[i] = scores[label_mask].max()
        elif aggregation == "mean":
            result[i] = scores[label_mask].mean()
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

    return result
