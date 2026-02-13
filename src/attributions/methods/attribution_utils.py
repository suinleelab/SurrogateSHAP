"""
Utility functions for data attribution calculation.
[1] GMValuator: Similarity-based Data Valuation for Generative Models
"""
import glob
import os
from typing import Callable, Optional

import clip
import lpips
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import DATASET_DIR
from src.datasets import ImageDataset, create_dataset


def process_images_np(file_list, max_size=None, target_size=None):
    """Function to load and process images into numpy arrays."""

    valid_extensions = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
    images = []
    filtered_files = [
        file for file in file_list if file.split(".")[-1].lower() in valid_extensions
    ]

    if max_size is not None:
        filtered_files = filtered_files[:max_size]

    for filename in tqdm(filtered_files):
        try:
            image = Image.open(filename).convert("RGB")

            # Resize only if image exceeds target_size
            if target_size is not None:
                w, h = image.size
                if max(w, h) > target_size:
                    from torchvision import transforms

                    # Resize so larger edge becomes target_size, then center crop
                    transform = transforms.Compose(
                        [
                            transforms.Resize(
                                target_size,
                                interpolation=transforms.InterpolationMode.BILINEAR,
                            ),
                            transforms.CenterCrop(target_size),
                        ]
                    )
                    image = transform(image)

            # Convert PIL Image to NumPy array and scale from 0 to 1
            image_np = np.array(image, dtype=np.float32) / 255.0

            # Normalize: shift and scale the image to have pixel values in range [-1, 1]
            image_np = (image_np - 0.5) / 0.5

            images.append(image_np)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    return np.stack(images) if images else np.array([])


class CLIPScore:
    """Class for initializing CLIP model and calculating clip score."""

    def __init__(self, device):
        self.device = device
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=device)

    def clip_score(
        self,
        aggregation,
        dataset_name,
        sample_size,
        sample_dir,
        training_dir,
    ):
        """
        Function that calculate CLIP score between generated and training data

        Args:
        ----
            aggregation: aggregate class based coefficients based on mean or max
            dataset_name: name of the dataset.
            sample_size: number of samples to calculate local model behavior
            sample_dir: directory of the first set of images.
            training_dir: directory of the second set of images.

        Return:
        ------
            Mean pairwise CLIP score as data attribution.
        """

        all_sample_features = []
        all_training_features = []
        num_workers = 4 if torch.get_num_threads() >= 4 else torch.get_num_threads()

        sample_dataset = ImageDataset(sample_dir, self.clip_transform, sample_size)
        sample_loader = DataLoader(
            sample_dataset, batch_size=64, num_workers=num_workers, pin_memory=True
        )
        train_dataset = ImageDataset(training_dir, transform=self.clip_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=64, num_workers=num_workers, pin_memory=True
        )

        # Assuming clip_transform is your image transformation pipeline

        with torch.no_grad():
            print(f"Calculating CLIP embeddings for {sample_dir}...")
            for sample_batch, _ in tqdm(sample_loader):
                features = self.clip_model.encode_image(sample_batch.to(self.device))
                all_sample_features.append(features.cpu().numpy())

            print(f"Calculating CLIP embeddings for {training_dir}...")
            for training_batch, _ in tqdm(train_loader):
                features = self.clip_model.encode_image(training_batch.to(self.device))
                all_training_features.append(features.cpu().numpy())

        # Concatenate all batch features
        all_sample_features = np.concatenate(all_sample_features, axis=0)
        all_training_features = np.concatenate(all_training_features, axis=0)

        all_sample_features = all_sample_features / np.linalg.norm(
            all_sample_features, axis=1, keepdims=True
        )
        all_training_features = all_training_features / np.linalg.norm(
            all_training_features, axis=1, keepdims=True
        )

        similarity = all_sample_features @ all_training_features.T

        # Average similarity across all sample images for each training image
        scores = np.mean(similarity, axis=0)

        dataset, _ = create_dataset(dataset_name=dataset_name, train=True)

        # Handle artbench with artist grouping
        if dataset_name == "artbench":
            # Load the same CSV used for removal to ensure alignment
            import pandas as pd

            from src.constants import DATASET_DIR

            artist_csv = os.path.join(
                DATASET_DIR,
                "artbench-10-imagefolder-split/train/post_impressionism_artists.csv",
            )
            artist_df = pd.read_csv(artist_csv)
            # Use CSV row order (not alphabetical!) to match removal indices
            unique_artists = artist_df.iloc[:, 0].tolist()
            artist_to_index = {artist: idx for idx, artist in enumerate(unique_artists)}
            artists = np.array(dataset["artist"])
            label_indices = np.array([artist_to_index[artist] for artist in artists])
            num_labels = len(unique_artists)
        elif dataset_name == "fashion":
            # Load the same CSV used for removal to ensure alignment
            import pandas as pd

            from src.constants import DATASET_DIR

            brand_csv = os.path.join(DATASET_DIR, "fashion-product/top100_brands.csv")
            brand_df = pd.read_csv(brand_csv, header=0)
            brand_df = brand_df.dropna(subset=["brand"])
            # Use CSV row order (not alphabetical!) to match removal indices
            unique_brands = brand_df["brand"].tolist()
            brand_to_index = {brand: idx for idx, brand in enumerate(unique_brands)}
            brands = np.array(dataset["brand"])
            label_indices = np.array([brand_to_index[brand] for brand in brands])
            num_labels = len(unique_brands)
        else:
            # Standard numeric label grouping
            labels = np.array(dataset.targets)
            unique_values = sorted(set(labels))
            # Map original labels to contiguous indices 0, 1, 2, ...
            value_to_index = {val: idx for idx, val in enumerate(unique_values)}
            label_indices = np.array([value_to_index[label] for label in labels])
            num_labels = len(unique_values)

        # Compute average/max score per class/artist
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


def pixel_distance(aggregation, dataset_name, sample_size, generated_dir, training_dir):
    """
    Function that calculate the pixel distance between two image sets,
    generated images and training images. Using the average distance
    across generated images as attribution value for training data.

    Args:
    ----
        aggregation : aggregated class based coefficients based on mean or max
        dataset_name: dataset
        sample_size: number of generated samples.
        generated_dir: directory of the generated images.
        training_dir: directory of the training set images.

    Return:
    ------
        Mean of pixel distance as data attribution.

    """
    print(f"Loading images from {generated_dir}..")

    generated_images = process_images_np(glob.glob(generated_dir + "/*"), sample_size)

    print(f"Loading images from {training_dir}..")

    ref_images = process_images_np(glob.glob(training_dir + "/*"))

    generated_images = generated_images.reshape(generated_images.shape[0], -1)
    ref_images = ref_images.reshape(ref_images.shape[0], -1)
    # Normalize the image vectors to unit vectors
    generated_images = generated_images / np.linalg.norm(
        generated_images, axis=1, keepdims=True
    )
    ref_images = ref_images / np.linalg.norm(ref_images, axis=1, keepdims=True)

    similarities = np.dot(generated_images, ref_images.T)

    # Average similarity across all generated images for each training image
    scores = np.mean(similarities, axis=0)

    dataset, _ = create_dataset(dataset_name=dataset_name, train=True)

    # Handle artbench with artist grouping
    if dataset_name == "artbench":

        artist_csv = os.path.join(
            DATASET_DIR,
            "artbench-10-imagefolder-split/train/post_impressionism_artists.csv",
        )
        artist_df = pd.read_csv(artist_csv)
        # Use CSV row order (not alphabetical!) to match removal indices
        unique_artists = artist_df.iloc[:, 0].tolist()
        artist_to_index = {artist: idx for idx, artist in enumerate(unique_artists)}
        artists = np.array(dataset["artist"])
        label_indices = np.array([artist_to_index[artist] for artist in artists])
        num_labels = len(unique_artists)
    elif dataset_name == "fashion":

        brand_csv = os.path.join(DATASET_DIR, "fashion-product/top100_brands.csv")
        brand_df = pd.read_csv(brand_csv, header=0)
        brand_df = brand_df.dropna(subset=["brand"])
        # Use CSV row order (not alphabetical!) to match removal indices
        unique_brands = brand_df["brand"].tolist()
        brand_to_index = {brand: idx for idx, brand in enumerate(unique_brands)}
        brands = np.array(dataset["brand"])
        label_indices = np.array([brand_to_index[brand] for brand in brands])
        num_labels = len(unique_brands)
    else:
        # Standard numeric label grouping
        labels = np.array(dataset.targets)
        unique_values = sorted(set(labels))
        # Map original labels to contiguous indices 0, 1, 2, ...
        value_to_index = {val: idx for idx, val in enumerate(unique_values)}
        label_indices = np.array([value_to_index[label] for label in labels])
        num_labels = len(unique_values)

    # Compute average/max score per class/artist
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


def gmv_topk_distance(
    aggregation: str,
    dataset_name: str,
    sample_size: int,
    generated_dir: str,
    training_dir: str,
    *,
    distance: str = "lpips",  # "cosine" | "l2" | "lpips"
    topk: int = 50,
    chunk_size: int = 2048,  # training chunk size to keep memory bounded
    lpips_net: str = "alex",  # "alex" | "vgg" | "squeeze"
    device: str = "cuda",
    quality_fn: Optional[
        Callable[[int], np.ndarray]
    ] = None,  # optional q_j per generated sample
):
    """GMValuator-style top-k attribution using a chosen distance measure from [1]."""
    # Load dataset first to get valid image filenames
    dataset, _ = create_dataset(dataset_name=dataset_name, train=True)

    print(f"Loading images from {generated_dir}..")
    gen_paths = glob.glob(os.path.join(generated_dir, "*"))
    generated_images = process_images_np(gen_paths, target_size=256)

    print(f"Loading images from {training_dir}..")
    # For fashion dataset, filter to only images in the dataset
    if dataset_name == "fashion":
        # Get valid filenames from dataset
        valid_filenames = set(dataset["filename"])
        train_paths = glob.glob(os.path.join(training_dir, "*"))
        # Filter to only files that are in the dataset
        train_paths = [p for p in train_paths if os.path.basename(p) in valid_filenames]
    else:
        train_paths = glob.glob(os.path.join(training_dir, "*"))

    ref_images = process_images_np(train_paths, target_size=256)

    # Ensure channel-last for numpy ops (N,H,W,C)
    def to_channel_last(x: np.ndarray) -> np.ndarray:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D images, got shape {x.shape}")
        # If looks like (N,C,H,W), convert to (N,H,W,C)
        if x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
            return np.transpose(x, (0, 2, 3, 1))
        return x

    generated_images = to_channel_last(generated_images).astype(np.float32)
    ref_images = to_channel_last(ref_images).astype(np.float32)

    m = generated_images.shape[0]
    n = ref_images.shape[0]
    k = min(topk, n)
    if k <= 0:
        raise ValueError("topk must be >= 1 and training set must be non-empty")

    # Optional quality weights per generated sample
    if quality_fn is None:
        q = np.ones(m, dtype=np.float32)
    else:
        q = np.asarray(quality_fn(m), dtype=np.float32).reshape(-1)
        if q.shape[0] != m:
            raise ValueError(
                "quality_fn must return an array of shape (num_generated,)"
            )

    # Accumulated per-training-image attribution
    phi = np.zeros(n, dtype=np.float32)

    dist = distance.lower().strip()

    if dist in ("cosine", "l2"):
        # Flatten
        gen_flat = generated_images.reshape(m, -1)
        ref_flat = ref_images.reshape(n, -1)

        if dist == "cosine":
            # Normalize to unit vectors
            gen_norm = gen_flat / (
                np.linalg.norm(gen_flat, axis=1, keepdims=True) + 1e-12
            )
            ref_norm = ref_flat / (
                np.linalg.norm(ref_flat, axis=1, keepdims=True) + 1e-12
            )

            # We need "distance" for top-k smallest,
            # so use cosine distance = 1 - similarity
            # We'll scan training in chunks to reduce memory.
            for j in range(m):
                best_d = np.full(k, np.inf, dtype=np.float32)
                best_i = np.full(k, -1, dtype=np.int64)

                gj = gen_norm[j : j + 1]  # (1,D)
                for t0 in range(0, n, chunk_size):
                    t1 = min(n, t0 + chunk_size)
                    sims = gj @ ref_norm[t0:t1].T  # (1,chunk)
                    dists = (1.0 - sims).reshape(-1).astype(np.float32)

                    # merge with current best
                    merged_d = np.concatenate([best_d, dists], axis=0)
                    merged_i = np.concatenate(
                        [best_i, np.arange(t0, t1, dtype=np.int64)], axis=0
                    )
                    idx = np.argpartition(merged_d, k - 1)[:k]
                    best_d = merged_d[idx]
                    best_i = merged_i[idx]

                # sort top-k
                order = np.argsort(best_d)
                best_d = best_d[order]
                best_i = best_i[order]

                # weights = softmax(-d)
                w = np.exp(-best_d - (-best_d).max())
                w = (w / (w.sum() + 1e-12)) * q[j]

                # accumulate
                np.add.at(phi, best_i, w)

        else:  # "l2" as mean squared distance
            for j in range(m):
                best_d = np.full(k, np.inf, dtype=np.float32)
                best_i = np.full(k, -1, dtype=np.int64)

                gj = gen_flat[j]  # (D,)
                for t0 in range(0, n, chunk_size):
                    t1 = min(n, t0 + chunk_size)
                    tj = ref_flat[t0:t1]  # (chunk,D)
                    diff = tj - gj[None, :]
                    dists = (diff * diff).mean(axis=1).astype(np.float32)  # (chunk,)

                    merged_d = np.concatenate([best_d, dists], axis=0)
                    merged_i = np.concatenate(
                        [best_i, np.arange(t0, t1, dtype=np.int64)], axis=0
                    )
                    idx = np.argpartition(merged_d, k - 1)[:k]
                    best_d = merged_d[idx]
                    best_i = merged_i[idx]

                order = np.argsort(best_d)
                best_d = best_d[order]
                best_i = best_i[order]

                w = np.exp(-best_d - (-best_d).max())
                w = (w / (w.sum() + 1e-12)) * q[j]
                np.add.at(phi, best_i, w)

    elif dist == "lpips":
        if lpips is None or torch is None:
            raise ImportError("distance='lpips' requires `pip install lpips torch`.")
        dev = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        net = lpips.LPIPS(net=lpips_net).to(dev).eval()

        # LPIPS expects (N,3,H,W) float in [-1,1]
        def to_torch_nchw(x: np.ndarray) -> torch.Tensor:
            # x: (N,H,W,C)
            x = np.transpose(x, (0, 3, 1, 2))  # (N,C,H,W)
            t = torch.from_numpy(x).to(dev)
            # assume x is in [0,1] or [0,255]; normalize to [0,1] if needed
            if t.max() > 1.5:
                t = t / 255.0
            return (t * 2.0 - 1.0).float()

        gen_t = to_torch_nchw(generated_images)
        ref_t = to_torch_nchw(ref_images)

        with torch.no_grad():
            for j in range(m):
                best_d = np.full(k, np.inf, dtype=np.float32)
                best_i = np.full(k, -1, dtype=np.int64)

                gj = gen_t[j : j + 1]  # (1,3,H,W)
                for t0 in range(0, n, chunk_size):
                    t1 = min(n, t0 + chunk_size)
                    tj = ref_t[t0:t1]  # (chunk,3,H,W)

                    # compute LPIPS between gj and each tj by expanding gj
                    gj_rep = gj.expand(tj.shape[0], -1, -1, -1)
                    d = (
                        net(gj_rep, tj)
                        .view(-1)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

                    merged_d = np.concatenate([best_d, d], axis=0)
                    merged_i = np.concatenate(
                        [best_i, np.arange(t0, t1, dtype=np.int64)], axis=0
                    )
                    idx = np.argpartition(merged_d, k - 1)[:k]
                    best_d = merged_d[idx]
                    best_i = merged_i[idx]

                order = np.argsort(best_d)
                best_d = best_d[order]
                best_i = best_i[order]

                w = np.exp(-best_d - (-best_d).max())
                w = (w / (w.sum() + 1e-12)) * q[j]
                np.add.at(phi, best_i, w)

    else:
        raise ValueError(
            f"Unsupported distance: {distance}. Use 'cosine', 'l2', or 'lpips'."
        )

    # ---- class aggregation (same pattern as your function) ----
    dataset, _ = create_dataset(dataset_name=dataset_name, train=True)

    # Handle artbench with artist grouping
    if dataset_name == "artbench":
        # Load the same CSV used for removal to ensure alignment
        import pandas as pd

        from src.constants import DATASET_DIR

        artist_csv = os.path.join(
            DATASET_DIR,
            "artbench-10-imagefolder-split/train/post_impressionism_artists.csv",
        )
        artist_df = pd.read_csv(artist_csv)
        # Use CSV row order (not alphabetical!) to match removal indices
        unique_artists = artist_df.iloc[:, 0].tolist()
        artist_to_index = {artist: idx for idx, artist in enumerate(unique_artists)}
        artists = np.array(dataset["artist"])
        label_indices = np.array([artist_to_index[artist] for artist in artists])
        num_labels = len(unique_artists)
    elif dataset_name == "fashion":
        # Load the same CSV used for removal to ensure alignment
        import pandas as pd

        from src.constants import DATASET_DIR

        brand_csv = os.path.join(DATASET_DIR, "fashion-product/top100_brands.csv")
        brand_df = pd.read_csv(brand_csv, header=0)
        brand_df = brand_df.dropna(subset=["brand"])
        # Use CSV row order (not alphabetical!) to match removal indices
        unique_brands = brand_df["brand"].tolist()
        brand_to_index = {brand: idx for idx, brand in enumerate(unique_brands)}
        brands = np.array(dataset["brand"])
        label_indices = np.array([brand_to_index[brand] for brand in brands])
        num_labels = len(unique_brands)
    else:
        # Standard numeric label grouping
        labels = np.array(dataset.targets)
        unique_values = sorted(set(labels))
        value_to_index = {val: idx for idx, val in enumerate(unique_values)}
        label_indices = np.array([value_to_index[label] for label in labels])
        num_labels = len(unique_values)

    result = np.zeros(num_labels, dtype=np.float32)
    for i in range(num_labels):
        mask = label_indices == i
        if aggregation == "max":
            result[i] = phi[mask].max() if np.any(mask) else 0.0
        elif aggregation == "mean":
            result[i] = phi[mask].mean() if np.any(mask) else 0.0
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")

    return result
