"""Dataset related functions and classes"""
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

import src.constants as constants
from datasets import load_dataset, load_from_disk


def _subset_sequence(seq, indices):
    """Return seq subset by indices for list/np.ndarray/torch.Tensor."""
    if isinstance(seq, np.ndarray):
        return seq[indices]
    if torch.is_tensor(seq):
        return seq[indices]
    # Fallback: treat as a Python sequence
    return [seq[i] for i in indices]


def _apply_subset_inplace(dataset, indices):
    """Subset dataset.data / dataset.targets (and embeddings if present) in-place."""
    # Keep ordering stable
    indices = list(indices)
    dataset.data = _subset_sequence(dataset.data, indices)
    dataset.targets = [dataset.targets[i] for i in indices]
    if hasattr(dataset, "embeddings"):
        dataset.embeddings = dataset.embeddings[indices]
    # If wrapped with WithEmbeddings, also subset the base for consistency
    if hasattr(dataset, "base"):  # i.e., WithEmbeddings
        base = dataset.base
        base.data = _subset_sequence(base.data, indices)
        base.targets = [base.targets[i] for i in indices]
    return dataset


class TensorDataset(Dataset):
    """Wraps tensor data for easy dataset operations."""

    def __init__(self, data, transform=None, label=None):
        """Initializes dataset with data tensor."""
        self.data = data
        self.transform = transform
        self.label = label

        if self.transform is not None:
            self.data = self.transform(self.data)

    def __len__(self):
        """Returns dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves sample at index `idx`."""
        if self.label is not None:
            return self.data[idx], self.label[idx]
        return self.data[idx]


class ImageDataset(Dataset):
    """Loads and transforms images from a directory."""

    def __init__(self, img_dir, transform=transforms.PILToTensor(), max_size=None):
        """Initializes dataset with image directory and transform."""
        self.img_dir = img_dir
        self.img_list = [
            img
            for img in os.listdir(img_dir)
            if img.split(".")[-1] in {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
        ]
        if max_size is not None:
            self.img_list = self.img_list[:max_size]
        self.transform = transform

    def __getitem__(self, idx):
        """Returns transformed image at index `idx`."""
        with Image.open(os.path.join(self.img_dir, self.img_list[idx])) as im:
            return self.transform(im), -1

    def __len__(self):
        """Returns total number of images."""
        return len(self.img_list)


class CIFAR20(CIFAR100):
    """
    Dataloader for CIFAR-100 dataset to include only animal classes.

    Return_
        3x32x32 CIFAR-100 images, and its corresponding label
        (filtered to only animals, 35 classes.)
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Update this list based on CIFAR-100 animal class indices
        self.classes = [
            40,
            41,
            42,
            43,
            44,  # Large carnivores
            55,
            56,
            57,
            58,
            59,  # Large omnivores and herbivores
            60,
            61,
            62,
            63,
            64,  # Medium mammals
            80,
            81,
            82,
            83,
            84,  # Small mammals
        ]
        # Filter the dataset

        filtered_indices = [
            i for i, target in enumerate(self.targets) if target in self.classes
        ]
        self.data = self.data[filtered_indices]

        # reset class label

        self.targets = [
            self.classes.index(target)
            for i, target in enumerate(self.targets)
            if target in self.classes
        ]


class FashionDatasetWrapper:
    """Dataset wrapper for fashion dataset from HuggingFace datasets."""

    def __init__(
        self,
        hf_dataset,
        size=1024,
        center_crop=False,
        custom_instance_prompts=True,
        pad_to_square=False,
    ):
        self.hf_dataset = hf_dataset
        self.custom_instance_prompts = custom_instance_prompts
        self.size = size
        self.center_crop = center_crop
        self.pad_to_square = pad_to_square

        # Image transforms
        if pad_to_square:
            # Resize so larger edge becomes size, then pad to square
            def resize_and_pad(img):
                w, h = img.size
                scale = size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = transforms.functional.resize(
                    img,
                    (new_h, new_w),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
                # Pad to square
                pad_w = size - new_w
                pad_h = size - new_h
                padding = (
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - pad_w // 2,
                    pad_h - pad_h // 2,
                )
                return transforms.functional.pad(
                    img, padding, fill=0, padding_mode="constant"
                )

            resize_transform = transforms.Lambda(resize_and_pad)
            crop_transform = transforms.Lambda(lambda x: x)  # No-op, already square
        else:
            # Default: resize smaller edge to size, then crop
            resize_transform = transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BILINEAR
            )
            crop_transform = (
                transforms.CenterCrop(size)
                if center_crop
                else transforms.RandomCrop(size)
            )

        self.image_transforms = transforms.Compose(
            [
                resize_transform,
                crop_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        """Length of dataset"""
        return len(self.hf_dataset)

    def __getitem__(self, index):
        """Return item"""
        example = self.hf_dataset[int(index)]
        instance_image = example["image"]
        instance_prompt = example["prompt"]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        instance_image = self.image_transforms(instance_image)

        return {
            "instance_images": instance_image,
            "instance_prompt": instance_prompt,
            "filename": example["filename"],
        }


class WithEmbeddings(Dataset):
    """
    Dataset class for embeddings.
    Preserves base dataset behavior, exposes .data/.targets, and carries embeddings.
    """

    def __init__(self, base, embeddings: torch.Tensor):
        self.base = base
        self._embeddings = embeddings  # torch.float32 [N, D]

    # forward .data/.targets so your removal fn works unchanged
    @property
    def data(self):
        return self.base.data

    @data.setter
    def data(self, v):
        self.base.data = v

    @property
    def targets(self):
        return self.base.targets

    @targets.setter
    def targets(self, v):
        self.base.targets = v

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, v):
        self._embeddings = v

    def __len__(self):
        """Return the length of the base dataset."""
        return len(self.base)

    def __getitem__(self, idx):
        """Get item with aligned embedding."""
        img, target = self.base[idx]  # keep transforms
        emb = self._embeddings[idx]  # aligned embedding
        return img, target, emb


def create_dataset(
    dataset_name: str,
    train: bool = True,
    dataset_dir: str = constants.DATASET_DIR,
    removal_dist: str = "all",
    removal_idx: int = 0,
    datamodel_alpha: float = 0.5,
    discrete_label: bool = True,
    percentile_indices: list = None,
    percentile_mode: str = "remove",
) -> torch.utils.data.Dataset:
    """Create a PyTorch Dataset corresponding to a dataset."""

    if dataset_name == "cifar20":
        preprocess = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Normalize to [0,1].
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1].
            ]
        )
        root_dir = os.path.join(dataset_dir, "cifar100")

        dataset = CIFAR20(
            root=root_dir, train=train, download=True, transform=preprocess
        )
        class_num = len(np.unique(dataset.targets))

        if not discrete_label:
            model_name = "dinov2_vitl14"
            path = os.path.join(
                dataset_dir,
                "cifar20",
                model_name,
                f"embeddings_{model_name}_train.npy",
            )
            emb = np.load(path).astype(np.float32)
            dataset = WithEmbeddings(dataset, torch.from_numpy(emb))

        if removal_dist == "all":
            # No removal
            pass
        elif removal_dist == "loo":
            # Leave-one-out removal
            dataset = remove_by_loo(dataset, removal_idx)
        elif removal_dist == "datamodel":
            dataset = remove_by_datamodel(
                dataset,
                removal_idx,
                alpha=datamodel_alpha,
                discrete_label=discrete_label,
            )
        elif removal_dist == "shapley":
            dataset = remove_by_shapley(
                dataset, removal_idx, discrete_label=discrete_label
            )
        elif removal_dist == "shapley_uniform":
            dataset = remove_by_shapley_uniform(
                dataset, removal_idx, discrete_label=discrete_label
            )
        elif removal_dist == "percentile":
            if percentile_indices is None:
                raise ValueError(
                    "percentile_indices must be provided for percentile removal"
                )
            dataset = remove_by_percentile_cifar20(
                dataset, percentile_indices, mode=percentile_mode
            )
        else:
            raise ValueError(f"removal_dist={removal_dist} does not exists")

    elif dataset_name == "artbench":
        # Create artbench -post-impressionism

        split = "train" if train else "test"
        data_dir = os.path.join(dataset_dir, "artbench-10-imagefolder-split", split)
        data_files = {}
        data_files[split] = os.path.join(data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
        )

        cls_idx = np.where(np.array(dataset[split]["style"]) == "post_impressionism")[0]
        dataset = dataset[split].select(cls_idx)
        assert dataset.num_rows == 5000

        # obtain artists num
        removal_unit_file = os.path.join(data_dir, "post_impressionism_artists.csv")
        removal_unit_df = pd.read_csv(removal_unit_file)
        class_num = len(removal_unit_df)

        # removal
        if removal_dist == "all":
            # No removal
            remaining_idx = [i for i in range(len(set(dataset["artist"])))]
            pass
        elif removal_dist == "datamodel":
            remaining_idx, removed_idx = remove_index_by_datamodel(
                removal_unit_df, datamodel_alpha, removal_idx
            )
        elif removal_dist == "shapley":
            remaining_idx, removed_idx = remove_index_by_shapley(
                removal_unit_df, removal_idx
            )
        elif removal_dist == "shapley_uniform":
            remaining_idx, removed_idx = remove_index_by_shapley_uniform(
                removal_unit_df, removal_idx
            )
        elif removal_dist == "loo":
            remaining_idx, removed_idx = remove_data_by_loo(
                removal_unit_df, removal_idx
            )
        elif removal_dist == "percentile":
            if percentile_indices is None:
                raise ValueError(
                    "percentile_indices must be provided for percentile removal"
                )
            remaining_idx, removed_idx = remove_by_percentile(
                removal_unit_df, percentile_indices, mode=percentile_mode
            )
        else:
            raise ValueError(f"removal_dist={removal_dist} does not exists")
        kept_units = removal_unit_df.iloc[remaining_idx, 0].tolist()
        train_units = np.array(dataset["artist"])
        dataset = dataset.select(np.where(np.isin(train_units, kept_units))[0])
        assert set(dataset["artist"]) == set(kept_units)

    elif dataset_name == "fashion":
        # Load from single flat directory containing only image files
        split = "train" if train else "test"

        # Check if cached processed dataset exists
        cache_dir = os.path.join(dataset_dir, "fashion-product", f"cache_{split}")

        if os.path.exists(cache_dir):
            # Load from cache
            dataset = load_from_disk(cache_dir)
        else:
            # Load and process dataset
            data_dir = os.path.join(dataset_dir, "fashion-product", "top100")
            dataset = load_dataset("imagefolder", data_dir=data_dir, split="train")

            # Load metadata to get brand and display_name (prompt) information
            metadata = pd.read_csv(
                os.path.join(dataset_dir, "fashion-product/final_top100_subsampled.csv")
            )
            # Remove rows where brand or display name is None/NaN
            metadata = metadata.dropna(subset=["brand", "display name"])

            # Create mappings from image filename to brand and display_name
            img_to_brand = {
                str(img): str(brand)
                for img, brand in zip(metadata["image"], metadata["brand"])
            }
            img_to_display_name = {
                str(img): str(name)
                for img, name in zip(metadata["image"], metadata["display name"])
            }
            img_to_category = {
                str(img): str(name)
                for img, name in zip(metadata["image"], metadata["category"])
            }

            # Add brand and display_name columns to dataset based on image filenames
            def add_metadata(example):
                # Extract just the filename from the image path
                filename = (
                    os.path.basename(example["image"].filename)
                    if hasattr(example["image"], "filename")
                    else os.path.basename(str(example["image"]))
                )
                return {
                    "filename": filename,
                    "brand": img_to_brand.get(filename, "unknown"),
                    "prompt": img_to_display_name.get(filename, ""),
                    "category": img_to_category.get(filename, ""),
                }

            dataset = dataset.map(add_metadata)

            # Filter out images that don't have valid metadata
            #  (brand or prompt is missing)
            dataset = dataset.filter(
                lambda x: x["brand"] != "unknown" and x["prompt"] != ""
            )

            # Save to cache for future use
            dataset.save_to_disk(cache_dir)

        # Load removal unit file (company brands)
        removal_unit_file = os.path.join(
            dataset_dir, "fashion-product", "top100_brands.csv"
        )
        removal_unit_df = pd.read_csv(removal_unit_file, header=0)
        removal_unit_df = removal_unit_df.dropna(subset=["brand"])

        class_num = len(removal_unit_df)

        # removal
        if removal_dist == "all":
            # No removal - keep all brands
            remaining_idx = [i for i in range(len(removal_unit_df))]
        elif removal_dist == "datamodel":
            remaining_idx, removed_idx = remove_index_by_datamodel(
                removal_unit_df, datamodel_alpha, removal_idx
            )
        elif removal_dist == "shapley":
            remaining_idx, removed_idx = remove_index_by_shapley(
                removal_unit_df, removal_idx
            )
        elif removal_dist == "shapley_uniform":
            remaining_idx, removed_idx = remove_index_by_shapley_uniform(
                removal_unit_df, removal_idx
            )
        elif removal_dist == "loo":
            remaining_idx, removed_idx = remove_data_by_loo(
                removal_unit_df, removal_idx
            )
        elif removal_dist == "percentile":
            if percentile_indices is None:
                raise ValueError(
                    "percentile_indices must be provided for percentile removal"
                )
            remaining_idx, removed_idx = remove_by_percentile(
                removal_unit_df, percentile_indices, mode=percentile_mode
            )
        else:
            raise ValueError(f"removal_dist={removal_dist} does not exists")

        # Get the brands to keep based on remaining indices
        kept_brands = removal_unit_df.iloc[remaining_idx, 0].tolist()
        dataset_brands = np.array(dataset["brand"])
        # Filter dataset to only include kept brands
        dataset = dataset.select(np.where(np.isin(dataset_brands, kept_brands))[0])
        assert set(dataset["brand"]) == set(kept_brands)
        dataset.remaining_idx = remaining_idx
    else:
        raise ValueError(
            f"dataset_name={dataset_name} should be one of "
            "['cifar20', 'mnist', 'celeba', 'artbench', 'fashion']"
        )

    return dataset, class_num


def remove_by_loo(dataset, removed_class):
    """Remove data based on leave-one-out class."""
    if removed_class < 0 or removed_class >= len(dataset.targets):
        raise ValueError("removed_class is out of bounds.")
    # Filter the dataset
    remaining_indices = [
        idx for idx, target in enumerate(dataset.targets) if target != removed_class
    ]
    dataset.data = dataset.data[remaining_indices]
    dataset.targets = [dataset.targets[i] for i in remaining_indices]

    return dataset


def remove_by_percentile_cifar20(dataset, percentile_indices, mode="remove"):
    """Remove or keep classes from cifar20 dataset based on percentile indices."""
    percentile_indices = set(percentile_indices)

    if mode == "remove":
        # Keep samples whose class is NOT in percentile_indices
        remaining_indices = [
            idx
            for idx, target in enumerate(dataset.targets)
            if target not in percentile_indices
        ]
    elif mode == "keep":
        # Keep samples whose class IS in percentile_indices
        remaining_indices = [
            idx
            for idx, target in enumerate(dataset.targets)
            if target in percentile_indices
        ]
    else:
        raise ValueError(f"mode must be 'remove' or 'keep', got {mode}")

    dataset.data = dataset.data[remaining_indices]
    dataset.targets = [dataset.targets[i] for i in remaining_indices]

    if hasattr(dataset, "embeddings"):
        dataset.embeddings = dataset.embeddings[remaining_indices]

    return dataset


def remove_data_by_loo(
    dataset: torch.utils.data.Dataset, loo_idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split dataset indices into leave-one-out remaining and removed indices."""
    dataset_size = len(dataset)
    removed_idx = np.array([loo_idx])
    remaining_idx = np.array([i for i in range(dataset_size) if i != loo_idx])
    return remaining_idx, removed_idx


def remove_by_percentile(
    removal_unit_df: pd.DataFrame, percentile_indices: list, mode: str = "remove"
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove or keep specific percentile players from dataset."""
    dataset_size = len(removal_unit_df)
    all_indices = np.arange(dataset_size)
    percentile_indices = np.array(percentile_indices)

    if mode == "remove":
        remaining_idx = np.array(
            [i for i in all_indices if i not in percentile_indices]
        )
        removed_idx = percentile_indices
    elif mode == "keep":
        remaining_idx = percentile_indices
        removed_idx = np.array([i for i in all_indices if i not in percentile_indices])
    else:
        raise ValueError(f"mode must be 'remove' or 'keep', got {mode}")

    return remaining_idx, removed_idx


def remove_by_datamodel(dataset, removal_idx=0, alpha=0.5, discrete_label=True):
    """Sample dataset based on a datamodel 'alpha'."""

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    rng = np.random.RandomState(removal_idx)

    if discrete_label:
        # classes as players
        players = np.unique(dataset.targets).tolist()
    else:
        # samples as players
        players = list(range(len(dataset.data)))

    rng.shuffle(players)  # important;in-place shuffle
    num_players = len(players)
    remaining_size = int(alpha * num_players)
    remaining_classes = players[:remaining_size]

    if discrete_label:
        remaining_indices = [
            idx
            for idx, target in enumerate(dataset.targets)
            if target in remaining_classes
        ]
    else:
        remaining_indices = remaining_classes

    dataset.data = dataset.data[remaining_indices]
    dataset.targets = [dataset.targets[i] for i in remaining_indices]

    if hasattr(dataset, "embeddings"):
        dataset.embeddings = dataset.embeddings[remaining_indices]

    return dataset


def remove_index_by_datamodel(
    dataset: torch.utils.data.Dataset,
    alpha: float = 0.5,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a PyTorch Dataset into indices with the remaining and removed data, where
    the remaining dataset is an `alpha` proportion of the full dataset.

    Args:
    ----
        dataset: The PyTorch Dataset to split.
        alpha: The proportion of the full dataset to keep in the remaining set.
        seed: Random seed for sampling which data points are selected to keep.

    Returns
    -------
        A numpy array with the remaining indices, and another numpy array with the
        indices corresponding to the removed data.
    """
    rng = np.random.RandomState(seed)

    dataset_size = len(dataset)
    all_idx = np.arange(dataset_size)

    num_selected = int(alpha * dataset_size)
    rng.shuffle(all_idx)  # Shuffle in place.

    remaining_idx = all_idx[:num_selected]
    removed_idx = all_idx[num_selected:]

    return remaining_idx, removed_idx


def remove_by_shapley(dataset, removal_idx=0, discrete_label=True):
    """Sample dataset based on a shapley kernel."""

    rng = np.random.RandomState(removal_idx)

    if discrete_label:
        # classes as players
        players = np.unique(dataset.targets).tolist()
    else:
        # samples as players
        players = list(range(len(dataset.data)))

    num_players = len(players)
    possible_remaining_sizes = np.arange(1, num_players)

    # First sample the remaining set size.
    # This corresponds to the term: (n - 1) / (|S| * (n - |S|)).

    remaining_size_probs = (num_players - 1) / (
        possible_remaining_sizes * (num_players - possible_remaining_sizes)
    )
    remaining_size_probs /= remaining_size_probs.sum()
    remaining_size = rng.choice(
        possible_remaining_sizes, size=1, p=remaining_size_probs
    )[0]
    # Now sample the actual remaining set.

    rng.shuffle(players)

    remaining_classes = players[:remaining_size]

    if discrete_label:
        remaining_indices = [
            idx
            for idx, target in enumerate(dataset.targets)
            if target in remaining_classes
        ]
    else:
        remaining_indices = remaining_classes

    dataset.data = dataset.data[remaining_indices]
    dataset.targets = [dataset.targets[i] for i in remaining_indices]

    if hasattr(dataset, "embeddings"):
        dataset.embeddings = dataset.embeddings[remaining_indices]

    return dataset


def remove_index_by_shapley(
    dataset: torch.utils.data.Dataset, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a PyTorch Dataset into indices with the remaining and removed data, where
    the remaining dataset is drawn from the Shapley kernel distribution, which has the
    probability mass function: p(S) = (n - 1) / (|S| * (n - |S|) * (n choose |S|)).

    Reference: https://captum.ai/api/kernel_shap.html#captum.attr.KernelShap.
    kernel_shap_perturb_generator.

    Args:
    ----
        dataset: The PyTorch Dataset to split.
        seed: Random seed for sampling which data points are selected to keep.

    Returns
    -------
        A numpy array with the remaining indices, and another numpy array with the
        indices corresponding to the removed data.
    """
    rng = np.random.RandomState(seed)

    dataset_size = len(dataset)

    # First sample the remaining set size.
    # This corresponds to the term: (n - 1) / (|S| * (n - |S|)).
    possible_remaining_sizes = np.arange(1, dataset_size)
    remaining_size_probs = (dataset_size - 1) / (
        possible_remaining_sizes * (dataset_size - possible_remaining_sizes)
    )
    remaining_size_probs /= remaining_size_probs.sum()
    remaining_size = rng.choice(
        possible_remaining_sizes, size=1, p=remaining_size_probs
    )[0]

    # Then sample uniformly given the remaining set size.
    # This corresponds to the term: 1 / (n choose |S|).
    all_idx = np.arange(dataset_size)
    rng.shuffle(all_idx)  # Shuffle in place.
    remaining_idx = all_idx[:remaining_size]
    removed_idx = all_idx[remaining_size:]

    return remaining_idx, removed_idx


def remove_index_by_shapley_uniform(
    dataset: torch.utils.data.Dataset, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a PyTorch Dataset into indices with the remaining and removed data, where
    the remaining dataset is drawn from a uniform distribution over all possible
    subsets (including empty set and full set).

    Args:
    ----
        dataset: The PyTorch Dataset to split.
        seed: Random seed for sampling which data points are selected to keep.

    Returns
    -------
        A numpy array with the remaining indices, and another numpy array with the
        indices corresponding to the removed data.
    """
    rng = np.random.RandomState(seed)

    dataset_size = len(dataset)
    all_idx = np.arange(dataset_size)

    # Sample remaining set size uniformly from 0 to n.
    remaining_size = rng.randint(0, dataset_size + 1)

    rng.shuffle(all_idx)  # Shuffle in place.

    remaining_idx = all_idx[:remaining_size]
    removed_idx = all_idx[remaining_size:]

    return remaining_idx, removed_idx


def remove_by_shapley_uniform(dataset, removal_idx=0, discrete_label=True):
    """Sample a coalition uniformly over sizes; include 0..n possibility."""
    rng = np.random.default_rng(removal_idx)

    # define the player universe
    players = (
        np.unique(dataset.targets).tolist()
        if discrete_label
        else list(range(len(dataset.data)))
    )
    n = len(players)

    # sample size uniformly from 0..n
    k = rng.integers(0, n + 1)

    # choose subset uniformly
    chosen = set(rng.choice(players, size=k, replace=False).tolist())

    # build indices to keep
    if discrete_label:
        keep = [i for i, y in enumerate(dataset.targets) if y in chosen]
    else:
        keep = [i for i in chosen]

    # mutate dataset to the sampled coalition (if you really want in-place)
    dataset.data = dataset.data[keep]
    dataset.targets = [dataset.targets[i] for i in keep]
    if hasattr(dataset, "embeddings"):
        dataset.embeddings = dataset.embeddings[keep]

    return dataset
