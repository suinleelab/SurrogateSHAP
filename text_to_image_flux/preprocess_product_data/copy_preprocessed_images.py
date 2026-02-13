"""
Copy preprocessed images to a separate folder
==============================================
This script reads the preprocessed CSV and copies only the images
that remain after preprocessing to a new 'preprocessed' folder.
"""

import csv
import os
import shutil
from pathlib import Path

from src.constants import DATASET_DIR

# Configuration
CSV_FILE = os.path.join(DATASET_DIR, "fashion-product", "data_preprocessed.csv")
SOURCE_DIR = os.path.join(DATASET_DIR, "fashion-product", "data")
DEST_DIR = os.path.join(DATASET_DIR, "fashion-product", "no_model_stats")


def main():
    print("=" * 80)
    print("Copying Preprocessed Images")
    print("=" * 80)

    # Create destination directory if it doesn't exist
    Path(DEST_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\nCreated directory: {DEST_DIR}")

    # Read image filenames from preprocessed CSV
    image_files = set()
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_files.add(row["image"])

    print(f"\nTotal images in preprocessed dataset: {len(image_files)}")

    # Copy images
    copied = 0
    not_found = 0

    print("\nCopying images...")
    for i, image_file in enumerate(image_files, 1):
        source_path = os.path.join(SOURCE_DIR, image_file)
        dest_path = os.path.join(DEST_DIR, image_file)

        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            copied += 1
            if copied % 1000 == 0:
                print(f"  Copied {copied}/{len(image_files)} images...")
        else:
            not_found += 1
            print(f"  Warning: {image_file} not found in source directory")

    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"Total images to copy:    {len(image_files):>10,}")
    print(f"Successfully copied:     {copied:>10,}")
    print(f"Not found:               {not_found:>10,}")
    print("-" * 80)
    print(f"\nImages copied to: {DEST_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
