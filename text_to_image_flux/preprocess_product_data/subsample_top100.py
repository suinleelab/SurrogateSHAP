"""Preprocessing fashion-product dataset"""
import os
import shutil

import pandas as pd
from tqdm import tqdm

# Read the original CSV
df = pd.read_csv("path_to_fashion-product/final.csv")

# Get top 100 brands by product count
top_100_brands = df["brand"].value_counts().head(100).index.tolist()

# Filter dataframe to only include top 100 brands
df_top_100 = df[df["brand"].isin(top_100_brands)]

# Subsample 1/4 (25%) of products from each brand
df_subsampled = df_top_100.groupby("brand", group_keys=False).apply(
    lambda x: x.sample(frac=0.25, random_state=42)
)

# Save to new CSV
df_subsampled.to_csv("path_to_fashion-product/final_top100_subsampled.csv", index=False)

# Save top 100 brand names to CSV
pd.DataFrame({"brand": top_100_brands}).to_csv(
    "path_to_fashion-product/top100_brands.csv", index=False
)

# Copy images to new folder
top100_folder = "path_to_fashion-product/top100"
os.makedirs(top100_folder, exist_ok=True)

print(f"\nCopying {len(df_subsampled)} images to {top100_folder}...")
for idx, row in tqdm(df_subsampled.iterrows(), total=len(df_subsampled)):
    src_path = os.path.join("path_to_fashion-product/train", row["image"])
    if os.path.exists(src_path):
        dst_path = os.path.join(top100_folder, row["image"])
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)

# Print statistics
print(f"Original dataset: {len(df)} products")
print(f"Top 100 brands total products: {len(df_top_100)} products")
print(f"Subsampled (1/4): {len(df_subsampled)} products")
print(f"\nNumber of unique brands: {df_subsampled['brand'].nunique()}")
print("\nProduct count per brand in subsample:")
print(df_subsampled["brand"].value_counts().sort_values(ascending=False).head(20))
