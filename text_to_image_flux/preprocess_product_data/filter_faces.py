"""
Filter out products with human faces in images
===============================================
This script uses OpenCV's Haar Cascade face detection to identify
and remove products that contain human faces from the dataset.
Creates a final CSV file without copying or linking images.
"""

import csv
import os
import shutil

import cv2

# Configuration
INPUT_CSV = "path_to_fashion-product/data_preprocessed.csv"
OUTPUT_CSV = "path_to_fashion-product/final.csv"
IMAGE_DIR = "path_to_fashion-product/no_model_stats"
OUTPUT_IMAGE_DIR = "path_to_fashion-product/final_images"


def detect_faces(image_path):
    """
    Detect faces in an image using OpenCV Haar Cascade.
    Returns True if faces are detected, False otherwise.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            print(f"  Warning: Could not read {image_path}")
            return False

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Load the face detection classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Detect faces
        # Parameters: scaleFactor, minNeighbors, minSize
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        return len(faces) > 0

    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return False


def main():
    """Main funciton to filter out human faces."""
    print("=" * 80)
    print("Filtering Products with Human Faces")
    print("=" * 80)

    # Create output image directory
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    print(f"\nOutput image directory: {OUTPUT_IMAGE_DIR}")

    # Statistics
    total_products = 0
    products_with_faces = 0
    products_without_faces = 0
    images_not_found = 0
    images_copied = 0

    # Read input CSV and filter
    print("\nScanning images for faces...")

    with open(INPUT_CSV, "r", encoding="utf-8") as infile, open(
        OUTPUT_CSV, "w", encoding="utf-8", newline=""
    ) as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total_products += 1
            image_file = row["image"]
            image_path = os.path.join(IMAGE_DIR, image_file)

            # Progress update
            if total_products % 1000 == 0:
                print(
                    f"  Processed {total_products} products... "
                    f"(Found {products_with_faces} with faces, "
                    f"{products_without_faces} without)"
                )

            # Check if image exists
            if not os.path.exists(image_path):
                images_not_found += 1
                continue

            # Detect faces
            has_faces = detect_faces(image_path)

            if has_faces:
                products_with_faces += 1
            else:
                products_without_faces += 1
                writer.writerow(row)

                # Copy image to output directory
                try:
                    output_path = os.path.join(OUTPUT_IMAGE_DIR, image_file)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    shutil.copy2(image_path, output_path)
                    images_copied += 1
                except Exception as e:
                    print(f"  Error copying {image_file}: {e}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"Total products scanned:          {total_products:>10,}")
    print(f"Products with faces (excluded):  {products_with_faces:>10,}")
    print(f"Products without faces:          {products_without_faces:>10,}")
    print(f"Images not found:                {images_not_found:>10,}")
    print(f"Images copied successfully:      {images_copied:>10,}")
    print("-" * 80)
    print(f"Percentage with faces: {products_with_faces/total_products*100:>9.2f}%")
    print(
        f"Percentage without faces:{products_without_faces/total_products*100:>9.2f}%"
    )
    print("-" * 80)
    print(f"\nFiltered dataset saved to: {OUTPUT_CSV}")
    print(f"Filtered images copied to: {OUTPUT_IMAGE_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
