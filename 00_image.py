import os
from PIL import Image
import sys

def analyze_imagenet1k(folder_path):
    total_images = 0
    corrupted_images = []
    total_size = 0  # in bytes

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is an image (you can add more formats if necessary)
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_images += 1
                total_size += os.path.getsize(file_path)

                # Check if the image is corrupt
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Only verifies if it can open, does not load image
                except (IOError, SyntaxError):
                    corrupted_images.append(file_path)
    
    # Convert total_size to MB for easier readability
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"Total images: {total_images}")
    print(f"Corrupted images: {len(corrupted_images)}")
    if corrupted_images:
        print("List of corrupted images:")
        for img in corrupted_images:
            print(img)
    print(f"Total size: {total_size_mb:.2f} MB")

# Specify the path to the imagenet1k folder
imagenet1k_path = "data/imagenet_val"
analyze_imagenet1k(imagenet1k_path)