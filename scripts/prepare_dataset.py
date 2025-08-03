import os
import random
from PIL import Image

# === Paths ===
DATASET_DIR = r"D:\avera\data\road_dataset"
PROCESSED_DIR = r"D:\avera\processed"

TRAIN_IMG_DIR = os.path.join(PROCESSED_DIR, "train/images")
TRAIN_MASK_DIR = os.path.join(PROCESSED_DIR, "train/masks")
VAL_IMG_DIR = os.path.join(PROCESSED_DIR, "val/images")
VAL_MASK_DIR = os.path.join(PROCESSED_DIR, "val/masks")

# Create output folders
for path in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR]:
    os.makedirs(path, exist_ok=True)

# Collect and pair images with masks
image_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".jpg")]
mask_files = {f.replace("_mask.png", ""): f for f in os.listdir(DATASET_DIR) if f.endswith("_mask.png")}

valid_pairs = []
for img_file in image_files:
    base = os.path.splitext(img_file)[0]
    if base in mask_files:
        valid_pairs.append((img_file, mask_files[base]))
    else:
        print(f"⚠️ Missing mask for: {img_file}")

print(f"✅ Found {len(valid_pairs)} valid image-mask pairs.")

# Shuffle and split
random.seed(42)
random.shuffle(valid_pairs)
split_idx = int(0.8 * len(valid_pairs))
train_pairs = valid_pairs[:split_idx]
val_pairs = valid_pairs[split_idx:]

# Resize and save
def resize_and_save(img_path, mask_path, dst_img_path, dst_mask_path, size=(256, 256)):
    img = Image.open(img_path).resize(size, Image.BILINEAR)
    mask = Image.open(mask_path).resize(size, Image.NEAREST)  # Avoid interpolation in masks
    img.save(dst_img_path)
    mask.save(dst_mask_path)

# Process function
def process_pairs(pairs, img_dir, mask_dir):
    for img_file, mask_file in pairs:
        resize_and_save(
            os.path.join(DATASET_DIR, img_file),
            os.path.join(DATASET_DIR, mask_file),
            os.path.join(img_dir, img_file),
            os.path.join(mask_dir, mask_file)
        )

process_pairs(train_pairs, TRAIN_IMG_DIR, TRAIN_MASK_DIR)
process_pairs(val_pairs, VAL_IMG_DIR, VAL_MASK_DIR)

print("✅ Dataset processed and saved.")
