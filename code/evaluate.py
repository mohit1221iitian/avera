import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scripts.data_loader import get_dataset

# Load model
model = tf.keras.models.load_model("outputs/best_model.h5", compile=False)

# Load full validation set
IMG_SIZE = (128, 128)
val_ds = get_dataset(
    "processed/val/images",
    "processed/val/masks",
    batch_size=1,
    img_size=IMG_SIZE,
    augment=False
)

# Metric accumulators
iou_scores = []
dice_scores = []
pixel_accuracies = []

def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union if union != 0 else 1.0

    dice = 2 * intersection / (y_true.sum() + y_pred.sum()) if (y_true.sum() + y_pred.sum()) != 0 else 1.0
    acc = (y_true == y_pred).sum() / y_true.size

    return iou, dice, acc

# Loop over all validation data
for imgs, masks in val_ds:  # No .take() — process full dataset
    preds = model.predict(imgs)
    preds = (preds > 0.5).astype(np.uint8)
    true_masks = masks.numpy().astype(np.uint8)

    for i in range(len(imgs)):
        iou, dice, acc = compute_metrics(true_masks[i, :, :, 0], preds[i, :, :, 0])
        iou_scores.append(iou)
        dice_scores.append(dice)
        pixel_accuracies.append(acc)

# Print global scores
print("\n==== Global Evaluation Metrics ====")
print(f"Mean IoU: {np.mean(iou_scores):.4f}")
print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
print(f"Mean Pixel Accuracy: {np.mean(pixel_accuracies):.4f}")
