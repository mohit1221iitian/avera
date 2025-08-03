import os
import tensorflow as tf
from models.unet_model import build_unet
from scripts.data_loader import get_dataset
import matplotlib.pyplot as plt

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# === Hyperparameters ===
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 30  # Higher with early stopping
LR = 5e-5    # Reduced learning rate

# === Paths ===
train_img_dir = "processed/train/images"
train_mask_dir = "processed/train/masks"
val_img_dir = "processed/val/images"
val_mask_dir = "processed/val/masks"

# === Prepare datasets ===
train_ds = get_dataset(train_img_dir, train_mask_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=True)
val_ds = get_dataset(val_img_dir, val_mask_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE)

# === Build model ===
model = build_unet(input_shape=(*IMG_SIZE, 3), base_filters=16)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Callbacks ===
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("outputs/best_model.h5", save_best_only=True, monitor="val_loss"),
]

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Save final model ===
os.makedirs("outputs", exist_ok=True)
model.save("outputs/unet_model.h5")
print("✅ Model saved")

# === Plot training curves ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png")
plt.show()
