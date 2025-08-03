import matplotlib.pyplot as plt
from scripts.data_loader import get_dataset

# Config
IMG_SIZE = (128, 128)
BATCH_SIZE = 4
IMG_DIR = "processed/train/images"
MASK_DIR = "processed/train/masks"

# Load dataset without augmentation
ds = get_dataset(IMG_DIR, MASK_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, augment=False)

# Get one batch
for imgs, masks in ds.take(1):
    for i in range(len(imgs)):
        plt.subplot(2, len(imgs), i + 1)
        plt.imshow(imgs[i])
        plt.axis("off")
        plt.title("Image")

        plt.subplot(2, len(imgs), i + 1 + len(imgs))
        plt.imshow(masks[i, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.title("Mask")
    plt.tight_layout()
    plt.show()
