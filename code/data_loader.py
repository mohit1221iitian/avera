import tensorflow as tf
import os

def load_img_mask(img_path, mask_path, img_size):
    img_path = tf.cast(img_path, tf.string)
    mask_path = tf.cast(mask_path, tf.string)

    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Normalize to [0, 1]

    # Load and preprocess mask (already binary 0/1)
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size)
    mask = tf.cast(mask, tf.float32)  # Keep 0 and 1 as-is

    return img, mask

def augment_fn(img, mask):
    img = tf.image.random_flip_left_right(img)
    mask = tf.image.random_flip_left_right(mask)

    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)

    return img, mask

def get_dataset(image_dir, mask_dir, batch_size=4, img_size=(256, 256), augment=False):
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith(".jpg")
    ])
    mask_paths = sorted([
        os.path.join(mask_dir, f)
        for f in os.listdir(mask_dir) if f.endswith(".png")
    ])

    buffer_size = len(image_paths)

    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)
    mask_paths = tf.convert_to_tensor(mask_paths, dtype=tf.string)

    img_size_const = tf.constant(img_size, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(lambda x, y: load_img_mask(x, y, img_size_const),
                          num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
