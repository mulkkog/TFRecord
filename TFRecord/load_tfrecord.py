import numpy as np
import tensorflow as tf


# Get Data
def _parse_image_function(example):
    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    features = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(features['image'], channels=3)
    image = tf.image.resize(image, size=[224, 224])

    # mean = np.array([103.939, 116.779, 123.68][::-1], dtype='float32')  # R, G, B
    # image -= mean
    # image = tf.image.random_crop(image, size=[224, 224, 3])
    # image = tf.image.flip_left_right(image)
    # image = tf.image.adjust_saturation(image, 0.5)

    label = tf.cast(features['label'], tf.int32)

    return image, label


def read_dataset(filename, batch_size):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1028)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset
