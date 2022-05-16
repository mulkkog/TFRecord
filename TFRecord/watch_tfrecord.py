import os
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt
from load_tfrecord import read_dataset

# tf.random.set_seed(1234)

BATCH_SIZE = 64
IMG_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/train/'
TFR_DIR = '/home/jang/Disk_1TB/Dataset/ImageNet2012/TFRecords_split/train/'
NUM_EXAMPLES = len(glob(IMG_DIR + '/*/*'))

# Load the data
FILENAMES = tf.io.gfile.glob(f"{TFR_DIR}*.tfrecord")

print("TFRecord Files:", len(FILENAMES))

# Read Data
dataset = read_dataset(FILENAMES, BATCH_SIZE)

image_batch, label_batch = next(iter(dataset))

classes = os.listdir(IMG_DIR)
classes.sort()


# Show Images and Labels
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(25):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n] / 255.0)
        if label_batch[n]:
            label = label_batch[n]
            label = classes[label]
            plt.title(label)
        plt.axis("off")
    plt.show()


show_batch(image_batch.numpy(), label_batch.numpy())
