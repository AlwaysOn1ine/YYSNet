import tensorflow as tf
import os

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 3

def read_image(train_path, label):
    image_path = tf.read_file(train_path)

    image_decoded = tf.image.decode_png(image_path)

    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label


def getDataset(train_path, labels, batch_size):
    root, dirs, files = os.walk(train_path).__next__()
    files = [root + file for file in files]

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(read_image).batch(batch_size=batch_size).repeat()
    return dataset
    

