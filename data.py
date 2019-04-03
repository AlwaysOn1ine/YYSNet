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
    

