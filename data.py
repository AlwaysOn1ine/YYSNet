import tensorflow as tf
import os

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHANNELS = 3

def read_image(train_path, label_path, batch_size):
    imagePaths = []
    labels = []
    label = 0

    imgPaths = os.walk('train_path').__next__()[3]
    for imgpath in imgPaths:
        if imgpath.endswith('.jpg') or imgpath.endswith('.png'):
            imagePaths.append(imgpath)
            labels.append(label)
            label = label + 1


    imagePaths = tf.convert_to_tensor(imagePaths, tf.string)
    labels = tf.convert_to_tensor(labels, tf.int32)

    imagePath, label = tf.train.slice_input_producer([imagePaths, labels], shuffle=True)

    image = tf.read_file(imagePath)
    image = tf.image.decode_png(image, channels=IMAGE_CHANNELS)

    image = tf.image.resize_images(image, size=[IMAGE_WIDTH, IMAGE_HEIGHT])
    image = image * 1.0 / 127.5 - 1

    X, Y = tf.train.batch([image, label], batch_size=batch_size, num_threads=4, capacity=batch_size * 8)

    print(Y)
    # labels = tf.one_hot()
    
