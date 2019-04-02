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

    return X, Y
    # labels = tf.one_hot()
    

if __name__ == "__main__":
    imageLists = []
    labels = []
    for i in range(20):
        imageLists.append(i)
        labels.append(19 - i)
    
    imageLists = tf.convert_to_tensor(imageLists, tf.int32)
    labels = tf.convert_to_tensor(labels, tf.int32)

    imageList, label = tf.train.slice_input_producer([imageLists, labels], shuffle=True)

    X, Y = tf.train.batch([imageList, label], batch_size=2, num_threads=4, capacity=16)


    try:
        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                coor = tf.train.Coordinator()
                tf.train.start_queue_runners(sess, coor)
                while not coor.should_stop():
                        i, l = sess.run([X, Y])
                        print(i)
                        print(l)
                        print("batch info")
    except TypeError:
            print("done")
    finally:
            coor.request_stop()



