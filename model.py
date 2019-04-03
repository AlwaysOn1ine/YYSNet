import tensorflow as tf
import numpy as np
import os
from data import read_image

BATCH_SIZE = 20
CLASS_COUNT = 293
EPOCH = 2000

def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def conv_2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], padding='SAME')

def max_pooling2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def inference(img_path):
    image, label = read_image(img_path, 0)


if __name__ == "__main__":
    root, dirs, files = os.walk('./dataset/train_data/').__next__()
    files = [root + file for file in files]
    
    labels = np.arange(0, len(files))

    labels = tf.one_hot(labels, CLASS_COUNT)

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))

    dataset = dataset.map(read_image).shuffle(80)

    dataset1 = dataset.batch(batch_size=CLASS_COUNT)

    dataset = dataset.batch(batch_size=BATCH_SIZE)

    dataset = dataset.repeat()

    data_it = dataset.make_one_shot_iterator()

    next_data = data_it.get_next()

    # with tf.Session() as sess:
    #     for i in range(1000):
    #         image, label = sess.run(next_data)
    #         print(len(image), len(label), image.shape, np.min(image), np.min(label))

    x = tf.placeholder("float", shape=[None, 28, 28, 3], name="input")
    y_ = tf.placeholder("float", shape=[None, CLASS_COUNT])

    # first conv layer
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv_2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pooling2x2(h_conv1)

    # second conv layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pooling2x2(h_conv2)

    # fully connected
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout layer
    keep_prob = tf.placeholder("float")
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    # softmax layer
    W_fc2 = weight_variable([1024, CLASS_COUNT])
    b_fc2 = bias_variable([CLASS_COUNT])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

    cross_entropy = - tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCH):
            batch = sess.run(next_data)
            sess.run(train_step, feed_dict={x : batch[0], y_ : batch[1], keep_prob : 0.5})
            print("train accuracy:%f" % sess.run(accuracy, feed_dict={x : batch[0], y_ : batch[1], keep_prob : 1.0}))
        saver.save(sess, "./model/yysnet%d" % EPOCH)
        data_1 = dataset1.make_one_shot_iterator()
        next_data_1 = data_1.get_next()
        X_train, Y_train = sess.run(next_data_1)
        print(len(X_train), len(Y_train), X_train.shape)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x : X_train, y_ : Y_train, keep_prob : 1}))
        









