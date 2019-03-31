import tensorflow as tf
import numpy as np
from data import read_image

class_count = 100

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

if __name__ == "__main__":
    batch = [1, 1]

    x = tf.placeholder("float", shape=[None, 784], name="input")
    y_ = 1
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # first conv layer
    W_conv1 = weight_variable([3, 3, 3, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pooling2x2(h_conv1)

    # second conv layer
    W_conv2 = weight_variable([3, 3, 32, 64])
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
    W_fc2 = weight_variable([1024, class_count])
    b_fc2 = bias_variable([class_count])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

    cross_entropy = - tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 2))
    accuracy = tf.reduce_sum(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            print("batch %d trained" % i)
            sess.run(train_step, feed_dict={x : batch[0], y_ : batch[1], keep_prob : 0.5})
        print(sess.run(accuracy, feed_dict={x : batch[0], y_ : batch[1], keep_prob : 1.0}))

        









