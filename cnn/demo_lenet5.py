import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data', one_hot=True)

batch_size = 100  # 每一轮训练的大小
learning_rate = 0.01  # 初始化学习率
learning_rate_decay = 0.99  # 学习率衰减
max_steps = 30000  # 最大训练步数


def hidden_layer(input_tensor, regularizer, avg_class, resuse):
    # 创建第一个卷积层
    with tf.variable_scope('C1-conv', resuse=resuse):
        conv1_weights = tf.get_variable('weight', [5, 5, 1, 32],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 创建第一个池化层
    with tf.variable_scope('S2-max_pool', ):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 创建第二个卷积层
    with tf.variable_scope('C3-conv', resuse=resuse):
        conv2_weights = tf.get_variable('weight', [5, 5, 32, 64],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [32], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 创建第二个池化层
    with tf.variable_scope('S4-max_pool', ):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        shape = pool2.shape.as_list()
        nodes = shape[1] * shape[2] * shape[3]
        reshape = tf.reshape(pool2, [shape[0], nodes])

    # 创建第一个全连层
    with tf.variable_scope('L5_full', resuse=resuse):
        fc1_weights = tf.get_variable('weights', [nodes, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        else:
            fc1 = tf.nn.relu(tf.matmul(reshape, avg_class.average(fc1_weights)) + avg_class.average(fc1_biases))
        fc1 = tf.nn.dropout(fc1, 0.5)

    # 创建第二个全连接层
    with tf.variable_scope('L6_full', resuse=resuse):
        fc2_weights = tf.get_variable('weights', [512, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [10], initializer=tf.constant_initializer(0.1))
        if avg_class == None:
            fc2 = tf.nn.relu(tf.matmul(reshape, fc2_weights) + fc2_biases)
        else:
            fc2 = tf.nn.relu(tf.matmul(reshape, avg_class.average(fc2_weights)) + avg_class.average(fc2_biases))

    return fc2
