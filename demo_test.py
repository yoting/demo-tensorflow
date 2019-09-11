import tensorflow as tf
import numpy as np

#
# v1 = tf.Variable(tf.truncated_normal([2, 5], stddev=1))
# v2 = tf.Variable(tf.constant(0.1, shape=[50]), name='biases1')
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     print(v1)
#     print(sess.run(v1))
#
#     print(sess.run(v2))
# no_train = tf.Variable(tf.constant([1, 2], shape=(1, 2), name='const'), trainable=True)
# weights1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1), name='weight1')
biases1 = tf.Variable(tf.constant(0.1, shape=[2000]), name='biases1')
# # trainable_var = tf.trainable_variables()
#
# init = tf.global_variables_initializer()
#
# train_var = tf.GraphKeys.TRAINABLE_VARIABLES
# train_coll = tf.get_collection(train_var)

g = tf.get_default_graph()

tf.variable_scope("scope1")
scope2= tf.name_scope("scope2")

tf.GraphKeys.GLOBAL_VARIABLES


const = tf.constant(0, dtype=tf.int32, shape=[10, 1], name='const')
zzz = tf.reshape(const, shape=[2, 5], name='reshap_const')

print(const.op)
print(zzz.op.name)
print(zzz.dtype)
print(zzz.graph)
print(zzz.name)
print(zzz.device)
print(zzz.shape)

graph = tf.Tensor.graph
op = tf.Tensor.op

with tf.Session() as sess:
    # init.run()

    print(sess.run(const))
    print(sess.run(zzz))

    # print(sess.run(train_var))
    # print(sess.run(train_coll))
    # print(trainable_var)
    # print(sess.run(trainable_var))
