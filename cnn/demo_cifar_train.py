import tensorflow as tf
import numpy as np
import time
import math
import cnn.demo_cifar_read as cifar_read

max_steps = 4000
batch_size = 100
num_examples_for_eval = 100000
data_dir = '..\\data\\cifar-10'

record = cifar_read.CIFAR10Record()


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weights_loss)
    return var


# 读取数据
images_train, labels_train = record.input(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = record.input(data_dir=data_dir, batch_size=batch_size, distorted=None)

# 创建数据数据
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3], 'input_image')
y_ = tf.placeholder(tf.int32, [batch_size], 'output_label')

# 第一个卷积层，核大小为5*5，核深度为3，当前层深度（输出深度）为64
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, wl=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, wl=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 拉直数据
reshape = tf.reshape(pool2, [batch_size, -1])  # 将每条数据变为一维的，-1表示拉直数据
dim = reshape.get_shape()[1].value  # 获取每条数据的特征点数

# 第一个全连层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, wl=0.0004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_out1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
# 第二个全连层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, wl=0.0004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_out2 = tf.nn.relu(tf.matmul(fc_out1, weight2) + fc_bias2)
# 第三个全连层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=0.04, wl=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.matmul(fc_out2, weight3) + fc_bias3  # 未使用激活函数

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection('losses'))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
train_op = tf.train.AdadeltaOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(result, y_, 1)  # 获取top k的准确率

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 开启多线程 (获取数据)
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)

            print('step %d, loss=%.2f(%.1f examples/sec; %.3f sec/batch)' % (
                step, loss_value, examples_per_sec, sec_per_batch))

    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    print('accuracy=%.3f%%' % (true_count / total_sample_count) * 100)
