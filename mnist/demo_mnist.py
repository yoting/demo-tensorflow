import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

print(mnist.train.images[0])
print(mnist.train.labels[0])

batch_size = 100  # 每一轮训练的大小
learning_rate = 0.8  # 初始化学习率
learning_rate_decay = 0.999  # 学习率衰减
max_steps = 30000  # 最大训练步数
training_step = tf.Variable(0, trainable=False)  # 训练轮数,随着训练一次增加，并且设置为不可训练


# 隐藏层和输出层的前向传播计算方式，激活函数使用relu
def hidden_layer(input_tensor, weights1, biases1, weights2, biases2, layer_name):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.nn.relu(tf.matmul(layer1, weights2) + biases2)


# x在运行时feed图片(image)数据
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
# y_在运行时feed答案(label)数据
y_ = tf.placeholder(tf.float32, [None, 10], name='y-output')

# 生成隐藏层参数
weights1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1), name='weight1')
biases1 = tf.Variable(tf.constant(0.1, shape=[2000]), name='biases1')

# 生成输出层参数
weights2 = tf.Variable(tf.truncated_normal([2000, 10], stddev=0.1), name='weight2')
biases2 = tf.Variable(tf.constant(0.1, shape=[10]), name='biases2')

# 计算经过神经网络前向传播后得到的y值
y = hidden_layer(x, weights1, biases1, weights2, biases2, 'y')

# 初始化一个滑动平均类，衰减率为0.99，使用参数的影子变量计算
average_class = tf.train.ExponentialMovingAverage(0.99, training_step)
average_op = average_class.apply(tf.trainable_variables())  # 训练除不可训练（trainable=False）的所有变量
average_y = hidden_layer(x, average_class.average(weights1), average_class.average(biases1),
                         average_class.average(weights2), average_class.average(biases2), 'average_y')
# average()真正执行了影子变量的计算

# 计算损失值loss，使用softmax函数
# argmax(input,axis,name,dimension)用于计算每个样例的预测答案，y_是一个batch_size行10列的的二维数组，然后选取每行最大值所在的索引（下标）
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
regularization = regularizer(weights1) + regularizer(weights2)
loss = tf.reduce_mean(cross_entropy) + regularization
# 指数衰减法设置学习率
decayed_learning_rate = tf.train.exponential_decay(learning_rate, training_step, mnist.train.num_examples / batch_size,
                                                   learning_rate_decay, staircase=False)
# 再使用梯度下降优化算法来优化交叉熵损失和正则化损失
optimizer = tf.train.GradientDescentOptimizer(decayed_learning_rate)
train_step = optimizer.minimize(loss, global_step=training_step)

# 训练时既要反向传播更新神经网络参数，又要更新参数的滑动平均值，所以使用group方法，同时更新两个tensor或operation
train_op = tf.group(train_step, average_op)
# with tf.control_dependencies([training_step, average_op]):
#     train_op = tf.no_op(name='train')

# 比较真实结果和模型结果是否相等
correct_prediction = tf.equal(tf.argmax(average_y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    graph = sess.graph
    writer = tf.summary.FileWriter('../logs/', graph)
    writer.close()

    tf.global_variables_initializer().run()

    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    for i in range(max_steps):
        if i % 1000 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_feed)
            print("After %d training step(s), validation accuracy using average model is %g%%" % (
                i, validate_accuracy * 100))

        xs, ys = mnist.train.next_batch(batch_size)
        sess.run(train_op, feed_dict={x: xs, y_: ys})
    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s), test accuracy using average model is %g%%" % (max_steps, test_accuracy * 100))
