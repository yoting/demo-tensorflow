import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../data', one_hot=True)

batch_size = 100  # 每一轮训练的大小
learning_rate = 0.8  # 初始化学习率
learning_rate_decay = 0.999  # 学习率衰减
max_steps = 30000  # 最大训练步数


# 隐藏层和输出层的前向传播计算方式，激活函数使用relu
def hidden_layer(input_tensor, regularizer, name):
    with tf.variable_scope('hidden_layer'):
        weights = tf.get_variable('weights', [784, 500], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        biases = tf.get_variable('biases', [500], initializer=tf.constant_initializer(0.0))
        hidden_layer = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)
    with tf.variable_scope('hidden_layer_output'):
        weights = tf.get_variable('weights', [500, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(weights))
        biases = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.0))
        hidden_layer_out = tf.nn.relu(tf.matmul(hidden_layer, weights) + biases)
    return hidden_layer_out


# x在运行时feed图片(image)数据
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
# y_在运行时feed答案(label)数据
y_ = tf.placeholder(tf.float32, [None, 10], name='y-output')

# 定义L2正则化方法，计算训练结果
regularizer = tf.contrib.layers.l2_regularizer(0.0001)
y = hidden_layer(x, regularizer, name='y')

training_step = tf.Variable(0, trainable=False)  # 训练轮数,随着训练一次增加，并且设置为不可训练

# 初始化一个滑动平均类，衰减率为0.99
average_class = tf.train.ExponentialMovingAverage(0.99, training_step)
average_op = average_class.apply(tf.trainable_variables())  # 训练除不可训练（trainable=False）的所有变量

# 计算损失值loss，使用softmax函数
# argmax(input,axis,name,dimension)用于计算每个样例的预测答案，y_是一个batch_size行10列的的二维数组，然后选取每行最大值所在的索引（下标）
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
loss = tf.reduce_mean(cross_entropy) + tf.add_n(tf.get_collection('losses'))
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

# 初始化Saver持久化类
saver = tf.train.Saver()

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(max_steps):
        x_train, y_train = mnist.train.next_batch(batch_size)
        _, loss_value, step = sess.run([train_op, loss, training_step], feed_dict={x: x_train, y_: y_train})

        if i % 1000 == 0:
            print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
            saver.save(sess, 'out/mnist_model.ckpt', global_step=training_step)
