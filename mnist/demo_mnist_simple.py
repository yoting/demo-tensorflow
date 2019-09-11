import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取样本数据
mnist = input_data.read_data_sets('../data', one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

print(mnist.train.images[0])
print(mnist.train.labels[0])

# x在运行时feed图片(image)数据
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
# y_在运行时feed答案(label)数据
y_ = tf.placeholder(tf.float32, [None, 10], name='y-output')

# 初始化训练参数
weights = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1), name='weight')
biases = tf.Variable(tf.constant(0.1, shape=[500]), name='biases')
# 计算训练结果
y = tf.matmul(x, weights) + biases

# 计算损失值loss，通过softmax函数计算交叉熵
# argmax(input,axis,name,dimension)用于计算每个样例的预测答案，y_是一个batch_size行10列的的二维数组，然后选取每行最大值所在的索引（下标）
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))

# 训练模型，使用梯度下降优化，使得交叉熵趋于最小值
train_op = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 比较真实结果和模型结果是否相等
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# tf.summary.histogram('loss', cross_entropy)
tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('input_reshape'):
    image_shape_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shape_input, 10)

merged = tf.summary.merge_all()

# 在会话中运行上面的图

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    writer = tf.summary.FileWriter('../logs', sess.graph)

    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    for i in range(1000):
        if i % 100 == 0:
            summary, validate_accuracy = sess.run([merged, accuracy], feed_dict=validate_feed)
            writer.add_summary(summary, i)
            print("After %d training step(s), validation accuracy using average model is %g%%" % (
                i, validate_accuracy * 100))

        xs, ys = mnist.train.next_batch(100)
        sess.run(train_op, feed_dict={x: xs, y_: ys})

    writer.close()

    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print("After %d training step(s), test accuracy using average model is %g%%" % (30000, test_accuracy * 100))
