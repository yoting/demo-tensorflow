import tensorflow as tf
import numpy as np

# 定义一个4X4X1的三维数组，4X4表示宽高，1表示深度,再reshape成四维数组（方便计算）
M = np.array([[[2], [1], [2], [-1]], [[0], [-1], [3], [0]], [[2], [1], [-1], [4]], [[-2], [0], [-3], [4]]],
             dtype='float32').reshape(1, 4, 4, 1)
print(M)

# 定义过滤器（卷积核）四维数组，第一维第二维表示核大小，第三维度表示当前层深度，第四维度表示核深度
filter_weight = tf.get_variable('weights', [2, 2, 1, 1], initializer=tf.constant_initializer([[-1, 4], [2, 1]]))

# 定义偏置项，深度为1等于神经网络下一层深度
biases = tf.get_variable('biases', [1], initializer=tf.constant_initializer(1))

x = tf.placeholder('float32', [1, None, None, 1], 'input_x')
# 计算卷积结果，卷积核的大小是2X2
conv = tf.nn.conv2d(x, filter_weight, strides=[1, 1, 1, 1], padding='SAME')  # SAME填充规则
# 卷积结果加上偏置项
add_bias = tf.nn.bias_add(conv, biases)

# 池化卷积结果,池化核的大小是2X2
pool = tf.nn.max_pool(add_bias, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    init_op.run()
    print(sess.run(filter_weight))
    M_conv = sess.run(conv, feed_dict={x: M})
    print('卷积结果：')
    print(M_conv)
    M_pool = sess.run(pool, feed_dict={x: M})
    print('池化结果')
    print(M_pool)
