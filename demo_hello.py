import tensorflow as tf
import numpy as np

# 系统默认的计算图
default_graph = tf.get_default_graph()
print(default_graph)
# 定义一个新计算图g1
g1 = tf.Graph()
print(g1)

with g1.as_default():
    a = tf.get_variable('a', [2], initializer=tf.ones_initializer())
    b = tf.get_variable('b', [2], initializer=tf.zeros_initializer())
    result = a + b

# 使用g1进行计算
with tf.Session(graph=g1) as sess:
    print(sess.graph)
    tf.global_variables_initializer().run()
    temp = sess.run(result)
    print(temp)

# 张量tensor 常量 操作-维度-类型
c = tf.constant([[1.0, 2.0], [3, 4], [3, 4]])

# 张量tensor 占位符
p = tf.placeholder(dtype='float32', shape=[2, 3], name='placeholder_p')

# 变量
v = tf.Variable(tf.random_normal(shape=[3, 1], stddev=1, seed=1), name='variable_v')
v2 = tf.get_variable('variable_v2', shape=[3, 1], initializer=tf.ones_initializer())

print(c)
print(p)
print(v)
print(v2)

a = tf.matmul(c, p)
y = tf.matmul(a, v)

init_op = tf.global_variables_initializer()

# 使用with/as 不需要手动关闭session
with tf.Session() as sess:
    g2 = sess.graph
    print(g2)
    writer = tf.summary.FileWriter('logs/', g2)
    writer.close()
    # sess.run(init_op)
    init_op.run()
    print(sess.run(y, feed_dict={p: np.arange(1, 7).reshape(2, 3)}))
