import tensorflow as tf
import numpy as np

# 创建数据 ¶
x_data = np.random.rand(100).astype('float32')
y_data = x_data * 0.8 + 0.3

# 搭建模型 ¶
weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases

# 计算误差 ¶ 计算 y 和 y_data 的误差
loss = tf.reduce_mean(tf.square(y - y_data))

# 传播误差 ¶ 反向传递误差的工作就交给optimizer了, 我们使用的误差传递方法是梯度下降法: Gradient Descent 让后我们使用 optimizer 来进行参数的更新.
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 训练 ¶ 到目前为止, 我们只是建立了神经网络的结构, 还没有使用这个结构.
# 在使用这个结构之前, 我们必须先初始化所有之前定义的Variable
init = tf.global_variables_initializer()
# 再创建会话 Session 并且用 Session 来 run 每一次 training 的数据. 逐步提升神经网络的预测准确性.
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(weights), sess.run(biases), sess.run(loss))
sess.close()
