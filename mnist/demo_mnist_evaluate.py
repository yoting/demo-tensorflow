import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读取数据
mnist = input_data.read_data_sets('data', one_hot=True)


# 定义相同的前向传播过程
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

# 不需要正则化损失值，不传正则方法
y = hidden_layer(x, None, name='y')

# 比较真实结果和模型结果是否相等
correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

variable_averages = tf.train.ExponentialMovingAverage(0.99)
saver = tf.train.Saver(variable_averages.variables_to_restore())

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}

    ckpt = tf.train.get_checkpoint_state('out/')

    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    print('The lastest ckpt is mnist_model.ckpt-%s' % global_step)

    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
    print('After %s training step(s), validation accuracy = %g%%' % (global_step, accuracy_score * 100))

    test_accuracy = sess.run(accuracy, feed_dict=test_feed)
    print('After %s training step(s), test accuracy = %g%%' % (global_step, accuracy_score * 100))
