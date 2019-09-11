import tensorflow as tf
from tensorflow.python.framework import graph_util

v0 = tf.Variable([[1, 2], [2, 3]], name='v0')
v1 = tf.placeholder(tf.int32, shape=(2, 2), name='v1')
v2 = tf.placeholder(tf.int32, shape=(2, 2), name='v2')

temp = tf.add(v1, v2, name='add_v1_v2')
result = tf.add(v0, temp, name='add_result')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # 计算模型结果
    sess.run(init)
    print(sess.run(result, feed_dict={v1: [[1, 2], [3, 4]], v2: [[5, 6], [7, 8]]}))
    print(result.name)

    # 可视化模型
    writer = tf.summary.FileWriter('logs/', sess.graph)
    writer.close()

    # 持久化模型
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add_result'])

    with tf.gfile.GFile('out/model_add.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
