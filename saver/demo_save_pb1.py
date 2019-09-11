import tensorflow as tf
from tensorflow.python.framework import graph_util

v1 = tf.Variable(tf.truncated_normal([2, 5], stddev=1), name='v1')
v2 = tf.Variable(tf.constant(0.1, shape=(5, 2)), name='v2')

result = tf.matmul(v1, v2, name='matmul_v1_v2')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    graph_def = tf.get_default_graph().as_graph_def()

    print(result.name)

    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['matmul_v1_v2'])

    with tf.gfile.GFile('out/model.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
