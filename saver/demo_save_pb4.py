import tensorflow as tf
from tensorflow.python.platform import gfile

with gfile.FastGFile('out/model_add.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    result, v1, v2 = tf.import_graph_def(graph_def, return_elements=['add_result:0', 'v1:0', 'v2:0'])

with tf.Session() as sess:
    print(sess.run(result, feed_dict={v1: [[9, 2], [3, 4]], v2: [[5, 6], [7, 8]]}))
