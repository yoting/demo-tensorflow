import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    with gfile.FastGFile('out/model.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    result = tf.import_graph_def(graph_def, return_elements=['matmul_v1_v2:0'])
    print(sess.run(result))
