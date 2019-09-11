import tensorflow as tf

with tf.Session() as sess:
    meta_graph_def = tf.saved_model.loader.load(sess, ['mymodel'], 'out/model_add')
    v1 = sess.graph.get_tensor_by_name('v1:0')
    v2 = sess.graph.get_tensor_by_name('v2:0')
    add_result = sess.graph.get_tensor_by_name('add_result:0')
    print(sess.run(add_result, feed_dict={v1: [[1, 2], [3, 4]], v2: [[5, 6], [7, 8]]}))
