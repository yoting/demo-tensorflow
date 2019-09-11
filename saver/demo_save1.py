import tensorflow as tf

meta_graph = tf.train.import_meta_graph('out/model.ckpt.meta')

with tf.Session() as sess:
    meta_graph.restore(sess, 'out/model.ckpt')
    print(sess.run(tf.get_default_graph().get_tensor_by_name('MatMul:0')))
