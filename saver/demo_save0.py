import tensorflow as tf

v1 = tf.Variable(tf.truncated_normal([2, 5], stddev=1), name='v1')
v2 = tf.Variable(tf.constant(0.1, shape=(5, 2)), name='v2')

result = tf.matmul(v1, v2)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
saver.export_meta_graph('out/model_ckpt_meta_json', as_text=True)
with tf.Session() as sess:
    sess.run(init)

    saver.save(sess, 'out/model.ckpt')

    print(v1)
    print(result)
    print(sess.run(v1))
    print(sess.run(result))
