import tensorflow as tf

v0 = tf.Variable([[1, 2], [2, 3]], name='v0')
v1 = tf.placeholder(tf.int32, shape=(2, 2), name='v1')
v2 = tf.placeholder(tf.int32, shape=(2, 2), name='v2')

temp = tf.add(v1, v2, name='add_v1_v2')
result = tf.add(v0, temp, name='add_result')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    builder = tf.saved_model.builder.SavedModelBuilder('out/model_add')
    builder.add_meta_graph_and_variables(sess, ['mymodel'])
    builder.save()
