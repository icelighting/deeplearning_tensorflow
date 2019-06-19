import tensorflow as tf

embeddings = tf.Variable(tf.truncated_normal([3500,128], -1, 1), name='embeddings')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(embeddings.shape)
    print(embeddings.eval())
