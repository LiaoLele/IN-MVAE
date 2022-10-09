import numpy as np
import tensorflow as tf



sess = tf.Session()
x_input_1 = tf.placeholder(tf.float32, shape=(10), name='x_input_1')
x_input_2 = tf.placeholder(tf.float32, shape=(10), name='x_input_2')
w = tf.Variable(np.ones(10).astype('float32'), name='weight')
loss_op_11 = tf.reduce_sum(x_input_1 * w)
loss_op_12 = tf.reduce_sum(x_input_2 * w)
x_input = tf.concat([x_input_1, x_input_2], axis=0)
# train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss_op)
gradients_node_1 = tf.train.GradientDescentOptimizer(0.1).compute_gradients(loss_op_11, [x_input_1, x_input_2])[1]
gradients_node_2 = tf.train.GradientDescentOptimizer(0.1).compute_gradients(loss_op_12, [x_input_1, x_input_2])[0]
gradients_node = tf.concat([tf.expand_dims(gradients_node_1, 0), tf.expand_dims(gradients_node_2, 0)], axis=0)
gradients_node_sum = tf.reduce_sum(gradients_node, axis=0)
gradients_node_mean = tf.reduce_mean(gradients_node, axis=0)
# loss_op_2 = tf.reduce_sum(tf.pow(gradients_node, 2))
# gradients_node_2 = tf.gradients(loss_op_2, w)[0]
# train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss_op_2)
# print(gradients_node_1)

init = tf.global_variables_initializer()
sess.run(init)
sess.run(tf.local_variables_initializer())

'''构造数据集'''
x_train_1 = np.random.randn(5, 10)
x_train_2 = np.random.randn(5, 10)

for i in range(5):
    gradients_node_sum_p, gradients_node_mean_p = sess.run([gradients_node_sum, gradients_node_mean],
                                                           feed_dict={x_input_1: x_train_1[i], x_input_2: x_train_2[i]})
    print(gradients_node_sum_p, gradients_node_mean_p)
    # gradients_1, gradients_2, _ = sess.run([gradients_node_1, gradients_node_2, train_op], feed_dict={x_input_1: x_train[i]})
    # print("epoch: {} \t gradients_1: {} \t gradients_2: {}".format(i, gradients_1, gradients_2))

sess.close()