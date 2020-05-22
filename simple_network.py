# import the tensorflow library
import tensorflow as tf
import numpy as np

# create the input placeholder
X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

# create network parameters
W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[784, 200], initializer=weight_initer)
bias_initer =tf.constant(0., shape=[200], dtype=tf.float32)
b = tf.get_variable(name="Bias", dtype=tf.float32, initializer=bias_initer)

# create MatMul node
x_w = tf.matmul(X, W, name="MatMul")
# create Add node
x_w_b = tf.add(x_w, b, name="Add")
# create ReLU node
h = tf.nn.relu(x_w_b, name="ReLU") 

# Add an Op to initialize variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
    # initialize variables
    sess.run(init_op)
    # create the dictionary:
    d = {X: np.random.rand(100, 784)}
    # feed it to placeholder a via the dict 
    print(sess.run(h, feed_dict=d))