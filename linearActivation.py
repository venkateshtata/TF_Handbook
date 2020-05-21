import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X_in = tf.Variable(tf.random_normal([1,3], 0, 1))

b = tf.Variable(tf.random_normal([1,1], 0, 1))
w = tf.Variable(tf.random_normal([3,1], 0, 1))

output = tf.matmul(X_in, w) + b

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	print("X_in", sess.run(X_in))
	print("w", sess.run(w))
	out = sess.run(output)
print(out)
