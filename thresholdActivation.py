import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def threshold(x):
	cond = tf.less(x, tf.zeros(tf.shape(x), dtype = x.dtype))
	out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
	return out

h = np.linspace(-1, 1, 50)
out = threshold(h)
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	y = sess.run(out)
	plt.xlabel('input value')
	plt.ylabel('output value')
	plt.title('Threshold Activation Function')
	plt.plot(h, y)
	plt.show()