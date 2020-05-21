import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

h = np.linspace(-10,10,50)
out = tf.nn.relu(h)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	y = sess.run(out)
	plt.xlabel('input')
	plt.ylabel('output')
	plt.title('Rectified Linear Unit')
	plt.plot(h, y)
	plt.show()