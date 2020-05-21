import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

h = np.linspace(-10, 10, 50)

out = tf.nn.softmax(h)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	y = sess.run(out)
	plt.xlabel('inputs')
	plt.ylabel('outputs')
	plt.title('Softmax Activation Function')
	plt.plot(h, y)
	plt.show()