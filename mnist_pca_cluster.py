import tensorflow as tf
import numpy as numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib.tensorboard.plugins import projector
import os

LOG_DIR = './pca'
metadata = 'meta.tsv'


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data')
images = tf.Variable(mnist.test.images, name='images')

with open(metadata, 'w') as metadata_file:
	for row in mnist.test.labels:
		metadata_file.write('%d\n' % row)

with tf.Session() as sess:
	saver = tf.train.Saver([images])

	sess.run(images.initializer)
	saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

	config = projector.ProjectorConfig()
	
	embedding = config.embeddings.add()
	embedding.tensor_name = images.name
	embedding.metadata_path = metadata

	projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)