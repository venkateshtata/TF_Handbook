import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target

def input_fn():
	return tf.constant(np.array(x), tf.float32, x.shape), None

kmeans = tf.contrib.learn.KMeansClustering(num_clusters=3, relative_tolerance=0.0001, random_seed=2)

kmeans.fit(input_fn=input_fn)

clusters = kmeans.clusters()

assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))

def scatterplot(X, Y, assignments=None, centers=None):
	if assignments is None:
		assignments = [0] * len(X)
		fig = plt.figure(figsize=(14, 8))
		cmap = ListedColormap(['red', 'green', 'blue'])
		plt.scatter(X, Y, c=assignments, cmap = cmap)
		if centers is not None:
			plt.scatter(centers[:,0], centers[:,1], c=range(len(centers)), marker='+', s=400, cmap=cmap)
			plt.xlabel('Sepia Length')
			plt.ylabel('Sepia Width')

scatterplot(x[:, 0], x[:, 1], assignments, clusters) 