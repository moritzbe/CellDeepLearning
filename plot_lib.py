import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D


def plot3d(X,y,title):
	fig = plt.figure(figsize=(6,6))
	fig.suptitle(title, fontsize=20)
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('red')
	ax.set_ylabel('green')
	ax.set_zlabel('blue')
	colors = ['r','g','b','black','white','c','m','y','#CCB974','#77BEDB']
	for i in np.unique(y):
		ax.scatter(X[np.where([y==i])[1], 0], X[np.where([y==i])[1], 1], X[np.where([y==i])[1], 2], c=colors[i], marker='o')
	plt.show()

def plot2d(X, y, title):
	fig = plt.figure(figsize=(6,6))
	fig.suptitle(title, fontsize=20)
	ax = fig.add_subplot(111)
	ax.set_xlabel("n1",fontsize=12)
	ax.set_ylabel("n2",fontsize=12)
	ax.grid(True,linestyle='-',color='0.75')
	colors = ['r','g','b','black','c','m','y','#CCB974','#77BEDB']

	for i in np.unique(y):
		ax.scatter(X[np.where([y==i])[1], 0], X[np.where([y==i])[1], 1], c=colors[i-1], marker='o')

	ax.set_xlabel('n1')
	ax.set_ylabel('n2')

	plt.show()

def plotImage(X, entry, label = None):
	img = X[entry,:].reshape((28, 28))
	plt.title('Scores are {label}'.format(label=label))
	plt.imshow(img, cmap='gray')
	plt.show()

def plotHistogram(x, bins, xlabel, ylabel, title):
	# the histogram of the data
	n, bins, patches = plt.hist(x, bins)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	# plt.axis([xmin, xmax, ymin, ymax])
	plt.grid(True)
	plt.show()
