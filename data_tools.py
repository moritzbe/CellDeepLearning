import numpy as np
from numpy import genfromtxt
from StringIO import StringIO
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
# import pandas
import code
# import lmdb



# This function loads the data into X and y, 
# outputs the feature names, the label dictionary, m and n
def loaddata(filename):
	data = genfromtxt(filename, delimiter=',', dtype=None)
	X = data[1:,:].astype(float)
	# y = data[1:,0].astype(int)	
	np.save("X_TEST", X, allow_pickle=True, fix_imports=True)
	# np.save("Y_TRAIN", y.T, allow_pickle=True, fix_imports=True)

def loadnumpy(filename):
	array = np.load(filename)
	return array

def addOffset(X):
	X = np.c_[np.ones((X.shape[0],1)), X]
	return X

def normalize(DATA):
	for i in xrange(DATA.shape[1]):
		DATA[:,i] = (DATA[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)
	return DATA


# def saveToCSV(y, alg_name="alg"):
# 	ImageId = []
# 	for i in range(1, y.shape[0]+1):
# 		ImageId.append(i)

# 	submission = pandas.DataFrame({
#         "ImageId": ImageId,
#         "Label": y
#     })
# 	filename = "kaggle_mnist_" + alg_name + ".csv"
# 	submission.to_csv(filename, index=False)



def setOtherLabelsZero(m, value):
	a = np.array([m.shape])
	a[m == value] = 1
	return a

def plotConfusionMatrix(y, pred):
	n_labels = len(np.unique(y))
	matrix = np.zeros([n_labels+1,n_labels+1])
	matrix[0,1:] = np.unique(y)
	matrix[1:,0] = np.unique(y)
	matrix[1:,1:] = confusion_matrix(y.astype(int), pred.astype(int))
	print "The confusion matrix (Truth X Prediction):"
	print matrix
	print "---------------------------------------------------\t"

def conf_M(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plotNiceConfusionMatrix(y_test, y_pred, class_names):
	# Compute confusion matrix
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	fig = plt.figure()
	conf_M(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
	fig.set_tight_layout(True)
	
	plt.show()

def accuracy(y_test, pred):
	rights = 0
	for i in xrange(len(y_test)):
		if y_test[i] == pred[i]:
			rights += 1
	accuracy = float(rights) / float(len(y_test))
	return round(accuracy, 4)*100

# def readLMDB(db_path):
# 	env = lmdb.open(db_path, readonly=True)
# 	with env.begin() as txn:
# 	    raw_datum = txn.get(b'00000000')

# 	datum = caffe.proto.caffe_pb2.Datum()
# 	datum.ParseFromString(raw_datum)

# 	flat_x = np.fromstring(datum.data, dtype=np.uint8)
# 	print flat_x.shape
# 	X = flat_x.reshape(datum.channels, datum.height, datum.width)
# 	# y = datum.label

	# with env.begin() as txn:
	#     cursor = txn.cursor()
	#     for key, value in cursor:
	#         print(key, value)



import numpy as Math
import pylab as Plot

def Hbeta(D = Math.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = Math.exp(-D.copy() * beta);
	sumP = sum(P);
	H = Math.log(sumP) + beta * Math.sum(D * P) / sumP;
	P = P / sumP;
	return H, P;


def x2p(X = Math.array([]), tol = 1e-5, perplexity = 30.0):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print "Computing pairwise distances..."
	(n, d) = X.shape;
	sum_X = Math.sum(Math.square(X), 1);
	D = Math.add(Math.add(-2 * Math.dot(X, X.T), sum_X).T, sum_X);
	P = Math.zeros((n, n));
	beta = Math.ones((n, 1));
	logU = Math.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print "Computing P-values for point ", i, " of ", n, "..."

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -Math.inf;
		betamax =  Math.inf;
		Di = D[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))];
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		while Math.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == Math.inf or betamax == -Math.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == Math.inf or betamin == -Math.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		P[i, Math.concatenate((Math.r_[0:i], Math.r_[i+1:n]))] = thisP;

	# Return final P-matrix
	print "Mean value of sigma: ", Math.mean(Math.sqrt(1 / beta));
	return P;


def pca(X = Math.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print "Preprocessing the data using PCA..."
	(n, d) = X.shape;
	X = X - Math.tile(Math.mean(X, 0), (n, 1));
	(l, M) = Math.linalg.eig(Math.dot(X.T, X));
	Y = Math.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
		print "Error: array X should have type float.";
		return -1;
	if round(no_dims) != no_dims:
		print "Error: number of dimensions should be an integer.";
		return -1;

	# Initialize variables
	X = pca(X, initial_dims).real;
	(n, d) = X.shape;
	max_iter = 1000;
	max_iter = 800;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	Y = Math.random.randn(n, no_dims);
	dY = Math.zeros((n, no_dims));
	iY = Math.zeros((n, no_dims));
	gains = Math.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity);
	P = P + Math.transpose(P);
	P = P / Math.sum(P);
	P = P * 4;									# early exaggeration
	P = Math.maximum(P, 1e-12);

	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = Math.sum(Math.square(Y), 1);
		num = 1 / (1 + Math.add(Math.add(-2 * Math.dot(Y, Y.T), sum_Y).T, sum_Y));
		num[range(n), range(n)] = 0;
		Q = num / Math.sum(num);
		Q = Math.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = Math.sum(Math.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - Math.tile(Math.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = Math.sum(P * Math.log(P / Q));
			print "Iteration ", (iter + 1), ": error is ", C

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

	# Return solution
	return Y;


# if __name__ == "__main__":
# 	print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
# 	print "Running example on 2,500 MNIST digits..."
# 	X = Math.loadtxt("mnist2500_X.txt");
# 	labels = Math.loadtxt("mnist2500_labels.txt");
	# Y = tsne(X, 2, 50, 20.0);
	# Plot.scatter(Y[:,0], Y[:,1], 20, labels);
	# Plot.show();


