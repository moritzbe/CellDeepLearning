import numpy as np
from numpy import genfromtxt
from sklearn.metrics import confusion_matrix, mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import itertools
import math
# import pandas
import code
# import lmdb

def conf_M2(cm, rel_cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	# code.interact(local=dict(globals(), **locals()))
	thresh = cm.max() / 2.
	plt.title("Absolute and Normalized confusion matrix")
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, str(cm[i, j]) + "\n" + str(round(rel_cm[i, j], 3)*100) + "%.", horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.title("Absolute and relative confusion matrix")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def plotBothConfusionMatrices():
	# Compute confusion matrix
    # class_names = ["CGRP+ RIIb+", "CGRP+ RIIb-", "CGRP- RIIb+", "CGRP- RIIb-"]
	class_names = ["RIIb+", "RIIb-"]
	cm = np.array([[17175  ,  1008  ],[  730  , 15179 ]])
	rel_cm = np.array([[0.94 , 0.06 ],[ 0.05 , 0.95 ]])
	# Plot non-normalized confusion matrix
	fig = plt.figure()
	conf_M2(cm, rel_cm, classes=class_names, title='Confusion matrix')
	fig.set_tight_layout(True)
	plt.show()

plotBothConfusionMatrices()
