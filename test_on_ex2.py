from data_tools import *
from algorithms import *
from plot_lib import *
# from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code 


# Loading the data into X, y
DATA = loadnumpy("data_no_zeros.npy").astype(np.float64)
ground_truth = loadnumpy("labels_no_zeros.npy").astype(int)

DATA_ex2 = loadnumpy("ex_2_data_no_zeros.npy").astype(np.float64)
ground_truth_ex2 = loadnumpy("ex_2_labels_no_zeros.npy").astype(int)


class_names = ["0", "1", "2", "3"]


print "Loaded data and ground_truth of exp 1 and 2."

# Normalization
# for i in xrange(DATA.shape[1]):
# 	DATA[:,i] = (DATA[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)
# 	DATA_ex2[:,i] = (DATA_ex2[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)


print "The test data will be the data of exp 2."
X_train = DATA
y_train = ground_truth

# Perform PCA: 
# th = 5
# pca = PCA()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# print "PCA done"
# plot3d(X_train[::th,:], ground_truth[::th], "PCA")


#Cross Validation
cv = 5

# Random Forest:
print "Training on exp. 1:"
K = 1000
n_features = 100

# K = 500, n = "auto" - 96%train, 74%test


rf = randForest(X_train, y_train, K, n_features = n_features)
print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X_train, y_train, cv=cv))
print "---------------------------------------"
print "Predicting on exp. 2:"
y_pred_ex2 = rf.predict(DATA_ex2)
print "The prediction accuracy on exp. 2 is " + str(accuracy(ground_truth_ex2, y_pred_ex2)) + "%."
code.interact(local=dict(globals(), **locals()))
plotNiceConfusionMatrix(ground_truth_ex2, y_pred_ex2, class_names)