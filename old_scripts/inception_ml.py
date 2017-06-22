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
DATA = loadnumpy("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full_no_zeros_in_cells_alexnet_inception_fc8.npy")
labels_and_thresholds = np.load("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_80_80_full_no_zeros_in_cells.npy").astype(int)
ground_truth = labels_and_thresholds[:,0]

print "Loaded data and ground_truth of exp 1."
print "Start preprocessing:"
# DATA2 = loadnumpy("data.npy").astype(np.float64)
# ground_truth2 = loadnumpy("labels.npy").astype(int)
# print "Loaded data and ground_truth of exp 1"

# Normalization
for i in xrange(DATA.shape[1]):
	DATA[:,i] = (DATA[:,i] - np.mean(DATA[:,i]))/(np.max(DATA[:,i])-np.min(DATA[:,i])+.001)
print "- normalized data"

# Remove class 5 from Data:
DATA = DATA[ground_truth!=4, :]
ground_truth = ground_truth[ground_truth!=4]
print "- removed the last class for comparison with cell profiler"

X_train, X_test, y_train, y_test = train_test_split(DATA, ground_truth, test_size=0.33, random_state=42)
print "- shuffled data and split in train-/testset"




# Perform PCA:
# print "performing PCA." 
# th = 10
# pca = pca(X_train, 300)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)
# print "PCA done"
#plot2d(X_train[::th,:], y_train[::th], "PCA")





# Lasso
# alpha = .05 #if alpha = 0, use normal unreg.
# tol = 0.000001
# lasso = lasso(X_train, y_train, alpha, tol)
# print "Lasso's sparse coefficients are:", lasso.sparse_coef_ 
# c1 = 20
# log_regl = logRegress(X_off[:,[0, 3]], y, c1)
# print "With the selected Coefficients, Logistic Regression gives accuracy:", np.mean(cross_val_score(log_regl, X_off[:,[0, 3]], y, cv=cv))
# plotConfusionMatrix(y, log_regl.predict(X_off[:,[0, 3]]))









#Cross Validation
cv = 7


# # Logistic Regression (Ridge)
# print "Performing Logistic Regression."
# reg = 1
# X_off = addOffset(X_train)
# log_reg = logRegress(X_off, y_train, reg)
# train_accuracy = np.mean(cross_val_score(log_reg, X_off, y_train, cv=cv))
# print "The train accuracy of Logistic Regression is", train_accuracy
	
# y_pred = log_reg.predict(addOffset(X_test))

# code.interact(local=dict(globals(), **locals()))


# plotConfusionMatrix(y_test.astype(int), y_pred.astype(int))
# np.set_printoptions(precision=2)

# print "The test accuracy is", testAccuracy(y_test.astype(int), y_pred.astype(int))


# SVM
# achieved Train Accuracy of 91% with c2 = .1 - 4
# best c2 = .1
# adding bias/offset


# X_off = addOffset(X_train)

# c2 = 0.3
# svm = supvecm(X_off,y_train,c2)
# train_accuracy = np.mean(cross_val_score(svm, X_off, y_train, cv=cv))
# print "The train accuracy of SVM is", train_accuracy
# y_pred = svm.predict(addOffset(X_test))
# plotConfusionMatrix(y_test, y_pred)


# 	if train_accuracy >= accuracy:
# 		accuracy = np.mean(cross_val_score(svm, X_off, y, cv=cv))
# 		best_c2 = i
# 		best_model = svm
# 		print "new best!"

# k_nn
# train accuracy with n=11 is 0.96
# neighbors = 11
# k_nn = k_nn(X, y, neighbors)
# print "The train accuracy of K-NN is", np.mean(cross_val_score(k_nn, X, y, cv=cv))
# plotConfusionMatrix(y, k_nn.predict(X))
# y_pred = k_nn.predict(X_test)

# Random Forest:
# K = 30
# rf = randForest(X_train, y_train, K)
# print "The train accuracy of Random Forest is", np.mean(cross_val_score(rf, X_train, y_train, cv=cv))
# y_pred = rf.predict(X_test)
# plotConfusionMatrix(y_test, y_pred)






# Fully Connected Net:
# After 100 epoqs achieved accuracy of 99.98% on training, Test_accuracy is 97% - Overfit
# y = y_train.T 
# model = fullyConnectedNet(X_train, y, epochs = 20)
# results = model.predict(X_test)
# y_pred = np.zeros([results.shape[0]])
# for i in xrange(results.shape[0]):
# 	y_pred[i] = np.argmax(results[i,:]).astype(int)


# T-SNE
X_tSNE = tsne(DATA, 3, 50, 5.0)
plot3d(X_tSNE,y,"tSNE-Plot")
code.interact(local=dict(globals(), **locals()))