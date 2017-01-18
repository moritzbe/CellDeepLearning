# This file contains logistic regression tools

import numpy as np
from sklearn import svm, cluster, linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier 



#---------------Log Regression--------------------#
#-------------------------------------------------#
def logRegress(X,y,c):
	log_reg = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=c, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=1, warm_start=False, n_jobs=1)

	log_reg.fit(X, y)
	return log_reg 
	

#----------------SVM------------------------------#
#-------------------------------------------------#
def supvecm(X,y,c2):
	lin_clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=c2, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
	lin_clf.fit(X, y)
	return lin_clf


#---------------K-Means---------------------------#
#-------------------------------------------------#
def k_means(X, y, clust):
	k_means = cluster.KMeans(n_clusters=clust, init='k-means++', n_init=10, max_iter=1000, tol=0.0001)
	k_means.fit(X, y)
	return k_means

#---------------K-NN------------------------------#
#-------------------------------------------------#
def k_nn(X, y, neighbors):
	k_nn = KNeighborsClassifier(n_neighbors=neighbors, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski')
	k_nn.fit(X,y)
	return k_nn

#---------------LASSO-----------------------------#
#-------------------------------------------------#
def lasso(X, y, alpha, tol):
	lasso = linear_model.Lasso(alpha = alpha, max_iter=100000, tol = tol)
	lasso.fit(X, y)
	return lasso

#---------------PCA-------------------------------#
#-------------------------------------------------#
def pca(X, n=None):
	pca = PCA(n_components=n)
	return pca.fit(X)

#---------------LDA-------------------------------#
#-------------------------------------------------#
def lda(X, y, n):
	lda = LinearDiscriminantAnalysis(n_components=n)
	lda.fit_transform(X, y)
	return lda, lda.fit_transform(X, y)

#---------------Random-Forsest--------------------#
#-------------------------------------------------#
def randForest(X, y, K=10, n_features="auto"):
	# typical K = 10, 30, 100, the higher, the better!
	rf = RandomForestClassifier(n_estimators=K, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=n_features, max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, verbose=3)
	rf.fit(X, y)
	return rf


