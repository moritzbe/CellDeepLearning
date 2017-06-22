from data_tools import *
from algorithms import *
from plot_lib import *
from sklearn import svm, cluster, linear_model
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code
import cPickle
import forestci as fci
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
import pylab as pl


intensities = loadnumpy("data_ch1_pred/ex_1_intensities_no_zeros.npy").astype(np.int16)
intensities_test = loadnumpy("data_ch1_pred/ex_2_intensities_no_zeros.npy").astype(np.int16)
intensities_test_ex3 = loadnumpy("data_ch1_pred/ex_3_intensities_no_zeros.npy").astype(np.int16)

code.interact(local=dict(globals(), **locals()))
# channel to predict
cv = 5
max_features=1 # 1 is better on testset
K = 1000
ch = 2
dapi = False

# 2) Predict RIIb intensity from PGP, CGRP
# 3) Predict DAPI intensity from PGP, CGRP

y = intensities[:,0,ch+1]
y_test = intensities_test[:,0,ch+1]
y_test_ex3 = intensities_test_ex3[:,0,ch+1]
if ch == 1:
    features = intensities[:,0,[1,3,4]]
    features_test = intensities_test[:,0,[1,3,4]]
    features_test_ex3 = intensities_test_ex3[:,0,[1,3,4]]
    ch = "CGRP"
if ch == 2:
    features = intensities[:,0,[1,2,4]]
    features_test = intensities_test[:,0,[1,2,4]]
    features_test_ex3 = intensities_test_ex3[:,0,[1,2,4]]
    ch = "RIIb"
if dapi == False:
    features = features[:,:2]
    features_test = features_test[:,:2]
    features_test_ex3 = features_test_ex3[:,:2]


print "Predicting channel ", ch
if dapi:
    print "with DAPI"

# code.interact(local=dict(globals(), **locals()))

rand_reg = RandomForestRegressor(n_estimators=K, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=max_features, max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=2, warm_start=False)
predicted = cross_val_predict(rand_reg, features, y, cv=cv)
rand_reg.fit(features, y)
predicted = rand_reg.predict(features)
y_pred_ex2 = rand_reg.predict(features_test)
y_pred_ex3 = rand_reg.predict(features_test_ex3)
rms1 = rms(y, predicted)**.5
rms2 = rms(y_test, y_pred_ex2)**.5
rms3 = rms(y_test_ex3, y_pred_ex3)**.5

#print "The rms error/loss on exp. 1 is: ", rms1
#print "The pearson correlation coefficient on exp. 1 is ", pearsonr(y, predicted)
print "The mean of exp. 1 is", np.mean(y)
print "The rms error/loss on exp. 1 is: ", rms1
print "The rms error/loss on exp. 2 is: ", rms2
print "The rms error/loss on exp. 3 is: ", rms3
print "The pearson correlation coefficient on exp. 2 is ", pearsonr(y_test, y_pred_ex2)
print "The Spearman correlation coefficient on exp. 2 is ", spearmanr(y_test, y_pred_ex2)

# i = np.argsort(y)
# y = y[i]
# predicted = predicted[i]


plt.scatter(y_pred_ex2, y_test, marker='.', color='r', s = 2, label='Pearson correlation = {1:2f}\nSpearman correlation = {1:2f}' ''.format(pearsonr(y_test, y_pred_ex2)[0], spearmanr(y_test, y_pred_ex2)[0]))
plt.xlim([0, 12000])
plt.ylim([0, 12000])
plt.ylabel('Prediction')
plt.xlabel('Ground Truth')
plt.title("Intensity prediction of channel " + str(ch) + " using Random Forest regression")
plt.legend(loc="upper right")
plt.show()


# plt.plot(predicted,c='r',alpha=0.5)
# plt.plot(y,c='blue',alpha=0.5)
#
i = np.argsort(y_test)
y_test = y_test[i]
y_pred_ex2 = y_pred_ex2[i]
plt.plot(y_pred_ex2,c='r',alpha=0.5)
plt.plot(y_test,c='blue',alpha=0.5, linewidth=3)
plt.xlim([0, len(y_test)])
plt.ylim([0, 6000])
plt.title(str(ch) + " intensity estimate and ground truth ordered by magnitude")
plt.ylabel('Intensity')
plt.xlabel('Cells ordered by ground truth intensity magnitude')
plt.show()

code.interact(local=dict(globals(), **locals()))
print "-------------------------------------"




intensities = loadnumpy("data_ch1_pred/ex_1_intensities_no_zeros.npy").astype(np.int16)
intensities_test = loadnumpy("data_ch1_pred/ex_2_intensities_no_zeros.npy").astype(np.int16)

# channel to predict
cv = 5


ch = 2
y = intensities[:,0,ch+1]
y_test = intensities_test[:,0,ch+1]
if ch == 1:
    features = intensities[:,0,[1,3,4]]
    features_test = intensities_test[:,0,[1,3,4]]
    ch = "CGRP"
if ch == 2:
    features = intensities[:,0,[1,2,4]]
    features_test = intensities_test[:,0,[1,2,4]]
    ch = "RIIb"

print "Predicting channel ", ch

rand_reg = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=2, max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=2, warm_start=False)
predicted = cross_val_predict(rand_reg, features, y, cv=cv)
rand_reg.fit(features, y)
y_pred_ex2 = rand_reg.predict(features_test)
# rms1 = rms(y, predicted)**.5
rms2 = rms(y_test, y_pred_ex2)**.5

def detectPredictedLabel(ch, y_test, y_pred):
    mean1 = y_test[0]
    mean4 = y_test[-1]
    if ch == "CGRP":
        mean2 = y_pred
        mean3 = y_test[1]
    if ch == "RIIb":
        mean2 = y_test[1]
        mean3 = y_pred
    # Neurons (+/+), CGRP positive, RIIb positive -> Red
    if (mean2 > 1800) & (mean3 > 700) & (mean4 < 1500):
        label = 0
    # +/-, CGRP positive, RIIb negative -> Green
    elif (mean2 > 1800) & (mean3 <= 700) & (mean4 < 1500):
        label = 1
    # -/+, CGRP negative, RIIb positive -> Blue
    elif (mean2 <= 1800) & (mean3 > 700) & (mean4 < 1500):
        label = 2
    # -/-, CGRP negative, RIIb negative -> Black
    elif (mean2 <= 1800) & (mean3 <= 700) & (mean4 < 1500):
        label = 3
    # outliers -> White
    elif mean4 >= 1500:
        label = 4
    else:
        pass

    return label

def detectTrueLabel(y_test):
    mean1 = y_test[0]
    mean2 = y_test[1]
    mean3 = y_test[2]
    mean4 = y_test[3]
    # Neurons (+/+), CGRP positive, RIIb positive -> Red
    if (mean2 > 1800) & (mean3 > 700) & (mean4 < 1500):
        label = 0
    # +/-, CGRP positive, RIIb negative -> Green
    elif (mean2 > 1800) & (mean3 <= 700) & (mean4 < 1500):
        label = 1
    # -/+, CGRP negative, RIIb positive -> Blue
    elif (mean2 <= 1800) & (mean3 > 700) & (mean4 < 1500):
        label = 2
    # -/-, CGRP negative, RIIb negative -> Black
    elif (mean2 <= 1800) & (mean3 <= 700) & (mean4 < 1500):
        label = 3
    # outliers -> White
    elif mean4 >= 1500:
        label = 4
    else:
        pass

    return label

pred_label = []
true_label = intensities_test[:,0,0]
for i in xrange(y_pred_ex2.shape[0]):
    pred_label.append(detectPredictedLabel(ch, features_test[i,:], y_pred_ex2[i]))

print "Accuracy :", accuracy(true_label, pred_label)


class_names = ["CGRP+ RIIb+", "CGRP+ RIIb-", "CGRP- RIIb+", "CGRP- RIIb-"]
plotNiceConfusionMatrix(true_label, pred_label, class_names)
plotRelativeConfusionMatrix(true_label, pred_label, class_names)
