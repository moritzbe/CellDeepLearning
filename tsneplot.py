from data_tools import *
# from algorithms import *
from plot_lib import *
from netDeepFlow import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import load_model
from sklearn.metrics import log_loss
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import itertools
import matplotlib.pyplot as plt
import numpy as np
import h5py
import _pickle as cPickle
import code
import os


split = 0.8
np.random.seed(seed=random_state)

def split_train_test(X_train, y_train, split):
    split_idx = int(round(y_train.shape[0]*split))
    indices = np.arange(0, y_train.shape[0])
    np.random.shuffle(indices)
    training, test = indices[:split_idx], indices[split_idx:]
    return X_train[training,:,:,:], y_train[training], X_train[test,:,:,:], y_train[test]


h5f = h5py.File('/home/moritz_berthold/blasiData/X_blasi_original.h5', 'r')
X = h5f['X'][()]
y = h5f['y'][()]
fnames = h5f['filenames'][()]
h5f.close()

# show_cell_image(X, y, fnames)

X_train_ex1 = np.zeros([X.shape[0], 66, 66, 2])
# np.unique(y, return_counts=True) = (array([0, 1, 2, 3, 4, 5, 6], dtype=uint8), array([   15, 14333,  8601,    68,   606,  8616,    27]))
print("Loading training and test data. Use ex1 for training and tuning and ex2 for testing.")
for i in range(X.shape[0]):
    X_train_ex1[i,16:48,16:48,:] = X[i,:,:,:]
y_train_ex1 = y
print("done")
print("Trainingdata shape = ", X_train_ex1.shape)
print("Traininglabels shape = ", y_train_ex1.shape)
X_train, y_train, X_test_temp, y_test_temp = split_train_test(X_train_ex1, y_train_ex1, split=split)
X_test = X_test_temp[:int(round(X_test_temp.shape[0]/2)),:,:,:]
y_test = y_test_temp[:int(round(X_test_temp.shape[0]/2))]
X_test_2 = X_test_temp[int(round(X_test_temp.shape[0]/2)):,:,:,:]
y_test_2 = y_test_temp[int(round(X_test_temp.shape[0]/2)):]



print("Reshaping done. Use Test, Train and Evaluation Data")
print(X_train.shape)
print(X_test.shape)
# code.interact(local=dict(globals(), **locals()))

def reduceClasses(labels):
    # G1 = 1, G2 = 2, S = 5, Pro = 4, Meta = 3, Telo = 7, Ana = 0
    print(labels.shape)
    out = np.zeros_like(labels)
    for i in range(labels.shape[0]):
        if labels[i] == 1 or labels[i] == 2 or labels[i] == 5:
            out[i] = 0
        if labels[i] == 4:
            out[i] = 1
        if labels[i] == 3:
            out[i] = 2
        if labels[i] == 0:
            out[i] = 3
        if labels[i] == 6:
            out[i] = 4
    return out

class_names = ["Ana","G1","G2","Meta","Pro","S","Telo"]

#### TSNE ####

h5f = h5py.File('2d_tsne.h5', 'r')
data = h5f['X'][()]
h5f.close()

code.interact(local=dict(globals(), **locals()))
plot2d(data, y_test_2, class_names, "tSNE last layer activations")
