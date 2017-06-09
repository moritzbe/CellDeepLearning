# from data_tools import *
# from algorithms import *
# from plot_lib import *
from netDeepFlow import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import log_loss
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import h5py
import _pickle as cPickle
import code
import os

server = True
train = True
data_normalization = False
gpu = [1]
batch_size = 32
epochs = 85
random_state = 17
n_classes = 7
num_folds = 5
split = .8
channels = [0,1]

### Optimizer ###
lr = 0.01
momentum = 0.9
decay = 0

modelpath = ""

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu])
np.random.seed(seed=random_state)

def loadnumpy(filename):
	array = np.load(filename)
	return array

def naiveReshape(X, target_pixel_size):
    X_out = np.zeros([X.shape[0], target_pixel_size, target_pixel_size, X.shape[-1]])
    for i in range(X.shape[0]):
        for ch in range(X.shape[-1]):
            X_out[i,:,:,ch]=X[i,:target_pixel_size,:target_pixel_size,ch]
    return X_out

def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
        return pv

def split_train_test(X_train, y_train, split):
    split_idx = int(round(y_train.shape[0]*split))
    indices = np.arange(0, y_train.shape[0])
    np.random.shuffle(indices)
    training, test = indices[:split_idx], indices[split_idx:]
    return X_train[training,:,:,:], y_train[training], X_train[test,:,:,:], y_train[test]


path_to_server_data = "/home/moritz_berthold/blasiData"

if server:
    h5f = h5py.File('/home/moritz_berthold/blasiData/X_blasi_original.h5', 'r')
    X = h5f['X'][()]
    y = h5f['y'][()]
    fnames = h5f['filenames'][()]
    h5f.close()

X_train_ex1 = np.zeros([X.shape[0], 66, 66, 2])
# np.unique(y, return_counts=True) = (array([0, 1, 2, 3, 4, 5, 6], dtype=uint8), array([   15, 14333,  8601,    68,   606,  8616,    27]))
print("Loading training and test data. Use ex1 for training and tuning and ex2 for testing.")
for i in range(X.shape[0]):
	X_train_ex1[i,16:48,16:48,:] = X[i,:,:,:]
y_train_ex1 = y
print("done")
print("Trainingdata shape = ", X_train_ex1.shape)
print("Traininglabels shape = ", y_train_ex1.shape)
X_train, y_train, X_test, y_test = split_train_test(X_train_ex1, y_train_ex1, split=split)

print("Reshaping done. Use Test, Train and Evaluation Data")
print(X_train.shape)
print(X_test.shape)

#### TRAINING ####

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a

def get_validation_predictions(train_data, predictions_valid):
    pv = []
    for i in range(len(train_data)):
        pv.append(predictions_valid[i])
    return pv

def run_cross_validation_create_models(nfolds, X_train, y_train):
    # input image dimensions
    train_data = X_train
    train_target = y_train

    yfull_train = dict()
    kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    accuracies = 0
    models = []
    for train_index, test_index in kf:
        model = deepflow(channels, n_classes, lr, momentum, decay)
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
        ]
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_data=(X_valid, Y_valid), callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        test_prediction = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        y_pred = np.zeros([test_prediction.shape[0]])
        for i in range(test_prediction.shape[0]):
            y_pred[i] = np.argmax(test_prediction[i,:]).astype(int)
        # plotConfusionMatrix(Y_valid.astype(int), y_pred.astype(int))

        scores = model.evaluate(X_valid.astype('float32'), Y_valid, verbose=0)
        print("Accuracy is: %.2f%%" % (scores[1]*100))
        accuracies += (scores[1]*100)


        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    final_accuracy = accuracies / nfolds
    print("Accuracy train independent avg: ", final_accuracy)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(epochs)
    return info_string, models

def process_test_with_cross_val(info_string, models, X_test):
    batch_size = 32
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data = X_test
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)

    print("Result on test data done: ", test_res.shape)
    return test_res

info_string, models = run_cross_validation_create_models(num_folds, X_train, y_train)
test_res = process_test_with_cross_val(info_string, models, X_test)

predictions_valid = model.predict(X_train.astype('float32'), batch_size=batch_size, verbose=2)
log_loss_train = log_loss(y_train, predictions_valid)
print('Score log_loss ex1: ', log_loss_train)
acc_train = model.evaluate(X_train.astype('float32'), y_train, verbose=0)
print("Score accuracy ex1: %.2f%%" % (acc_train[1]*100))

code.interact(local=dict(globals(), **locals()))
