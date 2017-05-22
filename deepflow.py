from data_tools import *
from algorithms import *
from plot_lib import *
from netDeepFlow import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.metrics import log_loss
import numpy as np
import tensorflow as tf
import code

data_normalization = False


# Paths
path_train_data = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full_no_zeros_in_cells.npy"
path_train_labels = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_80_80_full_no_zeros_in_cells.npy"
path_test_data = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/all_channels_80_80_full_no_zeros_in_cells.npy"
path_test_labels = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/labels_80_80_full_no_zeros_in_cells.npy"

print "Loading training and test data"
X_train = np.array(loadnumpy(path_train_data), dtype = np.uint8).astype('float32')
y_train = np.load(path_train_labels)[:,0]
X_test = np.array(loadnumpy(path_test_data), dtype = np.uint8).astype('float32')
y_test = np.load(path_test_labels)[:,0]
print "done"

# sensible?
if data_normalization:
    maximum = float(np.max(X_train))
    X_train /= maximum
    X_test /= maximum

print "Trainingdata shape = ", X_train.shape
print "Traininglabels shape = ", y_train.shape
print "Testdata shape = ", X_test.shape
print "Testlabels shape = ", y_test.shape

X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], X_train.shape[2], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], X_test.shape[2], X_test.shape[1])
print "Reshaping done"

print X_train.shape
print X_test.shape

X_train = X_train[y_train!=4, :]
y_train = y_train[y_train!=4]
X_test = X_test[y_test!=4, :]
y_test = y_test[y_test!=4]
print "- removed the last class for comparison with cell profiler"


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

def run_cross_validation_create_models(nfolds, X_train, X_test, y_train):
    # input image dimensions
    batch_size = 1
    nb_epoch = 1
    random_state = 51

    train_data = X_train
    train_target = y_train

    yfull_train = dict()
    kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    accuracies = 0
    models = []
    for train_index, test_index in kf:
        model = deepflow()
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
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=2, validation_data=(X_valid, Y_valid), callbacks=callbacks)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        test_prediction = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        y_pred = np.zeros([test_prediction.shape[0]])
        for i in xrange(test_prediction.shape[0]):
            y_pred[i] = np.argmax(test_prediction[i,:]).astype(int)
        plotConfusionMatrix(Y_valid.astype(int), y_pred.astype(int))

        scores = model.evaluate(X_valid.astype('float32'), Y_valid, verbose=0)
        # print("Accuracy is: %.2f%%" % (scores[1]*100))
        # accuracies += (scores[1]*100)


        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    print "No accuracy computed"
    final_accuracy = accuracies / nfolds
    print("Accuracy train independent avg: ", final_accuracy)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch)
    return info_string, models

def process_test_with_cross_val(info_string, models, X_test):
    batch_size = 16
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

    print "Result on test data done: ", test_res.shape
    return test_res

num_folds = 3
info_string, models = run_cross_validation_create_models(num_folds, X_train, X_test, y_train)
test_res = process_test_with_cross_val(info_string, models, X_test)
