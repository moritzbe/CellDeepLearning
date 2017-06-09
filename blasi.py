# from data_tools import *
# from algorithms import *
# from plot_lib import *
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

server = True
train = False
modelsave = False
data_normalization = False
gpu = [1]
batch_size = 32
epochs = 100
random_state = 17
channels = [0,1]
n_classes = 7
split = 0.8
# predict_classes = [0,1,2,3]

### Optimizer ###
lr = 0.01
momentum = 0.9
decay = 0
change_epoch = 85
lr2 = 0.001
decay2 = 0.0005


modelpath = "/home/moritz_berthold/dl/cellmodels/blasi/ch_[0, 1]_bs=32_epochs=100_norm=False_split=0.8_lr1=0.01_momentum=0.9_decay1=0_change_epoch=85_decay2=0.0005_lr2=0.001_acc1=[0.30349450050783999, 0.95775739168164631]_acc2=[1.5066425106198122, 0.78772854049544905].h5"

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

def schedule(epoch):
    if epoch == change_epoch:
        K.set_value(model.optimizer.lr, lr2)
        K.set_value(model.optimizer.decay, decay2)
        print("Set new learning rate", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

def show_cell_image(X, y, fnames):
    plt.imshow(X[-1,:,:,1])
    plt.show()
    code.interact(local=dict(globals(), **locals()))

if server:
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

# sensible?
if data_normalization:
    pass


path = "/home/moritz_berthold/dl/cellmodels/blasi/080617/"

#### TRAINING ####
if train:
    model = deepflow(channels, n_classes, lr, momentum, decay)
    change_lr = LearningRateScheduler(schedule)
    csvlog = CSVLogger(path+'_train_log.csv', append=True)
    checkpoint = ModelCheckpoint(path+'checkpoints/'+ 'checkpoint.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks = [
        change_lr,
        csvlog,
        checkpoint,
    ]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)

#### EVALUATION ####
if not train:
    print("Loading trained model", modelpath)
    model = load_model(modelpath)

predictions_valid = model.predict(X_train_ex1.astype('float32'), batch_size=batch_size, verbose=2)
log_loss_train = log_loss(y_train_ex1, predictions_valid)
print('Score log_loss train: ', log_loss_train)
acc_train = model.evaluate(X_train_ex1.astype('float32'), y_train_ex1, verbose=0)
print("Score accuracy train: %.2f%%" % (acc_train[1]*100))
acc_test = model.evaluate(X_test_2.astype('float32'), y_test_2, verbose=0)
print("Score accuracy test: %.2f%%" % (acc_test[1]*100))



#### Saving Model ####
if train & modelsave:
    modelname = "/home/moritz_berthold/dl/cellmodels/blasi/ch_" + str(channels) + "_bs=" + str(batch_size) + \
            "_epochs=" + str(epochs) + "_norm=" + str(data_normalization) + "_split=" + str(split) + "_lr1=" + str(lr)  + \
            "_momentum=" + str(momentum)  + "_decay1=" + str(decay) +  \
            "_change_epoch=" + str(change_epoch) + "_decay2=" + str(decay2) + \
            "_lr2=" + str(lr2)  + "_acc1=" + str(acc_train) + "_acc2=" + str(acc_test) + ".h5"
    model.save(modelname)
    print("saved model")

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


def plotNiceConfusionMatrix(y_test, y_pred, class_names, rel=False):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    fig = plt.figure()
    conf_M(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    fig.set_tight_layout(True)
    plt.show()

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

def accuracy(y_test, pred):
    rights = 0
    for i in range(len(y_test)):
        if y_test[i] == pred[i]:
            rights += 1
    accuracy = float(rights) / float(len(y_test))
    return round(accuracy, 4)*100

predictions_valid_2 = model.predict(X_test_2.astype('float32'), batch_size=batch_size, verbose=2)
predictions_valid_2 = np.argmax(predictions_valid_2, axis=1)



class_names = ["1","2","3","4","5","6","7"]
class_names2 = ["G1,G2,S","Pro","Meta","Ana","Telo"]
y_test_2_reduced = reduceClasses(y_test_temp)
predictions_valid_2_reduced = reduceClasses(predictions_valid_2)

# plotNiceConfusionMatrix(predictions_valid_2, y_test_2, class_names, rel=False)
# plotNiceConfusionMatrix(predictions_valid_2_reduced, y_test_2_reduced, class_names2, rel=False)
plotNiceConfusionMatrix(y_test_2_reduced, predictions_valid_2_reduced, class_names2, rel=False)
print("Reduced classes test accuracy is: ", accuracy(y_test_2_reduced, predictions_valid_2_reduced))
code.interact(local=dict(globals(), **locals()))
