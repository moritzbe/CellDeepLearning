from data_tools import *
# from algorithms import *
from plot_lib import *
from netDeepFlow import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import KFold
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.models import load_model
from sklearn.metrics import log_loss
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
from scipy.stats import pearsonr, spearmanr
import _pickle as cPickle
import code
import os

server = True
train = True
modelsave = True
data_normalization = False
data_augmentation = True
gpu = [1]
batch_size = 32
epochs = 40
random_state = 17
channels = [0]
n_classes = 1
split = 0.9

### Optimizer ###
lr = 0.01
momentum = 0.9
decay = 0
change_epoch = 85
lr2 = 0.001
decay2 = 0.0005


class_names = ["RIIb+", "RIIb-"]
# class_names = ["CGRP+ RIIb+", "CGRP+ RIIb-", "CGRP- RIIb+", "CGRP- RIIb-"]


# modelpath = "/home/moritz_berthold/dl/cellmodels/deepflow/140617r2b+-_prediction_ch_[0]_bs=32_epochs=100_norm=False_split=0.9_lr1=0.01_momentum=0.9_decay1=0.0005_change_epoch=85_decay2=0.0005_lr2=0.001_acc1=[2.0498867458626939e-05, 1.0]_acc2=[0.30993385431939385, 0.94195119090921142].h5"

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
    return X_train[:split_idx,:,:,:], y_train[:split_idx], X_train[split_idx:,:,:,:], y_train[split_idx:]

def schedule(epoch):
    if epoch == change_epoch:
        K.set_value(model.optimizer.lr, lr2)
        K.set_value(model.optimizer.decay, decay2)
        print("Set new learning rate", K.get_value(model.optimizer.lr))
    return K.get_value(model.optimizer.lr)

path_to_server_data = "/home/moritz_berthold/cellData"

if not server:
    path_train_data = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_train_labels = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_66_66_full_no_zeros_in_cells.npy"
    path_train_data2 = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_train_labels2 = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/labels_66_66_full_no_zeros_in_cells.npy"
    path_test_data = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/PreparedData/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_test_labels = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex3/PreparedData/labels_66_66_full_no_zeros_in_cells.npy"

if server:
    path_train_data = path_to_server_data + "/Ex1/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_train_labels = path_to_server_data + "/Ex1/labels_66_66_full_no_zeros_in_cells.npy"
    path_train_data2 = path_to_server_data + "/Ex2/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_train_labels2 = path_to_server_data + "/Ex2/labels_66_66_full_no_zeros_in_cells.npy"
    path_test_data = path_to_server_data + "/Ex3/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_test_labels = path_to_server_data + "/Ex3/labels_66_66_full_no_zeros_in_cells.npy"


print("Loading training and test data. Use ex1 and ex2 for training and tuning and ex3 for testing.")
X_train_ex1 = np.array(loadnumpy(path_train_data), dtype = np.uint8).astype('float32')
y_train_ex1 = np.load(path_train_labels)[:,3]
labels_train_ex1 = np.load(path_train_labels)[:,0]
X_train_ex2 = np.array(loadnumpy(path_train_data2), dtype = np.uint8).astype('float32')
y_train_ex2 = np.load(path_train_labels2)[:,3]
labels_train_ex2 = np.load(path_train_labels)[:,0]
X_test_ex3 = np.array(loadnumpy(path_test_data), dtype = np.uint8).astype('float32')
y_test_ex3 = np.load(path_test_labels)[:,3]
labels_test_ex3 = np.load(path_test_labels)[:,0]
print("done")

X_train_ex1 = X_train_ex1[labels_train_ex1!=4,:]
y_train_ex1 = y_train_ex1[labels_train_ex1!=4]
labels_train_ex1 = labels_train_ex1[labels_train_ex1!=4]
X_train_ex2 = X_train_ex2[labels_train_ex2!=4,:]
y_train_ex2 = y_train_ex2[labels_train_ex2!=4]
labels_train_ex2 = labels_train_ex2[labels_train_ex2!=4]
X_test_ex3 = X_test_ex3[labels_test_ex3!=4,:]
y_test_ex3 = y_test_ex3[labels_test_ex3!=4]
labels_test_ex3 = labels_test_ex3[labels_test_ex3!=4]
features_test_ex3 = np.load(path_test_labels)[:,1:]
features_test_ex3 = features_test_ex3[labels_test_ex3!=4,:]
print("- removed the last class for comparison with cell profiler")




# sensible?
if data_normalization:
    pass

print("Ex1 data shape = ", X_train_ex1.shape)
print("Ex1 labels shape = ", y_train_ex1.shape)
print("Ex2 data shape = ", X_train_ex2.shape)
print("Ex2 labels shape = ", y_train_ex2.shape)
print("Ex3 data shape = ", X_test_ex3.shape)
print("Ex3 labels shape = ", y_test_ex3.shape)


X_train_ex1 = X_train_ex1.reshape(X_train_ex1.shape[0], X_train_ex1.shape[3], X_train_ex1.shape[2], X_train_ex1.shape[1])
X_train_ex2 = X_train_ex2.reshape(X_train_ex2.shape[0], X_train_ex2.shape[3], X_train_ex2.shape[2], X_train_ex2.shape[1])
X_test_ex3 = X_test_ex3.reshape(X_test_ex3.shape[0], X_test_ex3.shape[3], X_test_ex3.shape[2], X_test_ex3.shape[1])

print("Combining Ex1 and Ex2 for training:")
X_train_c = np.vstack([X_train_ex1, X_train_ex2])
y_train_c = np.append(y_train_ex1, y_train_ex2)

X_train, y_train, X_test, y_test = split_train_test(X_train_c, y_train_c, split=split)
print("Reshaping done. Use Test, Train and Evaluation Data")
print("Shape training data:", X_train.shape)
print("Shape training evaluation data:", X_test.shape)

print("Selecting channels:", channels)
X_train = X_train[:,:,:,channels]
X_test = X_test[:,:,:,channels]
X_test_ex3 = X_test_ex3[:,:,:,channels]


if data_augmentation:
    print("Data augmentation")
    X_train_l = X_train[:,:,::-1,:]
    X_train_u = X_train[:,::-1,:,:]
    X_train_lu = X_train_u[:,:,::-1,:]
    X_train = np.vstack([X_train, X_train_l, X_train_u, X_train_lu])
    y_train = np.append(y_train, np.append(y_train, np.append(y_train, y_train)))

path = "/home/moritz_berthold/dl/cellmodels/deepflow/120617/"

#### TRAINING ####
if train:
    model = deepregression(channels, n_classes, lr, momentum, decay)
    change_lr = LearningRateScheduler(schedule)
    csvlog = CSVLogger(path+'predict_r2b_intensity_based_on_PGP.csv', append=True)
    checkpoint = ModelCheckpoint(path+'checkpoints/'+ 'predict_r2b_intensity_based_on_PGP.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks = [
        change_lr,
        csvlog,
        checkpoint,
    ]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)

if not train:
    print("Loading trained model", modelpath)
    model = load_model(modelpath)

#### EVALUATION EX1 + Ex2 ####
print("The mean of exp. 1 is", np.mean(y_train))

predictions_valid = model.predict(X_train.astype('float32'), batch_size=batch_size, verbose=2)
rms2 = rms(y_train, predictions_valid)**.5
print('Train Root mean squared error: ', rms2)

#### EVALUATION EX3 ####
predictions_valid_test = model.predict(X_test_ex3.astype('float32'), batch_size=batch_size, verbose=2)
rms3 = rms(y_test_ex3, predictions_valid_test)**.5
print('Test Root mean squared error: ', rms3)
spearman = spearmanr(y_test_ex3, predictions_valid_test)
print("The Spearman correlation coefficient on exp. 3 is ", spearman)

ch_to_predict = "RIIb"

def detectPredictedLabel(y_test, y_pred):
    mean1 = y_test[0]
    mean2 = y_test[1]
    mean3 = y_pred
    mean4 = y_test[-1]
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

pred_label = np.zeros_like(labels_test_ex3)
for i in range(predictions_valid_test.shape[0]):
    pred_label[i] = detectPredictedLabel(features_test_ex3[i,:], predictions_valid_test[i])

def convertLabels(ground_truth):
    ground_truth[np.argwhere(ground_truth==0)] = 0 # +ve
    ground_truth[np.argwhere(ground_truth==2)] = 0 # +ve
    ground_truth[np.argwhere(ground_truth==1)] = 1 # -ve
    ground_truth[np.argwhere(ground_truth==3)] = 1 # -ve
    return ground_truth

labels_test_ex3 = convertLabels(labels_test_ex3)
pred_label = convertLabels(pred_label)

acc = accuracy(labels_test_ex3, pred_label)
print("Accuracy :", acc)
#### Saving Model ####
if train & modelsave:
    modelname = "/home/moritz_berthold/dl/cellmodels/deepflow/intensity_prediction_r2b_based_on_ch_" + str(channels) + "_bs=" + str(batch_size) + \
            "_epochs=" + str(epochs) + "_norm=" + str(data_normalization) + "_aug=" + str(data_augmentation) + "_split=" + str(split) + "_lr1=" + str(lr)  + \
            "_momentum=" + str(momentum)  + "_decay1=" + str(decay) +  \
            "_rms2=" + str(rms2)  + "_rms3=" + "_acc=" + str(acc) + ".h5"
    model.save(modelname)
    print("saved model")

code.interact(local=dict(globals(), **locals()))
plotBothConfusionMatrices(pred_label, labels_test_ex3, class_names)
