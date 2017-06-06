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
from sklearn.metrics import log_loss
from keras.callbacks import LearningRateScheduler
import keras.backend as K
import numpy as np
import code
import os

server = True
train = True
data_normalization = False
gpu = [0]
batch_size = 32
nb_epoch = 40
random_state = 17
channels = [0,1,2,3]
n_classes = 4
split = 0.8
# predict_classes = [0,1,2,3]

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

# def lr_scheduler(epoch):
#     if epoch == 85:
#         K.set_value(opt.lr, 0.001)
#         # K.set_value(self.model.optimizer.decay, 0.0005)
#     return K.get_value(opt.lr)


path_to_server_data = "/home/moritz_berthold/cellData"

if not server:
    path_train_data = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_train_labels = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_66_66_full_no_zeros_in_cells.npy"
    path_test_data = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_test_labels = "/Volumes/MoritzBertholdHD/CellData/Experiments/Ex2/PreparedData/labels_66_66_full_no_zeros_in_cells.npy"

if server:
    path_train_data = path_to_server_data + "/Ex1/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_train_labels = path_to_server_data + "/Ex1/labels_66_66_full_no_zeros_in_cells.npy"
    path_test_data = path_to_server_data + "/Ex2/all_channels_66_66_full_no_zeros_in_cells.npy"
    path_test_labels = path_to_server_data + "/Ex2/labels_66_66_full_no_zeros_in_cells.npy"


print("Loading training and test data. Use ex1 for training and tuning and ex2 for testing.")
X_train_ex1 = np.array(loadnumpy(path_train_data), dtype = np.uint8).astype('float32')
y_train_ex1 = np.load(path_train_labels)[:,0]
X_test_ex2 = np.array(loadnumpy(path_test_data), dtype = np.uint8).astype('float32')
y_test_ex2 = np.load(path_test_labels)[:,0]
print("done")

# sensible?
if data_normalization:
    maximum = float(np.max(X_train))
    X_train /= maximum
    X_test /= maximum

print("Trainingdata shape = ", X_train_ex1.shape)
print("Traininglabels shape = ", y_train_ex1.shape)
print("Testdata shape = ", X_test_ex2.shape)
print("Testlabels shape = ", y_test_ex2.shape)

X_train_ex1 = X_train_ex1.reshape(X_train_ex1.shape[0], X_train_ex1.shape[3], X_train_ex1.shape[2], X_train_ex1.shape[1])
X_test_ex2 = X_test_ex2.reshape(X_test_ex2.shape[0], X_test_ex2.shape[3], X_test_ex2.shape[2], X_test_ex2.shape[1])

X_train_ex1 = naiveReshape(X_train_ex1, target_pixel_size=66)
X_test_ex2 = naiveReshape(X_test_ex2, target_pixel_size=66)

X_train, y_train, X_test, y_test = split_train_test(X_train_ex1, y_train_ex1, split=split)

print("Reshaping done. Use Test, Train and Evaluation Data")
print(X_train.shape)
print(X_test.shape)

X_train = X_train[y_train!=4,:]
y_train = y_train[y_train!=4]
X_test = X_test[y_test!=4,:]
y_test = y_test[y_test!=4]
X_train_ex1 = X_train_ex1[y_train_ex1!=4,:]
y_train_ex1 = y_train_ex1[y_train_ex1!=4]
X_test_ex2 = X_test_ex2[y_test_ex2!=4,:]
y_test_ex2 = y_test_ex2[y_test_ex2!=4]
print("- removed the last class for comparison with cell profiler")
print("Selecting channels:", channels)
X_train = X_train[:,:,:,channels]
X_test = X_test[:,:,:,channels]
X_train_ex1 = X_train_ex1[:,:,:,channels]
X_test_ex2 = X_test_ex2[:,:,:,channels]


#### TRAINING ####
if train:
    model = deepflow(channels, n_classes)
    # change_lr = LearningRateScheduler(lr_scheduler)
    # change_decay = DecayRateScheduler(decay_scheduler)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, verbose=0),
        # change_lr,
        # change_decay,
    ]
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)
#### EVALUATION EX1 ####
if not train:
    pass
predictions_valid = model.predict(X_train_ex1.astype('float32'), batch_size=batch_size, verbose=2)
log_loss_train = log_loss(y_test_ex1, predictions_valid)
print('Score log_loss ex2: ', log_loss_train)
acc_train = model.evaluate(X_train_ex1.astype('float32'), y_test_ex1, verbose=0)
print("Score accuracy ex2: %.2f%%" % (acc_train[1]*100))
#### EVALUATION EX2 ####
predictions_valid = model.predict(X_test_ex2.astype('float32'), batch_size=batch_size, verbose=2)
log_loss_test = log_loss(y_test_ex2, predictions_valid)
print('Score log_loss ex2: ', log_loss_test)
acc_test = model.evaluate(X_test_ex2.astype('float32'), y_test_ex2, verbose=0)
print("Score accuracy ex2: %.2f%%" % (acc_test[1]*100))

code.interact(local=dict(globals(), **locals()))
