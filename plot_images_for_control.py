from data_tools import *
# from algorithms import *
from plot_lib import *

import numpy as np
from scipy.stats import pearsonr, spearmanr
import _pickle as cPickle
import code
import os


# clean model
modelpath = "/home/moritz_berthold/dl/cellmodels/deepflow/intensity_prediction_r2b_nbts_based_on_ch_[0]_bs=32_epochs=45_norm=False_aug=False_split=0.9_lr1=0.01_momentum=0.9_decay1=0_rms2=63.1048360822_rms3=_acc=94.51.h5"
# dirty model
# modelpath = "/home/moritz_berthold/dl/cellmodels/deepflow/intensity_prediction_r2b_based_on_ch_[0]_bs=32_epochs=40_norm=False_aug=True_split=0.9_lr1=0.01_momentum=0.9_decay1=0_rms2=179.18797775_rms3=_acc=91.52.h5"
prevent_bleed_through = True
save_outcomes = True
save_name = "result_deep_regression_shifted_45_eps_cm_cd_rop=.3" #clean model = cm dirty data = dd
dropout = .3
server = True
train = True
modelsave = True
data_normalization = False
data_augmentation = False
gpu = [2]
batch_size = 32
epochs = 60
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
    if prevent_bleed_through:
        print("Prevent Bleed Through")
        if train:
            path_train_data = path_to_server_data + "/Ex1/all_channels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"
            path_train_labels = path_to_server_data + "/Ex1/labels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"
            path_train_data2 = path_to_server_data + "/Ex2/all_channels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"
            path_train_labels2 = path_to_server_data + "/Ex2/labels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"
        path_test_data = path_to_server_data + "/Ex3/all_channels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"
        path_test_labels = path_to_server_data + "/Ex3/labels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"
    else:
        if train:
            path_train_data = path_to_server_data + "/Ex1/all_channels_66_66_full_no_zeros_in_cells.npy"
            path_train_labels = path_to_server_data + "/Ex1/labels_66_66_full_no_zeros_in_cells.npy"
            path_train_data2 = path_to_server_data + "/Ex2/all_channels_66_66_full_no_zeros_in_cells.npy"
            path_train_labels2 = path_to_server_data + "/Ex2/labels_66_66_full_no_zeros_in_cells.npy"
        path_test_data = path_to_server_data + "/Ex3/all_channels_66_66_full_no_zeros_in_cells.npy"
        path_test_labels = path_to_server_data + "/Ex3/labels_66_66_full_no_zeros_in_cells.npy"


print("Loading training and test data. Use ex1 and ex2 for training and tuning and ex3 for testing.")
if train:
    X_train_ex1 = np.array(loadnumpy(path_train_data), dtype = np.uint8).astype('float32')
    y_train_ex1 = np.load(path_train_labels)[:,3]
    labels_train_ex1 = np.load(path_train_labels)[:,0]
    X_train_ex2 = np.array(loadnumpy(path_train_data2), dtype = np.uint8).astype('float32')
    y_train_ex2 = np.load(path_train_labels2)[:,3]
    labels_train_ex2 = np.load(path_train_labels2)[:,0]
X_test_ex3 = np.array(loadnumpy(path_test_data), dtype = np.uint8).astype('float32')
y_test_ex3 = np.load(path_test_labels)[:,3]
labels_test_ex3 = np.load(path_test_labels)[:,0]
print("done")

if train:
    X_train_ex1 = X_train_ex1[labels_train_ex1!=4,:]
    y_train_ex1 = y_train_ex1[labels_train_ex1!=4]
    labels_train_ex1 = labels_train_ex1[labels_train_ex1!=4]
    X_train_ex2 = X_train_ex2[labels_train_ex2!=4,:]
    y_train_ex2 = y_train_ex2[labels_train_ex2!=4]
    labels_train_ex2 = labels_train_ex2[labels_train_ex2!=4]
X_test_ex3 = X_test_ex3[labels_test_ex3!=4,:]
y_test_ex3 = y_test_ex3[labels_test_ex3!=4]
features_test_ex3 = np.load(path_test_labels)[:,1:]
features_test_ex3 = features_test_ex3[labels_test_ex3!=4,:]
labels_test_ex3 = labels_test_ex3[labels_test_ex3!=4]
print("- removed the last class for comparison with cell profiler")




# sensible?
if data_normalization:
    pass
if train:
    print("Ex1 data shape = ", X_train_ex1.shape)
    print("Ex1 labels shape = ", y_train_ex1.shape)
    print("Ex2 data shape = ", X_train_ex2.shape)
    print("Ex2 labels shape = ", y_train_ex2.shape)
print("Ex3 data shape = ", X_test_ex3.shape)
print("Ex3 labels shape = ", y_test_ex3.shape)
code.interact(local=dict(globals(), **locals()))

if train:
    # X_train_ex1 = X_train_ex1.reshape(X_train_ex1.shape[0], X_train_ex1.shape[3], X_train_ex1.shape[2], X_train_ex1.shape[1])
    # X_train_ex2 = X_train_ex2.reshape(X_train_ex2.shape[0], X_train_ex2.shape[3], X_train_ex2.shape[2], X_train_ex2.shape[1])
    X_train_ex1 = np.swapaxes(X_train_ex1, 1,3)
    X_train_ex2 = np.swapaxes(X_train_ex2, 1,3)



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

X_test_ex3 = np.swapaxes(X_test_ex3, 1,3)
# X_test_ex3 = X_test_ex3.reshape(X_test_ex3.shape[0], X_test_ex3.shape[3], X_test_ex3.shape[2], X_test_ex3.shape[1])
X_test_ex3 = X_test_ex3[:,:,:,channels]




# plotBothConfusionMatrices(pred_label, labels_test_ex3, class_names)
