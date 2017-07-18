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
import _pickle as cPickle
import code
import os

server = True
train = False
modelsave = False
data_normalization = False
data_augmentation = False
gpu = [3]
batch_size = 32
epochs = 100
random_state = 17
channels = [0,1]
n_classes = 4
split = 0.9

### Optimizer ###
lr = 0.01
momentum = 0.9
decay = 0
change_epoch = 85
lr2 = 0.001
decay2 = 0.0005


class_names = ["RIIb+", "RIIb-"]
def occlusion_heatmap(net, x, target, n_classes, square_length=7):
    x = np.swapaxes(x, 1,3)
    if (x.ndim != 4) or x.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(x.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

    num_classes = n_classes
    img = x[0].copy()
    bs, col, s0, s1 = x.shape

    heat_array = np.zeros((s0, s1))
    pad = square_length // 2 + 1
    x_occluded = np.zeros((s1, col, s0, s1), dtype=img.dtype)
    probs = np.zeros((s0, s1, num_classes))

    # generate occluded images
    for i in range(s0):
        # batch s1 occluded images for faster prediction
        for j in range(s1):
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded[j] = x_pad[:, pad:-pad, pad:-pad]
        y_proba = net.predict(np.swapaxes(x_occluded, 1,3), batch_size=batch_size, verbose=2)
        # code.interact(local=dict(globals(), **locals()))
        probs[i] = y_proba.reshape(s1, num_classes)

    # from predicted probabilities, pick only those of target class
    for i in range(s0):
        for j in range(s1):
            heat_array[i, j] = probs[i, j, target]
    return heat_array

# def convertLabels(ground_truth):
#     ground_truth[np.argwhere(ground_truth==0)] = 0 # +ve
#     ground_truth[np.argwhere(ground_truth==2)] = 0 # +ve
#     ground_truth[np.argwhere(ground_truth==1)] = 1 # -ve
#     ground_truth[np.argwhere(ground_truth==3)] = 1 # -ve
#     return ground_truth

### Optimizer ###
lr = 0.01
momentum = 0.9
decay = 0.0005



path = "/home/moritz_berthold/dl/cellmodels/deepflow/130717/"
checkpoint_path = path + "checkpoints/" + '4_way_clean_resize=0.125.hdf5'


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

path_to_server_data = "/home/moritz_berthold/cellData"


if server:
    path_test_data = path_to_server_data + "/Ex3/all_channels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"
    path_test_labels = path_to_server_data + "/Ex3/labels_66_66_full_no_zeros_in_cells_no_bleed_trough_shifted.npy"


X_test_ex3 = np.array(loadnumpy(path_test_data)).astype('float32')
# y_test_ex3 = convertLabels(np.load(path_test_labels)[:,0])
y_test_ex3 = np.load(path_test_labels)[:,0]

print("Ex3 data shape = ", X_test_ex3.shape)
print("Ex3 labels shape = ", y_test_ex3.shape)

# X_test_ex3_ = np.reshape(X_test_ex3, (X_test_ex3.shape[0], X_test_ex3.shape[3], X_test_ex3.shape[2], X_test_ex3.shape[1]))
X_test_ex3 = np.swapaxes(X_test_ex3, 1,3)

X_test_ex3 = X_test_ex3[y_test_ex3!=4,:]
y_test_ex3 = y_test_ex3[y_test_ex3!=4]
print("- removed the last class for comparison with cell profiler")
print("Selecting channels:", channels)
X_test_ex3 = X_test_ex3[:,:,:,channels]




# model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2, validation_data=(X_test, y_test), callbacks=callbacks)

if not train:
    print("Loading trained model", checkpoint_path)
    model = load_model(checkpoint_path)

sample_index = 1
true_label = y_test_ex3[sample_index]

heat_array = occlusion_heatmap(model, X_test_ex3[sample_index:sample_index+1,:,:,:], true_label, len(class_names), square_length=7)
code.interact(local=dict(globals(), **locals()))

plt.imshow(heat_array)
plt.show()


maximum = np.max(heat_array)
minimum = np.min(heat_array)
new_heat = ((heat_array - minimum)) / (maximum - minimum + 0.000001)
plt.imshow(new_heat)
plt.show()
code.interact(local=dict(globals(), **locals()))
