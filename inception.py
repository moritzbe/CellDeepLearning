# coding: utf-8
from data_tools import *
from algorithms import *
from plot_lib import *
import matplotlib.pyplot as plt
# from nets import *
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import RFE
from sklearn import preprocessing
import numpy as np
import code 
import caffe
import scipy as nd

# Container for the extraced features:
X = np.empty([0, 1000])

print "Loading images and labels..."
# Loading the data as images
DATA = np.load("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full_no_zeros_in_cells.npy")
labels = np.load("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/labels_80_80_full_no_zeros_in_cells.npy")
print "all loaded."

print 


print "Loading model AlexNet..."
#load the model
net = caffe.Net('ModelZoo/bvlc_alexnet/deploy.prototxt',
                'ModelZoo/bvlc_alexnet/bvlc_alexnet.caffemodel',
                caffe.TEST)
print "all loaded."


#since we classify only one image, we change batch size from 10 to 1
net.blobs['data'].reshape(1,3,227,227)

# Normalizing DATA and fitting to 255
DATA /= np.max(DATA)
DATA *= 255.

# Reshaping the data to fit AlexNet Model 227 X 227
# discard channel 4

print "AlexNet performing inception on images"

for i in xrange(DATA.shape[0]):

    single_image = np.zeros([3,227,227])
    single_image[0,:,:] = nd.misc.imresize(DATA[i,0,:,:], [227, 227], interp='bilinear', mode=None)
    single_image[1,:,:] = nd.misc.imresize(DATA[i,1,:,:], [227, 227], interp='bilinear', mode=None)
    single_image[2,:,:] = nd.misc.imresize(DATA[i,2,:,:], [227, 227], interp='bilinear', mode=None)

    #compute
    net.blobs['data'].data[...] = single_image
    out = net.forward()
    X = np.vstack((X, out["fc8"][0]))
    if i % 100 == 0:
        print "Image no.", i
        
print "Done!"
print "The shape of X is:", X.shape


code.interact(local=dict(globals(), **locals()))

np.save("/Volumes/MoritzBertholdHD/CellData/Experiments/Ex1/PreparedData/all_channels_80_80_full_no_zeros_in_cells_alexnet_inception_fc8", X, allow_pickle=True, fix_imports=True)




