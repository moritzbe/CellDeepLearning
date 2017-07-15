# implement a neural network for classification
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization as BatchNorm
from keras.layers import Activation, Dense, Input, concatenate
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
import numpy as np
from keras.utils import plot_model
import code
# Nick and Phils Training Details:
# The network was trained for 100 epochs using stochastic gradient descent
# with standard 318 parameters: 0.9 momentum, a fixed learning rate of 0.01
# up to epoch 85 and of 0.001 319 afterwards as well as a slightly regularizing
# weight decay of 0.0005.

# Basic Conv + BN + ReLU factory
# padding same - > same output size, padding is added according to kernel
def convFactory(data, num_filter, kernel, stride=(1,1), pad="valid", act_type="relu"): # valid
	conv = Conv2D(filters=num_filter, kernel_size=kernel, strides=stride, padding=pad)(data)
	bn = BatchNorm(axis=-1)(conv)
	act = Activation(act_type)(bn)
	return act

# A Simple Downsampling Factory
# pixel dimensions = ((a+2)/2,(a+2)/2)
def downsampleFactory(data, ch_3x3):
	# conv 3x3
	conv = convFactory(data=data, num_filter = ch_3x3, kernel=(3, 3), stride=(2, 2), pad="same", act_type="relu") # same
	pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None)(data) # its actually valid!
	# got shapes [(None, 33, 33, 80), (None, 32, 32, 80)]
	concat = concatenate([conv, pool], axis=-1)
	return concat

# A Simple module
def simpleFactory(data, ch_1x1, ch_3x3):
	# 1x1
	conv1x1 = convFactory(data=data, num_filter = ch_1x1, kernel=(1, 1), pad="valid", act_type="relu") # valid
	# 3x3
	conv3x3 = convFactory(data=data, num_filter = ch_3x3, kernel=(3, 3), pad="same", act_type="relu") # same
	#concat
	concat = concatenate([conv1x1, conv3x3], axis=-1)
	return concat

def deepflow(channels, n_classes, lr, momentum, decay, resize):
	# model = deepflow([1,2,3,4], 4, .01, .09, .0005)
	resize_factor = resize
	n_channels = len(channels)
	inputs = Input(shape=(66, 66, n_channels)) # 66x66
	conv1 = convFactory(data=inputs, num_filter=96 , kernel=(3,3), pad="same", act_type="relu") # same
	in3a = simpleFactory(conv1, 32, int(round(32*resize_factor)))
	in3b = simpleFactory(in3a, 32, int(round(48*resize_factor)))
	in3c = downsampleFactory(in3b, int(round(80*resize_factor))) # 33x33
	in4a = simpleFactory(in3c, 112, int(round(48*resize_factor)))
	in4b = simpleFactory(in4a, 96, int(round(64*resize_factor)))
	in4c = simpleFactory(in4b, 80, int(round(80*resize_factor)))
	in4d = simpleFactory(in4c, 48, int(round(96*resize_factor)))
	in4e = downsampleFactory(in4d, int(round(96*resize_factor))) # 17x17
	in5a = simpleFactory(in4e, 176, int(round(160*resize_factor)))
	in5b = simpleFactory(in5a, 176, int(round(160*resize_factor)))
	in6a = downsampleFactory(in5b, int(round(96*resize_factor))) # 8x8
	in6b = simpleFactory(in6a, 176, int(round(160*resize_factor)))
	in6c = simpleFactory(in6b, 176, int(round(160*resize_factor)))
	pool = AveragePooling2D(pool_size=(9, 9), strides=None, padding='valid', data_format=None)(in6c) # valid
	flatten = Flatten()(pool)
	# dropout = Dropout(drop_out, noise_shape=None, seed=17)(flatten)
	fc = Dense(n_classes, activation=None, name="last-layer-activations")(flatten)
	softmax = Activation(activation="softmax")(fc)
	model = Model(inputs=inputs, outputs=softmax)
	optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def deepregression(channels, n_classes, lr, momentum, decay, resize):
	# model = deepflow([1,2,3,4], 4, .01, .09, .0005)
	resize_factor = resize
	n_channels = len(channels)
	inputs = Input(shape=(66, 66, n_channels)) # 66x66
	conv1 = convFactory(data=inputs, num_filter=96 , kernel=(3,3), pad="same", act_type="relu") # same
	in3a = simpleFactory(conv1, 32, int(round(32*resize_factor)))
	in3b = simpleFactory(in3a, 32, int(round(48*resize_factor)))
	in3c = downsampleFactory(in3b, int(round(80*resize_factor))) # 33x33
	in4a = simpleFactory(in3c, 112, int(round(48*resize_factor)))
	in4b = simpleFactory(in4a, 96, int(round(64*resize_factor)))
	in4c = simpleFactory(in4b, 80, int(round(80*resize_factor)))
	in4d = simpleFactory(in4c, 48, int(round(96*resize_factor)))
	in4e = downsampleFactory(in4d, int(round(96*resize_factor))) # 17x17
	in5a = simpleFactory(in4e, 176, int(round(160*resize_factor)))
	in5b = simpleFactory(in5a, 176, int(round(160*resize_factor)))
	in6a = downsampleFactory(in5b, int(round(96*resize_factor))) # 8x8
	in6b = simpleFactory(in6a, 176, int(round(160*resize_factor)))
	in6c = simpleFactory(in6b, 176, int(round(160*resize_factor)))
	pool = AveragePooling2D(pool_size=(9, 9), strides=None, padding='valid', data_format=None)(in6c) # valid
	flatten = Flatten()(pool)
	# dropout = Dropout(drop_out, noise_shape=None, seed=17)(flatten)
	fc = Dense(n_classes, activation=None, name="last-layer-activations")(flatten) # move up, before fully connected layer!
	# possible batch normalization
	softmax = Activation(activation="linear")(fc)
	model = Model(inputs=inputs, outputs=softmax)
	optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)
	model.compile(loss="mse", optimizer="adam")
	return model

def showModel(model, name): #only works locally showModel(model, name = "pool9")
	plot_model(model, show_shapes=True, to_file="model_visualisation/" + name + ".png")

# ch = [0]
# model = deepregression(ch, 1, .1, .9, .9, .1)
# code.interact(local=dict(globals(), **locals()))
# model.summary()
