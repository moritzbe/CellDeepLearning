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
import code
# Nick and Phil´s Training Details:
# The network was trained for 100 epochs using stochastic gradient descent
# with standard 318 parameters: 0.9 momentum, a fixed learning rate of 0.01
# up to epoch 85 and of 0.001 319 afterwards as well as a slightly regularizing
# weight decay of 0.0005.

# Basic Conv + BN + ReLU factory
# padding same - > same output size, padding is added according to kernel
def convFactory(data, num_filter, kernel, stride=(1,1), pad="valid", act_type="relu"):
	conv = Conv2D(filters=num_filter, kernel_size=kernel, strides=stride, padding=pad)(data)
	bn = BatchNorm(axis=-1)(conv)
	act = Activation(act_type)(bn)
	return act

# A Simple Downsampling Factory
# pixel dimensions = ((a+2)/2,(a+2)/2)
def downsampleFactory(data, ch_3x3):
	# conv 3x3
	conv = convFactory(data=data, num_filter = ch_3x3, kernel=(3, 3), stride=(2, 2), pad="same", act_type="relu")
	# original: pool = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type='max'), default padding is (0,0)
	pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format=None)(data)
	# concat
	concat = concatenate([conv, pool], axis=-1)
	return concat


# A Simple module
def simpleFactory(data, ch_1x1, ch_3x3):
	# 1x1
	conv1x1 = convFactory(data=data, num_filter = ch_1x1, kernel=(1, 1), pad="valid", act_type="relu")
	# 3x3
	conv3x3 = convFactory(data=data, num_filter = ch_3x3, kernel=(3, 3), pad="same", act_type="relu")
	#concat
	concat = concatenate([conv1x1, conv3x3], axis=-1)
	return concat

def deepflow(channels, n_classes):
	n_channels = len(channels)
	inputs = Input(shape=(66, 66, n_channels)) # 66x66
	conv1 = convFactory(data=inputs, num_filter=96 , kernel=(3,3), pad="same", act_type="relu")
	in3a = simpleFactory(conv1, 32, 32)
	in3b = simpleFactory(in3a, 32, 48)
	in3c = downsampleFactory(in3b, 80) # 34x34
	in4a = simpleFactory(in3c, 112, 48)
	in4b = simpleFactory(in4a, 96, 64)
	in4c = simpleFactory(in4b, 80, 80)
	in4d = simpleFactory(in4c, 48, 96)
	in4e = downsampleFactory(in4d, 96) # 17x17
	in5a = simpleFactory(in4e, 176, 160)
	in5b = simpleFactory(in5a, 176, 160)
	in6a = downsampleFactory(in5b, 96) # 8x8
	in6b = simpleFactory(in6a, 176, 160)
	in6c = simpleFactory(in6b, 176, 160)
	pool = AveragePooling2D(pool_size=(8, 8), strides=None, padding='same', data_format=None)(in6c)
	flatten = Flatten()(pool)
	fc = Dense(n_classes, activation=None)(flatten)
	softmax = Activation(activation="softmax")(fc)
	model = Model(inputs=inputs, outputs=softmax)
	optimizer = SGD(lr=0.01, momentum=0.9, decay=0, nesterov=False)
	model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model
