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
import tensorflow as tf

# Basic Conv + BN + ReLU factory
# padding same - > same output size, padding is added according to kernel
def convFactory(data, num_filter, kernel, stride=(1,1), pad="valid", act_type="relu"):
	conv = Conv2D(filters=num_filter, kernel_size=kernel, strides=stride, padding=pad)(data)
	bn = BatchNorm(axis=-1)(conv)
	act = Activation(act_type)(bn)
	return act

# A Simple Downsampling Factory
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

def deepflow():
	inputs = Input(shape=(60, 60, 4))
	conv1 = convFactory(data=inputs, num_filter=96 , kernel=(3,3), pad="same", act_type="relu")
	in3a = simpleFactory(conv1, 32, 32)
	in3b = simpleFactory(in3a, 32, 48)
	in3c = downsampleFactory(in3b, 80)
	in4a = simpleFactory(in3c, 112, 48)
	in4b = simpleFactory(in4a, 96, 64)
	in4c = simpleFactory(in4b, 80, 80)
	in4d = simpleFactory(in4c, 48, 96)
	in4e = downsampleFactory(in4d, 96)
	in5a = simpleFactory(in4e, 176, 160)
	in5b = simpleFactory(in5a, 176, 160)
	in6a = downsampleFactory(in5b, 96)
	in6b = simpleFactory(in6a, 176, 160)
	in6c = simpleFactory(in6b, 176, 160)
	# pool = mx.symbol.Pooling(data=in6c, pool_type="avg", kernel=(8,8), name="global_avg")
	pool = AveragePooling2D(pool_size=(8, 8), strides=None, padding='valid', data_format=None)(in6c)
	flatten = Flatten()(pool)
	fc = Dense(7, activation=None)(flatten)
	softmax = Activation(activation="softmax")(fc)
	model = Model(inputs=inputs, outputs=softmax)
	model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=0.2)
	# model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model
model = deepflow()
