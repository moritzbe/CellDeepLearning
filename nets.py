# implement a neural network for classification
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import SGD
import numpy as np
import code
import tensorflow as tf

# def fullyConnectedNet(X, y, epochs):
# 	neurons = 100
# 	nb_features = X.shape[1]
# 	model = Sequential([
# 		Dense(neurons, input_dim=nb_features, init='uniform'),
# 		Activation('relu'),
# 		Dense(neurons, init='uniform'),
# 		Activation('relu'),
# 		Dense(10, init='uniform'),
# 		Activation('softmax'),
# 	])

# 	# Compile model
# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 	# mse does not work
# 	# Fit the model
# 	model.fit(X, y, nb_epoch=epochs, batch_size=100)

# 	# evaluate the model
# 	scores = model.evaluate(X, y)
# 	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# 	return model

# def covNet(X, y, batch_size = 128, epochs = 12):
# 	nb_filters = 64
# 	nb_classes = 10
# 	model = Sequential()
# 	model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(28, 28, 1)))
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(64, 3, 3))
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Convolution2D(64, 3, 3))
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Dropout(.25))

# 	model.add(Flatten())
# 	model.add(Dense(32))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(.5))
# 	model.add(Dense(10))
# 	model.add(Activation('softmax'))

# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 	# original categorical_crossentropy and adadelta
# 	model.fit(X, y, batch_size=batch_size, nb_epoch=epochs, verbose=1)
# 	score = model.evaluate(X, y, verbose=0)
# 	print 'Train score:', score[0]
# 	print 'Train accuracy:', score[1]
# 	return model


# INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC
# def covNetSimple(X, y, batch_size = 64, epochs = 12):




def covNetSimple():
	nb_classes = 4
	model = Sequential()
	model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(80, 80, 4)))
	# size (32, 80, 80)
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# size (32, 40, 40)
	model.add(Convolution2D(64, 3, 3))
	# size (64, 40, 40)
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	# size (64, 14, 14)
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dropout(.5))
	model.add(Dense(4))
	model.add(Activation('softmax'))

	# model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# original categorical_crossentropy and adadelta
	# model.fit(X, y, batch_size=64, nb_epoch=12, verbose=1)
	# score = model.evaluate(X, y, verbose=0)
	# print 'Train score:', score[0]
	# print 'Train accuracy:', score[1]
	sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')

	return model












# INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC
# def covNetExtended(X, y, batch_size = 64, epochs = 12):
# 	nb_classes = 8
# 	model = Sequential()
# 	model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(128, 128, 3)))
# 	# size (32, 128, 128)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(32, 3, 3, border_mode='same'))
# 	# size (32, 128, 128)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	# size (32, 64, 64)
# 	model.add(Convolution2D(64, 3, 3, border_mode='same'))
# 	# size (64, 64, 64)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(64, 3, 3, border_mode='same'))
# 	# size (64, 64, 64)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	# size (64, 32, 32)
# 	model.add(Convolution2D(128, 3, 3, border_mode='same'))
# 	# size (128, 32, 32)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(128, 3, 3, border_mode='same'))
# 	# size (128, 32, 32)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	# size (128, 16, 16)
# 	model.add(Dropout(.25))
# 	model.add(Flatten())
# 	model.add(Dense(300))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(.3))
# 	model.add(Dense(100))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(.5))
# 	model.add(Dense(8))
# 	model.add(Activation('softmax'))

# 	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 	# original categorical_crossentropy and adadelta
# 	model.fit(X, y, batch_size=batch_size, nb_epoch=epochs, verbose=1)
# 	score = model.evaluate(X, y, verbose=0)
# 	print 'Train score:', score[0]
# 	print 'Train accuracy:', score[1]
# 	return model

# def covNet3():
# 	nb_classes = 8
# 	model = Sequential()
# 	model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 32, 32), dim_ordering='th'))
# 	# size (32, 128, 128)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (32, 128, 128)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# 	# size (32, 64, 64)
# 	model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (64, 64, 64)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (64, 64, 64)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# 	# size (64, 32, 32)
# 	model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (128, 32, 32)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (128, 32, 32)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# 	# size (128, 16, 16)
# 	model.add(Dropout(.25))
# 	model.add(Flatten())
# 	model.add(Dense(200))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(.3))
# 	model.add(Dense(100))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(.5))
# 	model.add(Dense(8))
# 	model.add(Activation('softmax'))

# 	# model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 	# original categorical_crossentropy and adadelta
# 	# score = model.evaluate(X, y, verbose=0)

# 	sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
# 	model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')

# 	# print 'Train score:', score[0]
# 	# print 'Train accuracy:', score[1]
# 	# return model
# 	return model

# def create_model():
#     model = Sequential()
#     model.add(ZeroPadding2D((1, 1), input_shape=(3, 32, 32), dim_ordering='th'))
#     model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
#     model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#     model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

#     model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#     model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
#     model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
#     model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

#     model.add(Flatten())
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(8, activation='softmax'))

#     sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')

#     return model

# def covNet3Large():
# 	model = Sequential()
# 	model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3, 128, 128), dim_ordering='th'))
# 	# size (32, 128, 128)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(32, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (32, 128, 128)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# 	# size (32, 64, 64)
# 	model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (64, 64, 64)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(64, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (64, 64, 64)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# 	# size (64, 32, 32)
# 	model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (128, 32, 32)
# 	model.add(Activation('relu'))
# 	model.add(Convolution2D(128, 3, 3, border_mode='same', dim_ordering='th'))
# 	# size (128, 32, 32)
# 	model.add(Activation('relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
# 	# size (128, 16, 16)
# 	model.add(Dropout(.25))
# 	model.add(Flatten())
# 	model.add(Dense(200))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(.3))
# 	model.add(Dense(100))
# 	model.add(Activation('relu'))
# 	model.add(Dropout(.5))
# 	model.add(Dense(8))
# 	model.add(Activation('softmax'))

# 	# model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# 	# original categorical_crossentropy and adadelta
# 	# score = model.evaluate(X, y, verbose=0)

# 	sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
# 	model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy')

# 	# print 'Train score:', score[0]
# 	# print 'Train accuracy:', score[1]
# 	# return model
# 	return model
