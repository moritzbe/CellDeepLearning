# implement a neural network for classification
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, ZeroPadding2D, MaxPooling2D, Dropout, Flatten
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

def fullyConnectedNet(X, y, epochs):
	neurons = 100
	nb_features = X.shape[1]
	model = Sequential([
		Dense(neurons, input_dim=nb_features, init='uniform'),
		Activation('relu'),
		Dense(neurons, init='uniform'),
		Activation('relu'),
		Dense(10, init='uniform'),
		Activation('softmax'),
	])

	# Compile model
	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# mse does not work
	# Fit the model
	model.fit(X, y, nb_epoch=epochs, batch_size=100)

	# evaluate the model
	scores = model.evaluate(X, y)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	return model

def covNet(X, y, batch_size = 128, epochs = 12):
	nb_filters = 64
	nb_classes = 10
	model = Sequential()
	model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(28, 28, 1)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(64, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(.25))

	model.add(Flatten())
	model.add(Dense(32))
	model.add(Activation('relu'))
	model.add(Dropout(.5))
	model.add(Dense(10))
	model.add(Activation('softmax'))

	model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	# original categorical_crossentropy and adadelta
	model.fit(X, y, batch_size=batch_size, nb_epoch=epochs, verbose=1)
	score = model.evaluate(X, y, verbose=0)
	print 'Train score:', score[0]
	print 'Train accuracy:', score[1]
	return model

