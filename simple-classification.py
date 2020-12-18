import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy



#Create empty lists. One for training data and other for label data
train_labels = []
train_samples = []

# Generate dummy data
def simpleClassification():
	train_labels = []
	train_samples = []
	for i in range(100):
		random_yanger = randint(13,64)
		train_samples.append(random_yanger)
		train_labels.append(1)

		random_older = randint(65,100)
		train_samples.append(random_older)
		train_labels.append(0)
	for i in range(2000):
		random_yanger = randint(13,64)
		train_samples.append(random_yanger)
		train_labels.append(0)

		random_older = randint(65,100)
		train_samples.append(random_older)
		train_labels.append(1)

	train_labels = np.array(train_labels)
	train_samples = np.array(train_samples)
	train_labels, train_samples = shuffle(train_labels, train_samples)

	scaler = MinMaxScaler(feature_range=(0,1))
	scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

	## Checking devices

	## create model

	model = Sequential([
		Dense(units=16,input_shape=(1,), activation='relu'),
		Dense(units=32, activation='relu'),
		Dense(units=2, activation='softmax')
		])
	model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=100, verbose=2, shuffle=True)
if __name__=="__main__":
	simpleClassification()
