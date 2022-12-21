# Name: model.py
# Date: 20/12/2022
# Author: Isaac Cilia Attard
# Description: Trains and saves the deep learning model.

from preprocessing import Preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

"""Preprocessor class instantiation and feature and label preprocessing"""
preprocessor = Preprocessing()
preprocessor.data_normaliser()
preprocessor.label_vectoriser()

"""Retrieval of features and labels after preprocessing"""
data_train, data_valid = preprocessor.get_data()
labels_train, labels_valid = preprocessor.get_labels()

model = Sequential()
model.add(InputLayer(input_shape=(data_train.shape[1],))) # Input layer with number of neurons dependent upon dimensions of training data
model.add(Dense(12, activation='relu')) # Hidden layer that uses rectified linear unit as an activation function for each neuron
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='softmax')) # Output layer with 7 neurons, one for every possible classification that uses the softmax activation function

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compiles the model
model.fit(data_train, labels_train, epochs=3, batch_size=16) # Trains the model

model.save("model/classifier.h5") # Saves the trained model