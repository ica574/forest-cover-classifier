# Name: preprocessing.py
# Date: 18/12/2022
# Author: Isaac Cilia Attard
# Description: Preprocesses and conducts data analysis on the provided dataset.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from sklearn.metrics import classification_report

def load_data(): # Assimilates data from CSV file and seperates into data and labels
    data_frame = pd.read_csv("data/cover_data.csv") # Loads data from CSV file
    return data_frame[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4","Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40"]], data_frame["class"] # Returns data and labels respectively

data, labels = load_data() # Assigns data and labels to respective variables
data_train, data_valid, labels_train, labels_valid = train_test_split(data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=22) # Splits data into training and validation data via scikit-learn

transfomer = ColumnTransformer([("numeric", StandardScaler(), ["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"])]) # Establishes a transformer to apply data normalisation
data_train = transfomer.fit_transform(data_train) # Trains transformer on training data to normalise data
data_valid = transfomer.transform(data_valid) # Normalises data after training the transformer

le = LabelEncoder() # Converts class labels into integers appropriate for the neural network
labels_train = le.fit_transform(labels_train.astype(str)) # Fits the LabelEncoder() object to a training dataset
labels_valid = le.transform(labels_valid.astype(str)) # Applies the transform to the validation dataset

labels_train = to_categorical(labels_train) # Transforms training and validation datasets into binary vectors
labels_valid = to_categorical(labels_valid)

"""Neural network model"""
model = Sequential()
model.add(InputLayer(input_shape=(data_train.shape[1],))) # Input layer with number of neurons dependent upon dimensions of training data
model.add(Dense(12, activation='relu')) # Hidden layer that uses rectified linear unit as an activation function for each neuron
model.add(Dense(7, activation='softmax')) # Output layer with 7 neurons, one for every possible classification that uses the softmax activation function

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # Compiles the model
model.fit(data_train, labels_train, epochs=3, batch_size=16) # Trains the model