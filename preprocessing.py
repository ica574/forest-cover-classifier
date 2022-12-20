# Name: preprocessing.py
# Date: 18/12/2022
# Author: Isaac Cilia Attard
# Description: Preprocesses and conducts data analysis on the provided dataset.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical

class Preprocessing:
    def __init__(self): # Loads dataset and splits into features and labels
        self.dataset = pd.read_csv("data/cover_data.csv") # Loads data from CSV file
        self.data = self.dataset[["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points","Wilderness_Area1","Wilderness_Area2","Wilderness_Area3","Wilderness_Area4","Soil_Type1","Soil_Type2","Soil_Type3","Soil_Type4","Soil_Type5","Soil_Type6","Soil_Type7","Soil_Type8","Soil_Type9","Soil_Type10","Soil_Type11","Soil_Type12","Soil_Type13","Soil_Type14","Soil_Type15","Soil_Type16","Soil_Type17","Soil_Type18","Soil_Type19","Soil_Type20","Soil_Type21","Soil_Type22","Soil_Type23","Soil_Type24","Soil_Type25","Soil_Type26","Soil_Type27","Soil_Type28","Soil_Type29","Soil_Type30","Soil_Type31","Soil_Type32","Soil_Type33","Soil_Type34","Soil_Type35","Soil_Type36","Soil_Type37","Soil_Type38","Soil_Type39","Soil_Type40"]]
        self.labels = self.dataset["class"]
        self.data_train, self.data_valid, self.labels_train, self.labels_valid = train_test_split(self.data, self.labels, test_size=0.20, stratify=self.labels, shuffle=True, random_state=22) # Splits data into training and validation data via scikit-learn

    def data_normaliser(self): # Normalises features
        self.transfomer = ColumnTransformer([("numeric", StandardScaler(), ["Elevation","Aspect","Slope","Horizontal_Distance_To_Hydrology","Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am","Hillshade_Noon","Hillshade_3pm","Horizontal_Distance_To_Fire_Points"])]) # Establishes a transformer to apply data normalisation
        self.data_train = self.transfomer.fit_transform(self.data_train) # Trains transformer on training data to normalise data
        self.data_valid = self.transfomer.transform(self.data_valid) # Normalises data after training the transformer
        return self.data_train, self.data_valid

    def label_vectoriser(self): # Transforms labels into vectors
        self.le = LabelEncoder() # Converts class labels into integers appropriate for the neural network
        self.labels_train = self.le.fit_transform(self.labels_train.astype(str)) # Fits the LabelEncoder() object to a training dataset
        self.labels_valid = self.le.transform(self.labels_valid.astype(str)) # Applies the transform to the validation dataset
        self.labels_train = to_categorical(self.labels_train) # Transforms training and validation datasets into binary vectors
        self.labels_valid = to_categorical(self.labels_valid)
        return self.labels_train, self.labels_valid