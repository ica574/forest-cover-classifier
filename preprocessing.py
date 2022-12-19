# Name: preprocessing.py
# Date: 18/12/2022
# Author: Isaac Cilia Attard
# Description: Preprocesses and conducts data analysis on the provided dataset.

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(): # Assimilates data from CSV file and seperates into data and labels
    data_frame = pd.read_csv("data/cover_data.csv") # Loads data from CSV file
    return data_frame.iloc[:,:-1], data_frame.iloc[:,-1] # Returns data and labels respectively

data, labels = load_data() # Assigns data and labels to respective variables
x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=22) # Splits data into training and validation data via scikit-learn