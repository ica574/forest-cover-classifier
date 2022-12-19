# Name: preprocessing.py
# Date: 18/12/2022
# Author: Isaac Cilia Attard
# Description: Preprocesses and conducts data analysis on the provided dataset.

import pandas as pd

def load_data(): # Assimilates data from CSV file and seperates into data and labels
    data_frame = pd.read_csv("data/cover_data.csv") # Loads data from CSV file
    return data_frame.iloc[:,:-1], data_frame.iloc[:,-1] # Returns data and labels respectively

data, labels = load_data() # Assigns data and labels to respective variables