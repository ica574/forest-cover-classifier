# Name: predictor.py
# Date: 20/12/2022
# Author: Isaac Cilia Attard
# Description: Uses the trained model to perform multi-class classification.

from preprocessing import Preprocessing
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
import numpy as np

preprocessor = Preprocessing()

data_train, data_valid = preprocessor.data_normaliser()
labels_train, labels_valid = preprocessor.label_vectoriser()

model = load_model("model/classifier.h5")

prediction_estimate = model.predict(data_valid) # Uses the trained model to generate predictions
prediction_estimate = np.argmax(prediction_estimate, axis=1) # Returns only the most probable predictions from every label
actual_values = np.argmax(labels_valid, axis=1)

print(classification_report(actual_values, prediction_estimate)) # Prints a classification report with results of the predictions