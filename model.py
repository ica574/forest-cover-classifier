# Name: model.py
# Date: 20/12/2022
# Author: Isaac Cilia Attard
# Description: Finds optimal hyperparameters, trains, and saves the deep learning model.

from preprocessing import Preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

"""Preprocessor class instantiation and feature and label preprocessing"""
preprocessor = Preprocessing()
preprocessor.data_normaliser()
preprocessor.label_vectoriser()

"""Retrieval of features and labels after preprocessing"""
data_train, data_valid = preprocessor.get_data()
labels_train, labels_valid = preprocessor.get_labels()

def generate_model(optimizer='rmsprop', init='glorot_uniform'): # Generates feed-forward network
    model = Sequential() # Feed-forward network model
    model.add(InputLayer(input_shape=(data_train.shape[1],))) # Input layer with number of neurons dependent upon dimensions of training data
    model.add(Dense(12, activation='relu')) # Hidden layer that uses rectified linear unit as an activation function for each neuron
    model.add(Dense(7, activation='softmax')) # Output layer with 7 neurons, one for every possible classification that uses the softmax activation function
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']) # Compiles the model
    return model

model_init_batch_epoch_CV = KerasClassifier(build_fn=generate_model, verbose=1) # Wraps the model in a KerasClassifier

"""Parameters for grid search algorithm to exhaust"""
init_mode = ['glorot_uniform', 'uniform']
batches = [128, 512]
epochs = [10, 20]

"""Definition of grid search parameters"""
param_grid = dict(epochs=epochs, batch_size=batches, init=init_mode)
grid = GridSearchCV(estimator=model_init_batch_epoch_CV, 
                    param_grid=param_grid,
                    cv=3)
grid_result = grid.fit(data_train, labels_train)

"""Grid search results"""
print("\n")
print(f'Best Accuracy for {grid_result.best_score_:.4} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'mean={mean:.4}, std={stdev:.4} using {param}')

"""Training with optimised hyperparameters"""
print("\nTraining with optimised hyperparameters...")
model = generate_model() # Generates model and saves to variable for training
model.fit(data_train, labels_train, epochs=grid_result.best_params_["epochs"], batch_size=grid_result.best_params_["batch_size"]) # Trains the model
model.save("model/classifier.h5") # Saves the trained model