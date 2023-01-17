# forest-cover-classifier
This project uses a combination of _TensorFlow_, _scikit-learn_,and _Pandas_ to construct a multi-class classifier.

## Introduction
A feed-forward neural network is a type of neural network architecture consisting of a series of sequential layers that do not loop. The number of hidden layers determines whether it can be classified as a single-layer perceptron or a multi-layer perceptron. In this project, such an architecture is used to establish multi-class classification where data from a provided dataset is classified into one of multiple possible classes.

## Pre-Processing
The provided dataset contains cartographic variables that pertain to different forest cover types. In order to be used for training and prediction, it must be run through a series of pre-processing tasks to make it viable for use by the neural network.

Initially, it is split into data and label datasets, containing raw data, and prediction attributes respectively. Using _scikit-learn_, each is then split further into training and validation datasets, to train the neural network. After this is completed, all datasets are then normalised before being converted into binary vectors.

## AI Model
The previous datasets are then fed into the model. Using a grid-searching algorithm, the best hyperparameters are found, before a newly-generated model is trained with them. Upon completion of training, the latter is then saved into an _HDF5_ file.

## Prediction
To make predictions, the previously trained neural network is loaded. The preprocessed data and labels are again loaded for prediction purposes. Finally, a classification report of the results is displayed.