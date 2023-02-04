import model
import os
import GPy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # I dont want to see messages
from abc import ABC, abstractmethod
import sys
from IPython.display import display
from keras import layers
from keras import Sequential, Model
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from parent_model import model
import numpy as np

class nn_model(model):
    def generate(self):

        if self.verbose == 1:
            print("Generating model...")

        self.model_has_been_generated = True
        num_samples = self.input_shape[0]

        # Create the model
        self.specific_model = Sequential(
            [
                layers.Dense(1000, activation="relu", input_shape=self.input_shape),
                layers.Dense(2000, activation="relu"),
                layers.Dense(3000, activation="relu"),
                layers.Dense(3000, activation="relu"),
                layers.Dense(2000, activation="relu"),
                layers.Dense(1),
            ]
        )


class autoencoder_model(model):
    def generate(self):

        if self.verbose == 1:
            print("Generating model...")

        self.model_has_been_generated = True

        self.specific_model = Sequential(
            [
                Conv1D(
                    32,
                    3,
                    activation="relu",
                    padding="same",
                    input_shape=self.input_shape,
                ),
                MaxPooling1D(2, padding="same"),
                Dropout(0.5),
                Conv1D(64, 3, activation="relu", padding="same"),
                MaxPooling1D(2, padding="same"),
                Dropout(0.5),
                Conv1D(128, 3, activation="relu", padding="same"),
                MaxPooling1D(2, padding="same"),
                Dropout(0.5),
                Conv1D(128, 3, activation="relu", padding="same"),
                UpSampling1D(2),
                Dropout(0.5),
                Conv1D(64, 3, activation="relu", padding="same"),
                UpSampling1D(2),
                Dropout(0.5),
                Conv1D(32, 3, activation="relu", padding="same"),
                UpSampling1D(2),
                Dropout(0.5),
                Conv1D(1, 3, activation="sigmoid", padding="same"),
            ]
        )


class classic_model(model):
    def generate(self):
        pass

    def fit_model(self, X_train, y_train, X_valid, y_valid):
        pass

    def load_model(self, path):
        pass

    def predict(self, X_test):
        pass

class gaussian_model(model):
    def generate(self):
        pass

    def fit_model(self, X_train, y_train, X_valid, y_valid):
        pass

    def load_model(self, path):
        pass

    def predict(self, x):
        # Fit a Gaussian Process model to the data
        x = np.reshape(x, (-1, 1))
        kernel = GPy.kern.RBF(input_dim=1)
        model = GPy.models.GPRegression(np.array(x).reshape(-1, 1), np.zeros_like(x), kernel)
        model.optimize()
        
        # Calculate the confidence in the model fit
        confidence = model.log_likelihood()
        return confidence