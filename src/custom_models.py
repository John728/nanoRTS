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
from scipy.stats import norm

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
        arr = []
        for i in range(len(x) - 1):
            arr.extend(np.arange(x[i], x[i + 1], 0.1))

        arr.sort()

        arr1 = [ i for i in arr if i < 0.5]
        arr2 = [ i for i in arr if i >= 0.5]

        mu1, std1 = norm.fit(arr1)
        mu2, std2 = norm.fit(arr2)

        print(mu1, std1)
        print(mu2, std2)

        if mu1 > -0.4 and mu1 < 0.5 and mu2 > 0.8 and mu2 < 1.5:
            return 1
        else:
            return 0