import os

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
import keras
import numpy as np


class Print_Progress(Callback):
    def on_train_begin(self, logs={}):

        # clear the terminal
        sys.stdout.write("\033c")

        self.epochs = self.params["epochs"]
        with open("./src/ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    def on_epoch_end(self, epoch, logs={}):
        # sys.stdout.write("\r")
        # sys.stdout.write("\033c")

        percentage = round(((epoch + 1) / self.epochs) * 100, 3)
        sys.stdout.write(
            "\r{:.2f}% completed, loss: {:.4f}".format(percentage, logs["loss"])
        )
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print()
        print("done!")


class model:

    model_has_been_generated = False

    def __init__(self, output_shape, input_shape, epochs=1000, batch_size=32, verbose=0):
        self.output_shape = output_shape
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.specific_model=None

    @abstractmethod
    def generate():
        print("Model should be a given type")
        pass

    def fit_model(self, X_train, y_train, X_valid, y_valid):
        if not self.model_has_been_generated:
            print("Model needs to be generated first")
            return

        # Compile the model
        self.specific_model.compile(optimizer="adam", loss="mae")

        # Create the early stopping callback
        early_stopping = EarlyStopping(
            min_delta=0.001,  # minimium amount of change to count as an improvement
            patience=10,  # how many epochs to wait before stopping
            restore_best_weights=True,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=10, min_lr=0.0001
        )

        print_progress = Print_Progress()
        print_progress.set_model(self.specific_model)

        if self.verbose == 1:
            callbacks = [early_stopping, reduce_lr, print_progress]
        else:
            callbacks = [early_stopping, reduce_lr]
            
        # Fit the model
        history = self.specific_model.fit(
            X_train,
            X_valid,
            validation_data=(y_train, y_valid),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=self.verbose,
            initial_epoch=0,
        )

        self.history = history

    def save_model(self, path):
        # if path does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

        # save the model
        self.specific_model.save(path)

        if self.verbose:
            print("Model saved to {}".format(path))

    def load_model(self, path):
        self.specific_model = keras.models.load_model(path)

    def predict(self, X):
        X = np.reshape(X, (1, 1000))
        predicted_sig = self.specific_model.predict(X)
        predicted_sig = np.reshape(predicted_sig, self.output_shape)
        return predicted_sig
    
    def set_model(self, model):
        self.specific_model = model


# if __name__ == '__main__':

#     X_train, y_train, X_valid, y_valid = data_handle.load_data('./data')

#     model = nn_model((1000,), 100, 32, verbose=1)
#     model.generate()
#     model.fit_model(X_train, y_train, X_valid, y_valid)
