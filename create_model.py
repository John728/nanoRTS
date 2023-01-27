import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # I dont want to see messages
import sys
import numpy as np
import pandas as pd
from IPython.display import display
from keras import layers
from keras import Sequential
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
import pickle
from ast import literal_eval
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout
from keras.models import Model

class Print_Progress(Callback):
    def on_train_begin(self, logs={}):
        self.epochs = self.params['epochs']
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    def on_epoch_end(self, epoch, logs={}):
        percentage = round(((epoch + 1)/self.epochs) * 100, 3)
        sys.stdout.write("\r{:.2f}% completed, loss: {:.4f}".format(percentage, logs['loss']))
        sys.stdout.flush()
    
    def on_train_end(self, logs=None):
        print()
        print("done!")

# Load the data from each file in ./data
data = {'rts': [], 'noisy_signal': []}
for file in os.listdir('./data'):
    sys.stdout.write("\rLoading file: " + file + "...")
    with open('./data/' + file, 'rb') as f:
        data_tmp = pickle.load(f)
        data.update(data_tmp)
    f.close()


df = pd.DataFrame(data)

print("\nData loaded")

# find out the number of samples in each signal
num_samples = len(df['rts'][0])

# Create training and validation splits
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

# Split the data into features and target
X_train = np.array(df_train['noisy_signal'].tolist())
y_train = np.array(df_train['noisy_signal'].tolist())
X_valid = np.array(df_valid['rts'].tolist())
y_valid = np.array(df_valid['rts'].tolist())

def generate_autoencoder(input_shape, file_name, epochs=1000, batch_size=32, verbose=0,
                         num_samples=1000, callbacks=[Print_Progress(), EarlyStopping(patience=50, restore_best_weights=True)]):
    # define the model
    input_layer = Input(shape=input_shape)

    encoded = Conv1D(16, 3, activation='relu', padding='same')(input_layer)
    encoded = MaxPooling1D(2, padding='same')(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)
    encoded = Dropout(0.5)(encoded)

    decoded = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    decoded = UpSampling1D(2)(decoded)
    encoded = Dropout(0.5)(decoded)
    decoded = Conv1D(16, 3, activation='relu', padding='same')(decoded)
    decoded = UpSampling1D(2)(decoded)
    encoded = Dropout(0.5)(decoded)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)

    # create the autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    print("Pre-processing complete")

    # # Create the early stopping callback
    early_stopping = EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=50, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=10,
        min_lr=0.0001
    )

    print_progress = Print_Progress()
    print_progress.set_model(autoencoder)

    print("Fitting model...")

    # # Fit the model
    history = autoencoder.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=200,
        epochs=1000,
        callbacks=[early_stopping, print_progress, reduce_lr],
        verbose=0,
        initial_epoch=0
    )

    # save the history
    with open(file_name + '_history', 'wb') as f:
        pickle.dump(history.history, f)
    f.close()
    
    autoencoder.save(file_name)

if __name__ == "__main__":
    generate_autoencoder((num_samples, 1), './autoencoder_model')