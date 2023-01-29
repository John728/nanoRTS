import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # I dont want to see messages
import sys
import numpy as np
import pandas as pd
from IPython.display import display
import tensorflow as tf
from keras import layers
from keras import Sequential, Model
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
import pickle
from ast import literal_eval
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dropout, Dense, Flatten, Reshape
from keras.models import Model
import time
from sklearn.model_selection import train_test_split

class Print_Progress(Callback):
    def on_train_begin(self, logs={}):

        # clear the terminal
        sys.stdout.write("\033c")

        self.epochs = self.params['epochs']
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    def on_epoch_end(self, epoch, logs={}):
        sys.stdout.write("\r")
        # sys.stdout.write("\033c")

        # percentage = round(((epoch + 1)/self.epochs) * 100, 3)
        # sys.stdout.write("\r{:.2f}% completed, loss: {:.4f}".format(percentage, logs['loss']))
        # sys.stdout.flush()
    
    def on_train_end(self, logs=None):
        print()
        print("done!")



def load_data(data_dir, validation_split=0.3):
    data = {'rts': [], 'noisy_signal': []}
    for filename in os.listdir(data_dir):
        if filename.endswith('.tfrecord'):
            record_iterator = tf.compat.v1.io.tf_record_iterator(path=os.path.join(data_dir, filename))
            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                rts = example.features.feature['rts'].float_list.value
                noisy_signal = example.features.feature['noisy_signal'].float_list.value
                data['rts'].append(rts)
                data['noisy_signal'].append(noisy_signal)

    # df = pd.DataFrame(data)

    df = pd.DataFrame(data)

    # find out the number of samples in each signal
    num_samples = len(df['rts'][0])

    # Create training and validation splits
    df_train = df.sample(frac=0.7, random_state=0)
    df_valid = df.drop(df_train.index)

    # Split the data into features and target
    X_train = np.array(df_train['noisy_signal'].tolist())
    y_train = np.array(df_train['rts'].tolist())
    X_valid = np.array(df_valid['noisy_signal'].tolist())
    y_valid = np.array(df_valid['rts'].tolist())



    return X_train, y_train, X_valid, y_valid    

    

# def load_and_preprocess_data(filename):
#     data = {'rts': [], 'noisy_signal': []}
#     for filename in os.listdir('./data'):
#         if filename.endswith('.tfrecord'):
#             record_iterator = tf.compat.v1.io.tf_record_iterator(path=os.path.join('./data', filename))
#             for string_record in record_iterator:
#                 example = tf.train.Example()
#                 example.ParseFromString(string_record)
#                 rts = example.features.feature['rts'].float_list.value
#                 noisy_signal = example.features.feature['noisy_signal'].float_list.value
#                 data['rts'].append(rts)
#                 data['noisy_signal'].append(noisy_signal)


#     df = pd.DataFrame(data)

#     print("\nData loaded")

#     # find out the number of samples in each signal
#     num_samples = len(df['rts'][0])

#     # Create training and validation splits
#     df_train = df.sample(frac=0.7, random_state=0)
#     df_valid = df.drop(df_train.index)

#     # Split the data into features and target
#     X_train = np.array(df_train['noisy_signal'].tolist())
#     y_train = np.array(df_train['noisy_signal'].tolist())
#     X_valid = np.array(df_valid['rts'].tolist())
#     y_valid = np.array(df_valid['rts'].tolist())

#     return X_train, y_train, X_valid, y_valid, num_samples

# def generate_random_forrest(input_shape, file_name, epochs=1000, batch_size=32, verbose=0,
#                             num_samples=1000, callbacks=[Print_Progress(), EarlyStopping(patience=50, restore_best_weights=True)]):
#     # define random forrest
#     model = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)



def generate_nn(X_train, y_train, X_valid, y_valid, input_shape, path_to_save, 
                epochs=1000, batch_size=32, verbose=0):
    
    start_time = time.time()

    # define the model
    model = Sequential([
        layers.Dense(1000, activation='relu', input_shape=input_shape),
        layers.Dense(2000, activation='relu'),
        layers.Dense(3000, activation='relu'),
        layers.Dense(2000, activation='relu'),
        layers.Dense(input_shape[0])
    ])

    # create the autoencoder model
    # autoencoder = Model(input_layer, decoded)
    model.compile(optimizer='adam', loss='mae')

    if verbose:
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
    print_progress.set_model(model)

    if verbose:
        print("Fitting model...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
        initial_epoch=0
    )

    # save the history
    with open(path_to_save + '_history', 'wb') as f:
        pickle.dump(history.history, f)
    f.close()
    
    model.save(path_to_save)
    end_time = time.time()

    return end_time - start_time

def generate_autoencoder(X_train, y_train, X_valid, y_valid, input_shape, path_to_save, 
                         epochs=1000, batch_size=32, verbose=0):
    
    start_time = time.time()

    # define the model
    input_layer = Input(shape=input_shape)

    encoded = Conv1D(32, 3, activation='relu', padding='same')(input_layer)
    encoded = MaxPooling1D(2, padding='same')(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = Conv1D(64, 3, activation='relu', padding='same')(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)
    encoded = Dropout(0.5)(encoded)
    encoded = Conv1D(128, 3, activation='relu', padding='same')(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)
    encoded = Dropout(0.5)(encoded)

    decoded = Conv1D(128, 3, activation='relu', padding='same')(encoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Dropout(0.5)(decoded)
    decoded = Conv1D(64, 3, activation='relu', padding='same')(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Dropout(0.5)(decoded)
    decoded = Conv1D(32, 3, activation='relu', padding='same')(decoded)
    decoded = UpSampling1D(2)(decoded)
    decoded = Dropout(0.5)(decoded)
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)

    # create the autoencoder model
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')

    if verbose:
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

    if verbose:
        print("Fitting model...")

    if verbose:
        history = autoencoder.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, print_progress, reduce_lr],
            verbose=1,
            initial_epoch=0
        )
    else:
        history = autoencoder.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stopping, reduce_lr],
            verbose=0,
            initial_epoch=0
        )

    # save the history
    with open(path_to_save + '_history', 'wb') as f:
        pickle.dump(history.history, f)
    f.close()
    
    autoencoder.save(path_to_save)
    end_time = time.time()

    return end_time - start_time

# def split_data(data, validation_split=0.3):
#     # Shuffle the data
#     data = data.shuffle(buffer_size=len(data))

#     # Split the data into training and validation sets
#     train_data = data.take((1 - validation_split) * len(data))
#     val_data = data.skip((1 - validation_split) * len(data))

#     return train_data, val_data


if __name__ == "__main__":
    X_train, y_train, X_valid, y_valid = load_data('/home/johnhenderson/SQA/nanoRTS/src/data/')
    generate_nn(X_train, y_train, X_valid, y_valid, (1000, 1), './nn_model', verbose=1)