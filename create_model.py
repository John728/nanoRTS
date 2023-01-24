import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # I dont want to see messages
import sys
import numpy as np
import pandas as pd
from IPython.display import display
from keras import layers
from keras import Sequential
from keras.callbacks import EarlyStopping, Callback
import pickle
from ast import literal_eval


class Print_Progress(Callback):
    def on_train_begin(self, logs={}):
        self.epochs = self.params['epochs']
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    def on_epoch_end(self, epoch, logs={}):
        percentage = round(((epoch + 1)/self.epochs) * 100, 3)
        sys.stdout.write("\r{:.2f}% completed, loss: {:.4f}, time elapsed {}s".format(percentage, logs['loss'], logs['duration']))
        sys.stdout.flush()
    
    def on_train_end(self, logs=None):
        print()
        print("done!")

file = open('signals.pkl', 'rb')
data = pickle.load(file)
file.close()

df = pd.DataFrame(data)

print("Data loaded")

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

# Create the model
model = Sequential([
    layers.Dense(1000, activation='relu', input_shape=(num_samples,)),
    layers.Dense(2000, activation='relu'),
    # layers.Normalization(),
    # layers.Dropout(0.2),
    # layers.Dense(3000, activation='relu'),
    layers.Dense(3000, activation='relu'),
    # layers.Normalization(),
    # layers.Dropout(0.2),
    # layers.Dense(2000, activation='relu'),
    layers.Dense(2000, activation='relu'),
    layers.Dense(num_samples)
])

# # Compile the model
model.compile(optimizer='adam', loss='mae')

print("Pre-processing complete")

# # Create the early stopping callback
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

print_progress = Print_Progress()
print_progress.set_model(model)

print("Fitting model...")

# # Fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=200,
    epochs=1000,
    callbacks=[early_stopping, print_progress],
    verbose=0
)

model.save('./')