import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
from generateRTS import generate_rts, generate_gaussian_noise
import sys
import pandas as pd
from IPython.display import display
from keras import layers
from keras import Sequential
from keras.callbacks import EarlyStopping, Callback
import sys
from ast import literal_eval

class Print_Progress(Callback):
    def on_train_begin(self, logs={}):
        self.epochs = self.params['epochs']
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    def on_epoch_end(self, epoch, logs={}):
        percentage = round(((epoch + 1)/self.epochs) * 100, 3)
        sys.stdout.write("\r{:.2f}% completed".format(percentage))
        sys.stdout.flush()

    def on_train_end(self, logs=None):
        print('\n')


data = {'rts': [], 'noisy_signal': []}

num_data = 10
num_samples = 5

# print a pretty "processing" sign
with open("ascii_text.txt", "r") as file:
    for line in file:
        print(line[:-1])


for i in range(num_data):
    # Generate RTS signal
    rts = generate_rts(
        num_states=2, 
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]), 
        num_samples=num_samples
    )

    # Generate Gaussian noise
    noise = generate_gaussian_noise(num_samples=num_samples, mean=0, std=0.5)
    
    # Add the noise to the RTS signal
    noisy_signal = rts + noise

    data['rts'].append(np.reshape(rts, (1, num_samples)))
    data['noisy_signal'].append(np.reshape(noisy_signal, (1, num_samples)))

    # print a pretty percentage
    percentage = round(((i + 1)/num_data) * 100, 3)
    sys.stdout.write("\r{:.2f}% completed".format(percentage))
    sys.stdout.flush()


print()

print("Exporting data to file...")

# print(rts_arr)

# # Create a pandas dataframe
df = pd.DataFrame(data)

# # Export the dataframe to a CSV file
# df.to_csv('signals.csv', index=False)

# ###############

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

# data = pd.read_csv('./signals.csv')
# data = data['noisy_signal'].apply(lambda x: literal_eval(x))

# Create training and validation splits
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

# Split the data into features and target
X_train = df_train['noisy_signal']
X_valid = df_valid['noisy_signal']
y_train = df_train['rts']
y_valid = df_valid['rts']

X_train = np.array(X_train.tolist()).reshape(-1, num_samples)
X_valid = np.array(X_valid.tolist()).reshape(-1, num_samples)
y_train = np.array(y_train.tolist()).reshape(-1, num_samples)
y_valid = np.array(y_valid.tolist()).reshape(-1, num_samples)

# Create the model
model = Sequential([
    layers.Dense(1000, activation='relu', input_shape=(num_samples,)),
    layers.Dense(2000, activation='relu'),
    layers.Dense(3000, activation='relu'),
    layers.Dense(2000, activation='relu'),
    layers.Dense(num_samples)
])

# Compile the model
model.compile(optimizer='adam', loss='mae')

# Create the early stopping callback
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

print_progress = Print_Progress()
print_progress.set_model(model)

# Fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=200,
    epochs=1000,
    callbacks=[early_stopping, print_progress],
    verbose=0
)

model.save('./')