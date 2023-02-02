import os
import sys
import numpy as np
import tensorflow as tf
from generate_RTS_data import generate_data, generate_classification_data
from parent_model import model
from custom_models import autoencoder_model, nn_model
from data_handle import load_data
from generate_RTS import generate_RTS, generate_gaussian_noise, generate_gaussian_noise_SNR
import matplotlib.pyplot as plt
import pickle as plk
from data_handle import process_data

# load pickle data 

# model = autoencoder_model((1000,))
# model.load_model("./src/generated_models/autoencoder_model")

# generate_classification_data(model)

data = plk.load(open('./src/data/classification/classification_data.pickle', 'rb'))

X_train, y_train, X_valid, y_valid = process_data(data, 0.3)

# reshape the valid data
# X_valid = X_valid.reshape(-1, 1)
# y_valid = y_valid.reshape(-1, 1)
# X_train = X_train.reshape(-1, 1)
# y_train = y_train.reshape(-1, 1)

model = nn_model(
    input_shape=X_train.shape[1:],
    epochs=100,
    batch_size=32,
    verbose=1,
)

model.generate()

model.fit_model(X_train, y_train, X_valid, y_valid)

model.save_model('./src/generated_models/classification_model/')

# model = autoencoder_model((1000,))
# model.load_model("./generated_models/autoencoder_model")

# generate_classification_data(model)

# rts = generate_RTS(
#     num_samples=1000
# )

# noise = generate_gaussian_noise_SNR(
#     num_samples=1000,
#     signal=rts,
#     SNR=10
# )

# # predicted_rts = model.predict(rts + noise)

# plt.plot(rts, color="red")
# plt.plot(rts + noise, color="blue")
# # plt.plot(predicted_rts, color="blue")
# plt.show()