import os
import sys
import numpy as np
import tensorflow as tf
from generate_RTS_data import generate_data, generate_classification_data
from parent_model import model
from custom_models import autoencoder_model
from data_handle import load_data

# clear the data folder
# for file in os.listdir("./data/classification"):
#     os.remove("./data/classification" + file)
#     print("Removed file: {}".format(file))


# data = generate_data(
#     num_data_points=10_000,
#     num_samples=1000,
#     vary_noise=False,
#     verbose=True,
#     path="./data/classification/",
#     num_states=2,
#     transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
#     SNR=30,
#     save_data=True,
# )

# X_train, y_train, X_valid, y_valid = load_data('./data/classification/')

# model = autoencoder_model(
#     input_shape=(1000, 1),
#     verbose=True,
# )

# model.generate()

# model.fit_model(
#     X_train, y_train, X_valid, y_valid
# )

# model.save_model("./generated_models/autoencoder_model/")


# model = autoencoder_model((1000,))
# model.load_model("./src/generated_models/autoencoder_model")


# generate_classification_data(
#     model=model,
#     num_data_points=1000,
#     num_samples=1000,
#     vary_noise=False,
#     verbose=False,
#     path="./src/data/classification/",
#     num_states=2,
#     transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
#     SNR=0,
#     save_data=True,
#     use_std=True,
#     std=1
# )
