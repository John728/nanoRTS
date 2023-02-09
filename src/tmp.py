import os
import sys
import numpy as np
import tensorflow as tf
from generate_RTS_data import generate_data, generate_classification_data
from parent_model import model
from custom_models import autoencoder_model, nn_model, gaussian_model
from data_handle import load_data
from generate_RTS import generate_RTS, generate_gaussian_noise, generate_gaussian_noise_SNR, rolling_average
import matplotlib.pyplot as plt
import pickle as plk
from data_handle import process_data
from scipy.optimize import curve_fit
from math import exp

# rts_model = nn_model((1,), (1000,1))
# fit_model = autoencoder_model((1000,), (1000,1))
# fit_model.load_model("./src/generated_models/autoencoder_model/")
# rts_model.load_model("./src/generated_models/classification_model/")
# gausian_mod = gaussian_model((1,), (1000,1))

# rts = generate_RTS(
#     num_samples=1000,
# )

# # rts = np.zeros(1000)

# noise = generate_gaussian_noise(
#     num_samples=1000,
#     std=0.2,
# )

# confidence = rts_model.predict((rolling_average(rts + noise, 2)))

# print("My confidence: " + str(confidence))
# # print("Gaussion confidence: " + str(gausian_mod.predict(rolling_average(rts + noise, 2))))

# plt.plot(rts + noise, color="purple")
# if confidence > 0.5:
#     plt.plot(fit_model.predict(rolling_average(rts + noise, 2)), color="blue")
# # add text to the plot with confidence that the model contains rts put in bottom left corner
# # plt.text(0, -0.4, ))
# # plt.ylim(-0.5, 1.5)
# plt.show()

# =================================================================================================

# load pickle data 

model = autoencoder_model((1000,), (1000,))
model.load_model("./src/generated_models/autoencoder_model")

generate_classification_data(model)

data = plk.load(open('./src/data/classification/classification_data.pickle', 'rb'))

X_train, y_train, X_valid, y_valid = process_data(data, 0.3)

model = nn_model(
    input_shape=(1000,),
    output_shape=(1,),
    epochs=100,
    batch_size=32,
    verbose=1,
)

model.generate()

model.fit_model(X_train, y_train, X_valid, y_valid)

model.save_model('./src/generated_models/basic_classification_model/')

# =================================================================================================

# model = autoencoder_model((1000,), (1000,))
# model.load_model("./src/generated_models/autoencoder_model")

# # rts = generate_RTS(
# #     num_samples=1000,
# # )

# rts = np.zeros(1000)

# noise = generate_gaussian_noise(
#     num_samples=1000,
#     std=1,
# )

# plt.plot(rts + noise, color="purple")
# plt.plot(rts, color="red")
# plt.plot(model.predict(rts + noise), color="blue")
# plt.show()

# =================================================================================================

# def gauss(x,mu,sigma,A):
#     return A*exp(-(x-mu)**2/2/sigma**2)

# def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
#     return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)


# rts = generate_RTS(
#     num_samples=1000,
# )

# # rts = np.zeros(1000)

# noise = generate_gaussian_noise(
#     num_samples=1000,
#     std=1,
# )

# signal = rts + noise



# # plot the frequency of the arr
# plt.hist(arr, bins=100)

# plt.show()

# plt.plot(rts + noise)
# plt.show()