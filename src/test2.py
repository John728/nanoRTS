from tensorflow import keras
from generate_RTS import generate_RTS, generate_gaussian_noise, rolling_average, generate_sinusoidal_signal, generate_gaussian_noise_SNR
from matplotlib import pyplot as plt
import numpy as np

model = keras.models.load_model('./')
# model1 = keras.models.load_model('./model_2')

# Generate RTS signal and noise
rts = generate_RTS(
    num_states=2,
    transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
    num_samples=1000
)

# noise = generate_gaussian_noise_SNR(num_samples=1000, SNR=0.00000000001, signal=rts)
noise = generate_gaussian_noise(num_samples=1000, mean=0, std=1)

# sin = generate_sinusoidal_signal(num_samples=1000, freq=0.02, amplitude=0.4)

# Add the noise to the RTS signal
# noisy_signal = rts + noise + sin

# ra = rolling_average(rts+noise, 50)

noisy_signal = np.reshape(rts + noise, (1, 1000))
# # Predict the state of the RTS
predicted_rts = model.predict(noisy_signal)
# predicted_rts1 = model1.predict(noisy_signal)

predicted_rts = np.reshape(predicted_rts, (1000,))
# predicted_rts1 = np.reshape(predicted_rts1, (1000,))


threshold = 0.5


# # Plot the RTS signal and the predicted RTS
plt.plot(rts + noise, label='Noisy Signal', color='yellow')
# plt.plot(ra, label='Predicted RTS1', color='blue')
plt.plot(rts, label='RTS', color='red')
plt.plot(predicted_rts, label='Predicted RTS', color='green')
plt.ylim(-2, 2)
plt.legend()
plt.show()

# predicted_rts[predicted_rts > threshold] = 1
# predicted_rts[predicted_rts <= threshold] = 0

# plt.plot(rts, label='RTS', color='red')
# plt.plot(predicted_rts, label='Predicted RTS (Threshold)')
# plt.legend()
# plt.show()
