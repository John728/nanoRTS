from tensorflow import keras
from generate_RTS import generate_RTS, generate_gaussian_noise, rolling_average, generate_sinusoidal_signal
from matplotlib import pyplot as plt
import numpy as np

model = keras.models.load_model('./')
model1 = keras.models.load_model('./model_2')

# Generate RTS signal and noise
rts = generate_RTS(
    num_states=2,
    transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
    num_samples=1000
)

noise = generate_gaussian_noise(num_samples=1000, mean=0, std=0.2)

sin = generate_sinusoidal_signal(num_samples=1000, freq=0.02, amplitude=0.4)

# Add the noise to the RTS signal
# noisy_signal = rts + noise + sin

# noisy_signal = rolling_average(noisy_signal, 50)

noisy_signal = np.reshape(rts + noise + sin, (1, 1000))
# # Predict the state of the RTS
predicted_rts = model.predict(noisy_signal)
predicted_rts1 = model1.predict(noisy_signal)

predicted_rts = np.reshape(predicted_rts, (1000,))
predicted_rts1 = np.reshape(predicted_rts1, (1000,))


threshold = 0.5


# # Plot the RTS signal and the predicted RTS
plt.plot(rts + noise, label='Noisy Signal', color='yellow')
plt.plot(predicted_rts1, label='Predicted RTS1', color='blue')
plt.plot(rts, label='RTS', color='red')
plt.plot(predicted_rts, label='Predicted RTS', color='green')
plt.legend()
plt.show()

# predicted_rts[predicted_rts > threshold] = 1
# predicted_rts[predicted_rts <= threshold] = 0

# plt.plot(rts, label='RTS', color='red')
# plt.plot(predicted_rts, label='Predicted RTS (Threshold)')
# plt.legend()
# plt.show()
