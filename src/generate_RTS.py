import numpy as np
import matplotlib.pyplot as plt


def generate_RTS(
    num_samples,
    transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
    num_states=2,
    seed=None,
):
    # Define the number of states and the transition probabilities
    # Set the initial state
    current_state = 0

    if seed is not None:
        np.random.seed(seed)

    # Generate the RTS
    rts = []
    for _ in range(num_samples):
        # Append the current state to the RTS
        rts.append(current_state)
        # Generate a random number between 0 and 1
        rand_num = np.random.rand()
        # Determine the next state using the transition probabilities
        next_state = np.where(rand_num < np.cumsum(transition_probs[current_state]))[0][
            0
        ]
        # Update the current state
        current_state = next_state

    return rts


def generate_fixed_RTS(num_samples, transition_rate, num_states=2):
    # Set the initial state
    current_state = 0
    # Generate the RTS
    rts = []
    for i in range(num_samples):
        # Append the current state to the RTS
        rts.append(current_state)
        # Transition to the other state after a fixed number of samples
        if (i + 1) % transition_rate == 0:
            current_state = (current_state + 1) % num_states
    return rts


def generate_gaussian_noise(num_samples, mean, std, seed=None):
    # Generate Gaussian noise
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(mean, std, num_samples)
    return noise


def generate_gaussian_noise_SNR(num_samples, SNR, signal, seed=None):
    sig_avg_watts = np.mean([rts_value**2 for rts_value in signal])

    # use this to create a desired noise level
    noise_avg_watts = sig_avg_watts / (10 ** (SNR / 10))

    # Generate the noise based on the desired SNR
    noise = generate_gaussian_noise(
        num_samples=num_samples, mean=0, std=np.sqrt(noise_avg_watts), seed=seed
    )

    return noise


def rolling_average(signal, num_samples_per_average):
    # Create an array to hold the rolling averages
    rolling_averages = np.zeros(len(signal))
    # Calculate the rolling average
    for i in range(len(signal)):
        start = max(0, i - num_samples_per_average + 1)
        rolling_averages[i] = np.mean(signal[start : i + 1])
    return rolling_averages


def generate_sinusoidal_signal(num_samples, freq, amplitude, phase=0):
    # Generate a sinusoidal signal
    signal = amplitude * np.sin(2 * np.pi * freq * np.arange(num_samples) + phase)
    return signal


if __name__ == "__main__":

    rts = generate_RTS(
        num_samples=100_000,
    )

    state_one_count = len([x for x in rts if x > 0])
    state_zero_count = len([x for x in rts if x < 1])

    # they should roughly be 50000
    assert state_one_count == 50000
    assert state_zero_count == 50000
