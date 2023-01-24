import numpy as np
import matplotlib.pyplot as plt

def generate_RTS(transition_probs, num_samples, num_states = 2):

    # Define the number of states and the transition probabilities
    # Set the initial state
    current_state = 0

    # Generate the RTS
    rts = []
    for _ in range(num_samples):
        # Append the current state to the RTS
        rts.append(current_state)
        # Generate a random number between 0 and 1
        rand_num = np.random.rand()
        # Determine the next state using the transition probabilities
        next_state = np.where(rand_num < np.cumsum(transition_probs[current_state]))[0][0]
        # Update the current state
        current_state = next_state

    return rts


def generate_fixed_RTS(num_samples, transition_rate, num_states = 2):
    # Set the initial state
    current_state = 0
    # Generate the RTS
    rts = []
    for i in range(num_samples):
        # Append the current state to the RTS
        rts.append(current_state)
        # Transition to the other state after a fixed number of samples
        if (i+1) % transition_rate == 0:
            current_state = (current_state + 1) % num_states
    return rts

def generate_gaussian_noise(num_samples, mean, std):
    # Generate Gaussian noise
    noise = np.random.normal(mean, std, num_samples)
    return noise

def rolling_average(signal, num_samples_per_average):
    # Create an array to hold the rolling averages
    rolling_averages = np.zeros(len(signal))
    # Calculate the rolling average
    for i in range(len(signal)):
        start = max(0, i - num_samples_per_average + 1)
        rolling_averages[i] = np.mean(signal[start:i+1])
    return rolling_averages

def get_sinusoidal_signal(num_samples, freq, amplitude):
    # Generate a sinusoidal signal
    signal = amplitude * np.sin(2 * np.pi * freq * np.arange(num_samples))
    return signal


if __name__ == "__main__":

    rts = generate_RTS(
        num_states=2, 
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]), 
        num_samples=1000
    )

    sinusoidal = get_sinusoidal_signal(num_samples=1000, freq=0.02, amplitude=0.2)
    noise = generate_gaussian_noise(num_samples=1000, mean=0, std=0.5)

    noisy_rts = noise + rts

    plt.plot(sinusoidal + rts + noise, '-o', markersize=2)
    plt.plot(rts, '-o', markersize=2)
    plt.xlabel('Time')
    plt.ylabel('State')
    plt.show()