import numpy as np
import pandas as pd
from generate_RTS import generate_RTS, generate_gaussian_noise, generate_sinusoidal_signal
import sys
import pickle 

def generate_data(num_data_points=10000, num_samples=1000, 
                  noise=generate_gaussian_noise(num_samples=1000, mean=0, std=0.5), 
                  verbose=True, file_name='signals.pkl',
                  num_states=2, transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])):

    data = {'rts': [], 'noisy_signal': []}


    # print a pretty "processing" sign
    if verbose:
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    for i in range(num_data_points):

        # Generate RTS signal
        rts = generate_RTS(
            num_states=num_states, 
            transition_probs=transition_probs, 
            num_samples=num_samples
        )

        # Add the noise to the RTS signal
        noisy_signal = rts + noise

        data['rts'].append(rts)
        # data['gaussian_noise'].append(noise)
        data['noisy_signal'].append(noisy_signal)

        # print a pretty percentage
        if verbose:
            percentage = round(((i + 1)/num_data_points) * 100, 3)
            sys.stdout.write("\r{:.2f}% completed".format(percentage))
            sys.stdout.flush()


    if verbose:
        print("\nExporting data to file...")
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    f.close()


if __name__ == '__main__':
    generate_data(
        num_data_points=10000,
        num_samples=1000,
        # noise=generate_gaussian_noise(num_samples=1000, mean=0, std=0.3),
        noise=np.zeros(1000),
        verbose=True,
        file_name='signals.pkl',
        num_states=2,
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])
    )