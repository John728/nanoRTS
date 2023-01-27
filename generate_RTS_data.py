import numpy as np
import pandas as pd
from generate_RTS import generate_RTS, generate_gaussian_noise, generate_sinusoidal_signal
import sys
import pickle 
import shelve

def generate_data(num_data_points=10000, num_samples=1000, 
                  noise=generate_gaussian_noise(num_samples=1000, mean=0, std=0.5),
                  vary_noise=False,
                  verbose=True, file_name='signals.pkl',
                  num_states=2, transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])):

    data = {'rts': [], 'noisy_signal': []}

    # print a pretty "processing" sign
    if verbose:
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    verbose_name = "./data/" + file_name + "_" + "10000"

    for i in range(num_data_points):

        # Generate RTS signal
        rts = generate_RTS(
            num_states=num_states, 
            transition_probs=transition_probs, 
            num_samples=num_samples
        )

        if vary_noise:
            # noise = noise + generate_gaussian_noise(num_samples=num_samples, mean=0, std=np.random.uniform(0, 0.5))
            noise = noise + generate_sinusoidal_signal(num_samples=num_samples, freq=np.random.uniform(0, 0.2), amplitude=np.random.uniform(0, 0.1), phase=np.random.uniform(0, 2*np.pi))
            

        # Add the noise to the RTS signal
        noisy_signal = rts + noise

        # add the noise and the rts to the data
        data['rts'].append(np.reshape(rts, (num_samples)))
        data['noisy_signal'].append(np.reshape(noisy_signal, (num_samples)))

        if ((i + 1) % 10_000 == 0):
            with open("./data/" + file_name + "_" + str(i + 1), 'wb') as f:
                pickle.dump(data, f)
            f.close()
            verbose_name = "./data/" + file_name + "_" + str(i + 1)

        # print a pretty percentage
        if verbose:
            percentage = round(((i + 1)/num_data_points) * 100, 3)
            sys.stdout.write("\r{:.2f}% completed, writing file: {}".format(percentage, verbose_name))
            sys.stdout.flush()

    with open("./data/" + file_name + "_" + str(i + 1), 'wb') as f:
        pickle.dump(data, f)
    f.close()
    print()


if __name__ == '__main__':
    generate_data(
        num_data_points=10_000,
        num_samples=1000,
        # noise=np.zeros(1000),
        noise=generate_gaussian_noise(num_samples=1000, mean=0, std=0.2) + 
              generate_sinusoidal_signal(num_samples=1000, freq=0.02, amplitude=0.02),
        # vary_noise=True,
        verbose=True,
        file_name='signals',
        num_states=2,
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])
    )