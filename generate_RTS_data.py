import numpy as np
import pandas as pd
from generate_RTS import generate_RTS, generate_gaussian_noise
import sys

data = {'rts': [], 'noisy_signal': []}

num_data = 10
num_samples = 5

# print a pretty "processing" sign
with open("ascii_text.txt", "r") as file:
    for line in file:
        print(line[:-1])

for i in range(num_data):
    # Generate RTS signal
    rts = generate_RTS(
        num_states=2, 
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]), 
        num_samples=num_samples
    )

    # Generate Gaussian noise
    noise = generate_gaussian_noise(num_samples=num_samples, mean=0, std=0.5)
    
    # Add the noise to the RTS signal
    noisy_signal = rts + noise

    data['rts'].append(rts)
    # data['gaussian_noise'].append(noise)
    data['noisy_signal'].append(noisy_signal)

    # print a pretty percentage
    percentage = round(((i + 1)/num_data) * 100, 3)
    sys.stdout.write("\r{:.2f}% completed".format(percentage))
    sys.stdout.flush()

print()

print("Exporting data to file...")

# Create a pandas dataframe
df = pd.DataFrame(data)

# Export the dataframe to a CSV file
df.to_csv('signals.csv', index=False)

