import tensorflow as tf
import numpy as np
import pandas as pd
from generate_RTS import generate_RTS, generate_gaussian_noise, generate_sinusoidal_signal
import sys
import pickle 
import os
import time
import matplotlib.pyplot as plt

def generate_data(num_data_points=10000, num_samples=1000, 
                  vary_noise=False,
                  verbose=True, file_name='signals.pkl',
                  num_states=2, transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
                  SNR=0, save_data=True):

    data = {'rts': [], 'noisy_signal': []}

    # print a pretty "processing" sign
    if verbose:
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    verbose_name = "./data/" + file_name + "_" + "10000"

    if save_data:
        writer = tf.io.TFRecordWriter("./data/" + file_name)

    for i in range(num_data_points):

        # Generate RTS signal
        rts = generate_RTS(
            num_states=num_states, 
            transition_probs=transition_probs, 
            num_samples=num_samples
        )

        # find average signal power
        sig_avg_watts = np.mean([rts_value**2 for rts_value in rts])

        # use this to create a desired noise level
        noise_avg_watts = sig_avg_watts / (10 ** (SNR / 10))

        # Generate the noise based on the desired SNR
       
        if vary_noise:
            noise = generate_gaussian_noise(num_samples=1000, mean=0, std=np.sqrt(noise_avg_watts) * np.random.uniform(0.01, 3))
        else:
            noise = generate_gaussian_noise(num_samples=1000, mean=0, std=np.sqrt(noise_avg_watts))

        # Add the noise to the RTS signal
        noisy_signal = rts + noise

        # add the noise and the rts to the data
        # data['rts'].append()
        # data['noisy_signal'].append()

        # if ((i + 1) % 10_000 == 0):
        #     with open("./data/" + file_name + "_" + str(i + 1), 'wb') as f:
        #         pickle.dump(data, f)
        #     f.close()
        #     verbose_name = "./data/" + file_name + "_" + str(i + 1)

        # print a pretty percentage
        if verbose:
            percentage = round(((i + 1)/num_data_points) * 100, 3)
            sys.stdout.write("\r{:.2f}% completed, writing file: {}".format(percentage, verbose_name))
            sys.stdout.flush()

        # write the data to a tfrecord file
        if save_data:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'rts': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(rts, (num_samples)))),
                        'noisy_signal': tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(noisy_signal, (num_samples))))
                    }
                )
            )

        data['rts'].append(rts)
        data['noisy_signal'].append(noisy_signal)

        if save_data:
            writer.write(example.SerializeToString())
    
    if save_data:
        writer.close()

    return data
    # print("That took {} seconds".format(time.time() - current_time))

    # random_indicies = np.random.uniform(0, num_data_points, 4).astype(int)

    # # plot the data
    # for i in random_indicies:
    #     rts = data['rts'][i]
    #     noisy_signal = data['noisy_signal'][i]

    #     # plot the data
    #     plt.figure(figsize=(10, 5), dpi=300, )
    #     plt.plot(noisy_signal, label='Noisy Signal', color='purple', linewidth=1, alpha=0.5)    
    #     plt.plot(rts, label='RTS', color='orange', linewidth=2)
    #     plt.legend()
    #     plt.title("RTS Signal and Noisy Signal")
    #     plt.xlabel("Samples")
    #     plt.ylabel("Amplitude")
    #     plt.ylim(-0.5, 1.5)
    #     plt.savefig("./data/plot_{}.png".format(i))
    #     plt.close()



if __name__ == '__main__':

    # clear the data folder
    for file in os.listdir("./data"):
        os.remove("./data/" + file)
        print("Removed file: {}".format(file))

    generate_data(
        num_data_points=10_000,
        num_samples=1000,
        vary_noise=True,
        verbose=True,
        file_name='signals.tfrecord',
        num_states=2,
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
        SNR=10
    )