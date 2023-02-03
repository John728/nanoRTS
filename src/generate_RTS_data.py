import tensorflow as tf
import numpy as np
import pandas as pd
from generate_RTS import (
    generate_RTS,
    generate_gaussian_noise,
    generate_sinusoidal_signal,
)
import sys
import pickle
import os
import time
import matplotlib.pyplot as plt
import random


def generate_data(
    num_data_points=10000,
    num_samples=1000,
    vary_noise=False,
    verbose=True,
    file_name="signals.tfrecord",
    num_states=2,
    transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
    SNR=0,
    save_data=True,
    path="./",
    use_std=False,
    std=0.1,
):

    data = {"rts": [], "noisy_signal": []}

    # print a pretty "processing" sign
    if verbose:
        with open("ascii_text.txt", "r") as file:
            for line in file:
                print(line[:-1])

    verbose_name = "./data/" + file_name + "_" + "10000"

    if save_data:
        writer = tf.io.TFRecordWriter(path + file_name)

    for i in range(num_data_points):

        # Generate RTS signal
        rts = generate_RTS(
            num_states=num_states,
            transition_probs=transition_probs,
            num_samples=num_samples,
        )

        # find average signal power
        sig_avg_watts = np.mean([rts_value**2 for rts_value in rts])

        # use this to create a desired noise level
        noise_avg_watts = sig_avg_watts / (10 ** (SNR / 10))

        # Generate the noise based on the desired SNR

        if vary_noise:
            noise = generate_gaussian_noise(
                num_samples=1000,
                mean=0,
                std=np.sqrt(noise_avg_watts) * np.random.uniform(0.01, 3),
            )
        else:
            noise = generate_gaussian_noise(
                num_samples=1000, mean=0, std=np.sqrt(noise_avg_watts)
            )

        if use_std:
            noise = generate_gaussian_noise(num_samples=1000, mean=0, std=std)

        # Add the noise to the RTS signal
        noisy_signal = rts + noise

        # print a pretty percentage
        if verbose:
            percentage = round(((i + 1) / num_data_points) * 100, 3)
            sys.stdout.write(
                "\r{:.2f}% completed, writing file: {}".format(percentage, verbose_name)
            )
            sys.stdout.flush()

        # write the data to a tfrecord file
        if save_data:
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "rts": tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=np.reshape(rts, (num_samples))
                            )
                        ),
                        "noisy_signal": tf.train.Feature(
                            float_list=tf.train.FloatList(
                                value=np.reshape(noisy_signal, (num_samples))
                            )
                        ),
                    }
                )
            )

        data["rts"].append(rts)
        data["noisy_signal"].append(noisy_signal)

        if save_data:
            writer.write(example.SerializeToString())

    if save_data:
        writer.close()

    return data


def generate_classification_data(model):
    
    print("generating classification data...")

    rts_data = generate_data(
        num_data_points=5_000,
        num_samples=1000,
        vary_noise=False,
        verbose=False,
        file_name="classification_rts.tfrecord",
        num_states=2,
        path="./data/classification/",
        save_data=False,
        use_std=True,
        std=1,
    )

    rts_data['rts'] = [1] * 5000

    no_rts_data = generate_data(
        num_data_points=5_000,
        num_samples=1000,
        vary_noise=False,
        verbose=False,
        file_name="classification_rts.tfrecord",
        num_states=2,
        path="./data/classification/",
        save_data=False,
        transition_probs=np.array([[1, 0], [0.01, 0.99]]),
        use_std=True,
        std=1,
    )

    no_rts_data['rts'] = [0] * 5000

    print("processing classification data...")

    # combine the data
    rts_data["noisy_signal"].extend(no_rts_data["noisy_signal"])
    rts_data["rts"].extend(no_rts_data["rts"])

    random.Random(4).shuffle(rts_data["noisy_signal"])
    random.Random(4).shuffle(rts_data["rts"])

    print("preparing classification data...")
    
    for i in range(len(rts_data["noisy_signal"])):
        sys.stdout.write("\r{} complete".format(i))
        rts_data["noisy_signal"][i] = model.predict(rts_data["noisy_signal"][i])

    print("saving data...")

    # save the data
    with open("./src/data/classification/classification_data.pickle", "wb") as file:
        pickle.dump(rts_data, file)



if __name__ == "__main__":

    # clear the data folder
    for file in os.listdir("./data"):
        os.remove("./data/" + file)
        print("Removed file: {}".format(file))

    generate_data(
        num_data_points=10_000,
        num_samples=1000,
        vary_noise=False,
        verbose=True,
        file_name="signals.tfrecord",
        num_states=2,
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
        SNR=35,
    )
