import numpy as np
import pandas as pd
import tensorflow as tf
import os
from generate_RTS import generate_RTS, generate_gaussian_noise_SNR
import matplotlib.pyplot as plt
from fractions import Fraction


def process_data(data, validation_split=0.3):
    df = pd.DataFrame(data)

    # Create training and validation splits
    df_train = df.sample(frac=(1 - validation_split), random_state=0)
    df_valid = df.drop(df_train.index)

    # Split the data into features and target
    X_train = np.array(df_train["noisy_signal"].tolist())
    y_train = np.array(df_train["noisy_signal"].tolist())
    X_valid = np.array(df_valid["rts"].tolist())
    y_valid = np.array(df_valid["rts"].tolist())

    return X_train, y_train, X_valid, y_valid


def load_data(data_dir, validation_split=0.3):
    return process_data(get_data(data_dir), validation_split)


def get_data(data_dir):
    data = {"rts": [], "noisy_signal": []}
    for filename in os.listdir(data_dir):
        if filename.endswith(".tfrecord"):
            record_iterator = tf.compat.v1.io.tf_record_iterator(
                path=os.path.join(data_dir, filename)
            )
            for string_record in record_iterator:
                example = tf.train.Example()
                example.ParseFromString(string_record)
                rts = example.features.feature["rts"].float_list.value
                noisy_signal = example.features.feature["noisy_signal"].float_list.value
                data["rts"].append(rts)
                data["noisy_signal"].append(noisy_signal)

    return data


def process_sample(sample_signal):
    s = 0.99 * min(sample_signal)

    sample_signal = [x - s for x in sample_signal]

    d = 1.01 * max(sample_signal)

    sample_signal = [x / d for x in sample_signal]

    original_length = len(sample_signal)
    sample_signal = resize_sample(sample_signal, 1000)

    return sample_signal, s, d, original_length


def resize_sample(sample, new_length):
    old_len = len(sample)

    if old_len == new_length:
        return sample

    frac = Fraction(new_length, old_len)
    factor_larger = frac.numerator
    factor_smaller = frac.denominator

    if factor_larger > 1:

        larger_list = [None] * (len(sample) * factor_larger)

        for i in range(old_len):
            larger_list[i * factor_larger] = sample[i]

        for i in range(len(larger_list)):
            if larger_list[i] == None:
                larger_list[i] = curr_item
            else:
                curr_item = larger_list[i]

        sample = larger_list

    if factor_smaller > 1:

        smaller_list = [None] * (int(len(sample) / factor_smaller))

        for i in range(len(smaller_list)):
            smaller_list[i] = sample[i * factor_smaller]

        sample = smaller_list

    return sample


if __name__ == "__main__":

    rts = generate_RTS(
        num_states=2,
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
        num_samples=1200,
    )

    rts = [x * 0.3 + 2 for x in rts]

    noise = generate_gaussian_noise_SNR(1200, 30, rts)

    rts = rts + noise

    plt.plot(rts)
    plt.show()

    rts = process_sample(rts)[0]

    plt.plot(rts)
    plt.show()

    # rts = process_sample(rts)[0]

    # plt.plot(rts)
    # plt.show()

    # rts = process_sample(rts)[0]

    # plt.plot(rts)
    # plt.show()
