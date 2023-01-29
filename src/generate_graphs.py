import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
from generate_RTS import generate_RTS, generate_gaussian_noise, generate_gaussian_noise_SNR
from create_model import generate_autoencoder, generate_nn, load_data
from generate_RTS_data import generate_data
from tensorflow import keras
import sys

def generate_SNR_data_vs_SNR_test(model_func, snr_training_range: np.arange, snr_test_range: np.arange, file_name: str, num_avg: int = 10, verbose: int = 1):
    """
    Generate a graph of the SNR of the data vs. the SNR of the test data for a given model. Each pixel in the graph will 
    represent the average loss of the model for a given SNR of the training data and a given SNR of the test data.
    :param model: The model to generate the graph for.
    :param snr_training_range: The range of SNR values to generate the graph for.
    :param snr_test_range: The range of SNR values to generate the graph for.
    :param file_name: The name of the file to save the graph to.
    :param num_avg: The number of times to average the loss for a given SNR value.
    :param verbose: The level of verbosity.
    :return: None
    """

    if verbose:
        print(f"Generating SNR data vs. SNR test graph...")

    data = {'data_snr': [], 'test_snr': [], 'loss': []}

    for data_snr in snr_training_range:
        
        if verbose:
            # clear screen
            os.system('clear')
            sys.stdout.write(f"\rGenerating data for SNR: {data_snr}")

        data['data_snr'].append(data_snr)

        # Generate the data
        generate_data(
            num_data_points=400,
            num_samples=1000,
            verbose=False,
            file_name= str(data_snr) + '_' + "data",
            num_states=2,
            transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
            SNR=data_snr,
            vary_noise=False
        )

        # Load the data
        x_train, y_train, x_valid, y_valid, num_samples = load_and_preprocess_data('./data/')
        
        if verbose:
            sys.stdout.write(f"\rTraining model for SNR: {data_snr}\n")

        # Train the model
        model_func(x_train, y_train, x_valid, y_valid, (num_samples, 1), verbose=1, file_name='./')

        if verbose:
            sys.stdout.write(f"Loading model for SNR: {data_snr}\n")

        # load model
        model = keras.models.load_model('./')
        loss_list = []
        # find the models loss on a range of SNR values
        for test_snr in snr_test_range:
                
            data['test_snr'].append(test_snr)
        
            # average the loss over a number of trials
            for i in range(num_avg):
                
                if verbose:
                    sys.stdout.write(f"\rCalculating loss for SNR: {test_snr} ({i}/{num_avg})")

                loss_tmp = 0

                # Generate the rts data + noise
                rts = generate_RTS(
                    num_samples=1000,
                    num_states=2,
                    transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])
                )

                noise = generate_gaussian_noise_SNR(
                    num_samples=1000,
                    SNR=test_snr,
                    signal=rts
                )


                # Test the model
                noisy_signal = np.reshape(rts + noise, (1, 1000))
                predicted_rts = model.predict(noisy_signal, verbose=0)

                # calculate loss
                loss_tmp = loss_tmp + np.mean(np.square(rts - predicted_rts))

            loss_list.append(loss_tmp/num_avg)
        
        data['loss'].append(loss_list)

        # delete the data
        os.system('rm -rf ./data/*')
    
    # create the directory if it doesn't exist
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    
    # store the data
    np.save(file_name + 'data.npy', data, allow_pickle=True)

    # plot the data where on the y-axis you have SNR of the training data
    # and on the x-axis you have the SNR of the test data and the color
    # is the loss
    plt.figure()
    plt.imshow(data['loss'], cmap='hot', interpolation='nearest')
    plt.xlabel('SNR of test data')
    plt.ylabel('SNR of training data')
    plt.xticks(np.arange(len(snr_test_range)), snr_test_range)
    plt.yticks(np.arange(len(snr_training_range)), snr_training_range)
    plt.colorbar()
    plt.savefig(file_name + 'graph.png')
    plt.close()


def generate_data_size_vs_time(model_function, data_size_range: np.arange, file_name: str, verbose: int = 1):
    """
    Generate a graph of the time taken to train vs. the amount of data.
    :param model: The model to generate the graph for.
    :param data_size_range: The range of data sizes to generate the graph for.
    :return: None
    """

    if verbose:
        print(f"Generating time vs. data size graph...")

    data = {'time': [], 'size': []}

    for data_size in data_size_range:

        os.system('rm -rf ./data/*')

        # Generate the data
        generate_data(
            num_data_points=data_size,
            num_samples=1000,
            noise=generate_gaussian_noise(num_samples=1000, mean=0, std=0.2),
            verbose=True,
            file_name=str(data_size),
            num_states=2,
            transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])
        )

        # load the data
        x_train, y_train, x_valid, y_valid, num_samples = load_and_preprocess_data('./data/')
        time_to_run = model_function(x_train, y_train, x_valid, y_valid, (num_samples, 1), './autoencoder_model')

        # store the time and data size
        data['time'].append(time_to_run)
        data['size'].append(data_size)

        # clear the data
        os.system('rm -rf ./data/*')

    # create the directory if it doesn't exist
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    
    # store the data
    np.save(file_name + 'data.npy', data, allow_pickle=True)

    # save the plot
    plt.plot(data_size_range, data['time'])
    plt.xlabel('Data Size')
    plt.ylabel('Time (s)')
    plt.savefig(file_name + 'graph.png')

def generate_loss_vs_noise(model, snr_range: np.arange, file_name: str, num_avg: int = 10, verbose: int = 1):
    """
    Generate a graph of the loss vs. noise for a given model.
    :param model: The model to generate the graph for.
    :param snr_range: The range of SNR values to generate the graph for.
    :return: None
    """

    if verbose:
        print(f"Generating loss vs. SNR graph, each point averaged over {num_avg} runs...")

    # Generate the RTS signal
    rts = generate_RTS(
        num_states=2,
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
        num_samples=1000
    )

    loss = []

    for target_snr in snr_range:


        average_loss = 0
        for i in range(num_avg):
            if verbose:
                sys.stdout.write(f"\rGenerating loss for SNR: {round(target_snr, 2)}, iteration {i + 1}")
                sys.stdout.flush()

            # find average signal power
            sig_avg_watts = np.mean([rts_value**2 for rts_value in rts])

            # use this to create a desired noise level
            noise_avg_watts = sig_avg_watts / (10 ** (target_snr / 10))

            # Generate the noise based on the desired SNR
            noise = generate_gaussian_noise(num_samples=1000, mean=0, std=np.sqrt(noise_avg_watts))

            noisy_signal = np.reshape(rts + noise, (1, 1000))

            # Predict the state of the RTS
            predicted_rts = model.predict(noisy_signal, verbose=0)

            predicted_rts = np.reshape(predicted_rts, (1000,))

            # Calculate the loss (MAE)
            average_loss = average_loss + (np.mean(np.square(rts - predicted_rts)))

        average_loss = average_loss / num_avg
        loss.append(average_loss)

    # store the loss values and snr values in a dictionary
    loss_vs_snr = {
        'loss': loss,
        'snr': snr_range
    }

    # create the directory if it doesn't exist
    if not os.path.exists(file_name):
        os.makedirs(file_name)
    
    # store the data
    np.save(file_name + 'data.npy', loss_vs_snr, allow_pickle=True)

    # save the plot
    plt.plot(snr_range, loss)
    plt.xlabel('SNR')
    plt.ylabel('Loss')
    
    plt.savefig(file_name + 'figure.png')

    if verbose:
        print(f"\nLoss vs. SNR graph saved to {file_name}")


if __name__ == '__main__':

    x_train, y_train, x_valid, y_valid = load_data('./data/')

    generate_nn(
        x_train, y_train, x_valid, y_valid,
        (1000, 1),
        './nn_model/',
        verbose=1
    )

    generate_loss_vs_noise(
        keras.models.load_model('./nn_model/'),
        np.arange(0, 30, 0.1),
        './nn_model_loss_vs_noise/',
        verbose = 1
    )

    # generate_SNR_data_vs_SNR_test(
    #     generate_autoencoder,
    #     np.arange(0, 10, 0.5),
    #     np.arange(0, 10, 0.5),
    #     file_name='./autoencoder_model_SNR_data_vs_SNR_test/',
    #     verbose=1,
    #     num_avg=50
    # )

    # # Generate the RTS data
    # generate_data(
    #     num_data_points=1_000,
    #     num_samples=1000,
    #     # noise=np.zeros(1000),
    #     noise=generate_gaussian_noise(num_samples=1000, mean=0, std=0.2),
    #         #   generate_sinusoidal_signal(num_samples=1000, freq=0.02, amplitude=0.02),
    #     # vary_noise=True,
    #     verbose=True,
    #     file_name='signals',
    #     num_states=2,
    #     transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])
    # )

    # # Generate the loss vs. noise graph

    # generate_loss_vs_noise(
    #     keras.models.load_model('./autoencoder_model'), 
    #     np.arange(0, 2, 0.1), 
    #     './autoencoder_model_loss_vs_noise/',
    #     num_avg=100,
    #     verbose=1
    # )

    # Generate the time vs. data size graph

    # generate_data_size_vs_time(
    #     generate_autoencoder,
    #     np.arange(1000, 10_000, 2000),
    #     './autoencoder_model_time_vs_data_size/',
    #     verbose=1
    # )

    # Generate SNR vs. loss graph



