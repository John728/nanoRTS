import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from matplotlib import pyplot as plt
from generate_RTS import generate_RTS, generate_gaussian_noise, generate_gaussian_noise_SNR
from model import generate_autoencoder, generate_nn, load_data, process_data, get_data
from generate_RTS_data import generate_data
from tensorflow import keras
import sys

def generate_SNR_data_vs_SNR_test(model_func, snr_training_range: np.arange, snr_test_range: np.arange, path: str, num_avg: int = 10, verbose: int = 1):
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
        model_data = generate_data(
            num_data_points=1000,
            num_samples=1000,
            vary_noise=False,
            verbose=False,
            num_states=2,
            transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
            SNR=data_snr,
            save_data=False,
        )

        # Load the data
        x_train, y_train, x_valid, y_valid = process_data(model_data)
        num_samples = 1000
        
        if verbose:
            sys.stdout.write(f"\rTraining model for SNR: {data_snr}\n")

        # Train the model
        model, _ = model_func(
            x_train, y_train, x_valid, y_valid, 
            (num_samples, 1), 
            verbose=0, 
            save_model=False
        )

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

    # create the directory if it doesn't exist
    if not os.path.exists(path):
        os.makedirs(path)
    
    # store the data
    np.save(path + 'SNR_data_vs_SNR_test.npy', data, allow_pickle=True)

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
    plt.savefig(path + 'SNR_data_vs_SNR_test_figure.png')
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
        x_train, y_train, x_valid, y_valid, num_samples = load_data('./data/')
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
    np.save(file_name + 'Loss_vs_noise_data.npy', loss_vs_snr, allow_pickle=True)

    # save the plot
    plt.plot(snr_range, loss, color='orange', linewidth=2)
    plt.xlabel('SNR')
    plt.ylabel('Loss')
    # plt.ylim(0, 0.5)
    plt.title('Loss vs. SNR')
    
    plt.savefig(file_name + 'Loss_vs_noise_figure.png')

    if verbose:
        print(f"\nLoss vs. SNR graph saved to {file_name}")

def generate_loss_vs_time(model_func, data, path: str, num_avg: int = 10, verbose: int = 1):
    X_train, Y_train, X_valid, Y_valid = process_data(data)

    # average over multiple runs

    _, hist = model_func(X_train, Y_train, X_valid, Y_valid, (1000, 1), verbose=1, save_model=False)

    # for i in range(num_avg - 1):
    #     if verbose:
    #         sys.stdout.write(f"\rGenerating loss for iteration {i + 1}/{num_avg}")
    #         sys.stdout.flush()
    #     _, histTmp = model_func(X_train, Y_train, X_valid, Y_valid, (1000, 1), verbose=0, save_model=False)
    #     # add hist and histTmp together for loss
    #     hist.history['loss'] = [x + y for x, y in zip(hist.history['loss'], histTmp.history['loss'])]

    # divide by num_avg
    hist.history['loss'] = [x / num_avg for x in hist.history['loss']]        

    # save data
    np.save(path + 'loss_vs_time_data.npy', hist.history['loss'], allow_pickle=True)

    # save the plot
    plt.plot(hist.history['loss'], color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. time')
    plt.savefig(path + 'loss_vs_time_figure.png')

def generate_SNR_animation(model, SNR_range, path, starting_count=0):
    i = starting_count

    rts = generate_RTS(
        num_states=2,
        transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]]),
        num_samples=1000,
        seed = 0
    )
    
    for target_SNR in SNR_range:

        sys.stdout.write(f"\rGenerating SNR animation, iteration {i + 1}/{len(SNR_range)}")

        # Generate the RTS signal

        # find average signal power
        sig_avg_watts = np.mean([rts_value**2 for rts_value in rts])

        # use this to create a desired noise level
        noise_avg_watts = sig_avg_watts / (10 ** (target_SNR / 10))

        # Generate the noise based on the desired SNR
        noise = generate_gaussian_noise(num_samples=1000, mean=0, std=np.sqrt(noise_avg_watts), seed=0)

        noisy_signal = rts + noise

        # Predict the state of the RTS
        predicted_rts = model.predict(np.reshape(noisy_signal, (1, 1000)), verbose=0)

        predicted_rts = np.reshape(predicted_rts, (1000,))

        # plot the signal
        plt.figure(dpi=150, figsize=(10, 5))
        plt.plot(noisy_signal, linewidth=1, color='purple', alpha=0.2, label='Noisy RTS Signal')
        plt.plot(rts, color='red', linewidth=1, label='RTS Signal')
        plt.plot(predicted_rts, color='orange', linewidth=2, label='Predicted RTS Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.ylim(-0.5, 1.5)
        plt.title(f'RTS Signal vs. Predicted RTS Signal for SNR = {target_SNR}')
        # add loss to bottom of figure
        plt.text(0, -0.4, f'Loss = {np.mean(np.square(rts - predicted_rts))}')
        if i == 0:
            plt.show()
        plt.savefig(path + f'figure_{i}.png')
        plt.close()
        i = i + 1

if __name__ == '__main__':

    # x_train, y_train, x_valid, y_valid = load_data('./data/')

    # ac_model, _ = generate_autoencoder(
    #     x_train, y_train, x_valid, y_valid,
    #     (1000, 1),
    #     './models/autoencoder/',
    #     verbose=1,
    #     save=True,
    #     epochs=100,
    #     batch_size=10
    # )

    # generate_loss_vs_noise(
    #     ac_model,
    #     np.arange(0, 30, 0.1),
    #     './models/autoencoder_model/',
    #     verbose = 1
    # )

    # generate_SNR_data_vs_SNR_test(
    #     generate_autoencoder,
    #     np.arange(0, 10, 0.5),
    #     np.arange(0, 10, 0.5),
    #     path='./models/autoencoder_model/',
    #     verbose=1,
    #     num_avg=10
    # )

    # generate_loss_vs_time(
    #     generate_autoencoder,
    #     get_data('./data/'),
    #     path='./models/autoencoder_model/',
    #     verbose=1,
    #     num_avg=10
    # )

    generate_SNR_animation(
        keras.models.load_model('./models/autoencoder_model/'),
        np.arange(0.005, 10, 0.1),
        './models/autoencoder_model/animation_figures/',
        starting_count=0
    )

    # generate_SNR_animation(
    #     keras.models.load_model('./models/autoencoder_model/'),
    #     np.arange(1.095, 2, 0.005),
    #     './models/autoencoder_model/animation_figures/',
    #     starting_count=219
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



