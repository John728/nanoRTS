This model is trained with:

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

model = Sequential([
    layers.Dense(1000, activation='relu', input_shape=(num_samples,)),
    layers.Dense(2000, activation='relu'),
    # layers.Normalization(),
    # layers.Dropout(0.2),
    # layers.Dense(3000, activation='relu'),
    layers.Dense(3000, activation='relu'),
    # layers.Normalization(),
    # layers.Dropout(0.2),
    # layers.Dense(2000, activation='relu'),
    layers.Dense(2000, activation='relu'),
    layers.Dense(num_samples)
])

and no variations in the noise it sees.