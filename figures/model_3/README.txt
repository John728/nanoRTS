1.

Std of 5 used to generate the data. The data is not passed through the 
autoencoder.

rts_data = generate_data(
    num_data_points=num_traces//2,
    num_samples=1000,
    vary_noise=False,
    verbose=False,
    file_name="classification_rts.tfrecord",
    num_states=2,
    path="./data/classification/",
    save_data=False,
    use_std=True,
    std=5,
)

[
    layers.Dense(1000, activation="relu", input_shape=self.input_shape),
    layers.Dense(2000, activation="relu"),
    layers.Dense(3000, activation="relu"),
    layers.Dense(2000, activation="relu"),
    layers.Dense(1),
]


2.

Same, but passed though an autoencoder
