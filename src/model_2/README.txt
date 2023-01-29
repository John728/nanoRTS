generate_data(
    num_data_points=5_000,
    num_samples=1000,
    noise=generate_gaussian_noise(num_samples=1000, mean=0, std=0.001) + 
            generate_sinusoidal_signal(num_samples=1000, freq=0.02, amplitude=0.01),
    vary_noise=True,
    verbose=True,
    file_name='signals',
    num_states=2,
    transition_probs=np.array([[0.99, 0.01], [0.01, 0.99]])
)

model = Sequential([
    layers.Dense(1000, activation='relu', input_shape=(num_samples,)),
    layers.Normalization(),
    layers.Dense(2000, activation='relu'),
    layers.Dropout(0.5),
    layers.Normalization(),
    layers.Dense(3000, activation='relu'),
    layers.Dropout(0.5),
    layers.Normalization(),
    layers.Dense(2000, activation='relu'),
    layers.Dense(num_samples)
])

# # Compile the model
model.compile(optimizer='adam', loss='mae')

print("Pre-processing complete")

# # Create the early stopping callback
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)

print_progress = Print_Progress()
print_progress.set_model(model)

print("Fitting model...")

# # Fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=800,
    epochs=10_000,
    callbacks=[early_stopping, print_progress],
    verbose=0
)

model.save('./')