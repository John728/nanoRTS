Made with the same as model_3, but with 

[
                layers.Dense(1000, activation="relu", input_shape=self.input_shape),
                layers.Dense(2000, activation="relu"),
                layers.Dense(3000, activation="relu"),
                layers.Dense(3000, activation="relu"),
                layers.Dense(2000, activation="relu"),
                layers.Dense(1000, activation="relu"),                
                layers.Dense(1, activation="relu"),
            ]