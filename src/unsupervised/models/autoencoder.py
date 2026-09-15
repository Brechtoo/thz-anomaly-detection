import numpy as np

from sklearn.neural_network import MLPRegressor


def train_autoencoder(X_train, random_state):

    model = MLPRegressor(
        hidden_layer_sizes=(16, 8, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
        verbose=False,
    )

    model.fit(X_train, X_train)

    return model


def calculate_reconstruction_scores(model, X):

    X_reconstructed = model.predict(X)

    scores = np.mean((X - X_reconstructed) ** 2, axis=1)

    return scores