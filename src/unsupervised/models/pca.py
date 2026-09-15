import numpy as np

from sklearn.decomposition import PCA


def train_pca(X_train, n_components):

    model = PCA(n_components=n_components)

    model.fit(X_train)

    return model


def calculate_reconstruction_scores(model, X):

    X_reduced = model.transform(X)


    X_reconstructed = model.inverse_transform(X_reduced)

    scores = np.mean((X - X_reconstructed) ** 2, axis=1)

    return scores