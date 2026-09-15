from sklearn.ensemble import IsolationForest


def train_isolation_forest(X_train, random_state):
    model = IsolationForest(
        n_estimators=100,
        contamination="auto",
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train)

    return model


def calculate_anomaly_scores(model, X):
    scores = -model.score_samples(X)
    labels = model.predict(X)

    return scores, labels