import numpy as np


HOHLRAUM_LABEL = 1
YELLOW_LABEL = 2
NO_HOHLRAUM_LABEL = 3


def threshold_predict(probabilities, classes, red_threshold=0.5, green_threshold=0.99):

    class_to_idx = {label: idx
        for idx, label in enumerate(classes)
    }

    predictions = []

    for row in probabilities:
        p1 = row[class_to_idx[HOHLRAUM_LABEL]]
        p3 = row[class_to_idx[NO_HOHLRAUM_LABEL]]

        if p1 >= red_threshold:
            predictions.append(HOHLRAUM_LABEL)

        elif p3 >= green_threshold:
            predictions.append(NO_HOHLRAUM_LABEL)

        else:
            predictions.append(YELLOW_LABEL)

    return np.array(predictions)


def predict_with_thresholds(model, X, red_threshold=0.5, green_threshold=0.99):

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)

        return threshold_predict(probabilities, model.classes_, red_threshold, green_threshold)

    return model.predict(X)