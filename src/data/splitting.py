from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

def create_splits(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.36, stratify=y, random_state=RANDOM_STATE)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=20 / 36, stratify=y_temp, random_state=RANDOM_STATE)

    return X_train, X_val, X_test, y_train, y_val, y_test