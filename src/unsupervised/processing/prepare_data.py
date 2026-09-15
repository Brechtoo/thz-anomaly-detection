import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_feature_data(data_path, target_col, exclude_columns, return_df=False):

    df = pd.read_csv(data_path)

    feature_cols = [col for col in df.select_dtypes(include=np.number).columns if col not in exclude_columns]

    X = df[feature_cols]

    y = df[target_col]

    if return_df: return df, X, y

    return X, y


def prepare_feature_data(X, y, random_state=42, test_size=0.2, scale=True):

    X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=test_size, random_state=random_state))

    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)

    X_test = imputer.transform(X_test)

    if scale:

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)

        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test