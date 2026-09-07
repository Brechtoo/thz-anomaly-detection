import pandas as pd

def load_measurement(file_path):

    df = pd.read_csv(file_path, sep=r"\s+", header=None, comment="#", names=["x", "y"], usecols=[0, 1])

    df["x"] = pd.to_numeric(df["x"], errors="coerce")

    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.dropna()

    return df