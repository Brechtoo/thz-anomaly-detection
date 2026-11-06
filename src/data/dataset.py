
TARGET_COL = "label"

EXCLUDE_COLUMNS = [
    "label",
    "file_path",
    "x",
    "y",
    "source_dataset",
    "instance_id",
]


def prepare_dataset(df):
    X = df.drop(columns=EXCLUDE_COLUMNS) # alle außer die feature
    y = df[TARGET_COL]

    return X, y
