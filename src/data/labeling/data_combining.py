"""Combine labeled datasets without import-time reads or writes."""
from pathlib import Path

import pandas as pd

from src.paths import (
    COMBINED_DATASET_ROOT,
    DATA1_LABELED_ROOT,
    DATA2_LABELED_ROOT,
    DATA3_LABELED_ROOT,
)

DATASETS = [
    ("messreihe_1", DATA1_LABELED_ROOT),
    ("messreihe_2", DATA2_LABELED_ROOT),
    ("messreihe_3", DATA3_LABELED_ROOT),
]
OUTPUT_PATH = COMBINED_DATASET_ROOT


def combine_labeled_data(datasets):
    """Keep given dataset/row order and the historical source_x_y ID format."""
    frames = []
    names = set()
    for name, frame in datasets:
        if name in names:
            raise ValueError("Source dataset names must be unique")
        names.add(name)
        if not {"file_path", "filename", "x", "y", "z", "label"} <= set(frame):
            raise ValueError(f"Missing labeled metadata in {name}")
        if frame.empty or not frame.label.isin([0, 1, 2, 3]).all():
            raise ValueError(f"Missing or invalid original labels in {name}")
        for column in ("x", "y"):
            if frame[column].isna().any() or not (frame[column] == frame[column].round()).all():
                raise ValueError("Coordinates must be non-null integers")
        df = frame.drop(columns=["label_found"], errors="ignore").copy()
        df["source_dataset"] = name
        df["instance_id"] = name + "_x" + df.x.astype(int).astype(str) + "_y" + df.y.astype(int).astype(str)
        frames.append(df)
    if not frames:
        raise ValueError("No labeled datasets supplied")
    combined = pd.concat(frames, ignore_index=True)
    if not combined.instance_id.is_unique or not combined.file_path.is_unique:
        raise ValueError("Duplicate measurement IDs or paths")
    return combined


def main(output_file=OUTPUT_PATH):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(output_file)
    combined = combine_labeled_data([(name, pd.read_csv(path)) for name, path in DATASETS])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x") as stream:
        combined.to_csv(stream, index=False)
    return combined


if __name__ == "__main__":
    main()
