from pathlib import Path
import re

import pandas as pd

from src.paths import MEASUREMENTS_ROOT, HOHLRAUM_TESTKÖRPER_1_ROOT, HOHLRAUM_TESTKÖRPER_2_ROOT, HOHLRAUM_TESTKÖRPER_3_ROOT, LABELS_1_ROOT, LABELS_2_ROOT, LABELS_3_ROOT

OUTPUT_DIR = MEASUREMENTS_ROOT

DATASETS = [
    (
        Path(HOHLRAUM_TESTKÖRPER_1_ROOT),
        Path(LABELS_1_ROOT),
        OUTPUT_DIR / "data1_labeled.csv",
    ),
    (
        Path(HOHLRAUM_TESTKÖRPER_2_ROOT),
        Path(LABELS_2_ROOT),
        OUTPUT_DIR / "data2_labeled.csv",
    ),
    (
        Path(HOHLRAUM_TESTKÖRPER_3_ROOT),
        Path(LABELS_3_ROOT),
        OUTPUT_DIR / "data3_labeled.csv",
    ),
]

COORD_PATTERN = re.compile(
    r"\[(-?\d+(?:\.\d+)?),"
    r"(-?\d+(?:\.\d+)?),"
    r"(-?\d+(?:\.\d+)?)\]"
)


def extract_coordinates(filename: str):

    match = COORD_PATTERN.search(filename)

    if match is None:
        return None, None, None

    x = float(match.group(1))
    y = float(match.group(2))
    z = float(match.group(3))

    return int(round(x)), int(round(y)), z


def label_dataset(measurement_root, label_file, output_file):

    labels = pd.read_csv(label_file, sep=r"\s+", comment="#", names=["x", "y", "label"])

    labels = labels.astype({
        "x": int,
        "y": int,
        "label": int,
    })

    label_lookup = {(row.x, row.y): row.label for row in labels.itertuples(index=False)}

    rows = []

    for file_path in measurement_root.rglob("*.txt"):

        x, y, z = extract_coordinates(file_path.name)

        label = (label_lookup.get((x, y))
            if x is not None and y is not None
            else None
        )

        rows.append({
            "file_path": str(file_path),
            "filename": file_path.name,
            "x": x,
            "y": y,
            "z": z,
            "label": label,
            "label_found": label is not None,
        })

    df = (pd.DataFrame(rows).sort_values(["x", "y"]).reset_index(drop=True))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    print(f"{output_file.name}: {len(df)} messungen, {df['label_found'].sum()} labels gefunden")

    return df


def main():

    for measurement_root, label_file, output_file in DATASETS:

        label_dataset(measurement_root, label_file, output_file)


if __name__ == "__main__":
    main()