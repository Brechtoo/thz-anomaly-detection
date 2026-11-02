from pathlib import Path
import re

import pandas as pd

from src.paths import (
    MEASUREMENTS_ROOT, HOHLRAUM_TESTKÖRPER_1_ROOT, HOHLRAUM_TESTKÖRPER_2_ROOT,
    HOHLRAUM_TESTKÖRPER_3_ROOT, LABELS_1_ROOT, LABELS_2_ROOT, LABELS_3_ROOT,
)

OUTPUT_DIR = MEASUREMENTS_ROOT
DATASETS = [
    (Path(HOHLRAUM_TESTKÖRPER_1_ROOT), Path(LABELS_1_ROOT), OUTPUT_DIR / "data1_labeled.csv"),
    (Path(HOHLRAUM_TESTKÖRPER_2_ROOT), Path(LABELS_2_ROOT), OUTPUT_DIR / "data2_labeled.csv"),
    (Path(HOHLRAUM_TESTKÖRPER_3_ROOT), Path(LABELS_3_ROOT), OUTPUT_DIR / "data3_labeled.csv"),
]
COORD_PATTERN = re.compile(r"\[(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\]")


def extract_coordinates(filename: str):
    match = COORD_PATTERN.search(filename)
    if match is None:
        return None, None, None
    x, y, z = map(float, match.groups())
    # Preserve the historical Python round() convention, including half ties.
    return int(round(x)), int(round(y)), z


def build_labeled_dataset(measurement_root, label_file):
    """Read existing annotation assignments, never infer or replace labels."""
    root = Path(measurement_root)
    labels = pd.read_csv(label_file, sep=r"\s+", comment="#", names=["x", "y", "label"])
    numeric = labels.apply(pd.to_numeric, errors="raise")
    if numeric.isna().any().any() or not (numeric == numeric.round()).all().all():
        raise ValueError("Annotation coordinates and labels must be non-null integers")
    labels = numeric.astype({"x": int, "y": int, "label": int})
    if not labels.label.isin([0, 1, 2, 3]).all():
        raise ValueError("Annotations must use original labels 0/1/2/3")
    if labels.duplicated(["x", "y"]).any():
        raise ValueError("Ambiguous duplicate annotation coordinates")
    lookup = {(r.x, r.y): r.label for r in labels.itertuples(index=False)}
    rows = []
    for path in sorted(root.rglob("*.txt")):
        x, y, z = extract_coordinates(path.name)
        if x is None or (x, y) not in lookup:
            raise ValueError(f"Missing coordinate or original annotation: {path}")
        rows.append({"file_path": str(path), "filename": path.name,
                     "x": x, "y": y, "z": z, "label": lookup[(x, y)], "label_found": True})
    if not rows:
        raise ValueError(f"No measurement files found: {root}")
    frame = pd.DataFrame(rows)
    if frame.duplicated(["x", "y"]).any():
        raise ValueError("Multiple measurements map to the same rounded coordinates")
    if set(zip(frame.x, frame.y)) != set(lookup):
        raise ValueError("Annotations and measurement coordinates are not bijective")
    return frame.sort_values(["x", "y"], kind="stable").reset_index(drop=True)


def label_dataset(measurement_root, label_file, output_file):
    output_file = Path(output_file)
    if output_file.exists():
        raise FileExistsError(output_file)
    frame = build_labeled_dataset(measurement_root, label_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x") as stream:
        frame.to_csv(stream, index=False)
    print(f"{output_file.name}: {len(frame)} measurements, all original annotations found")
    return frame


def main():
    for measurement_root, label_file, output_file in DATASETS:
        label_dataset(measurement_root, label_file, output_file)


if __name__ == "__main__":
    main()
