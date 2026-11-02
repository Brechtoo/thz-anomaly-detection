"""Read-only leakage diagnostics; these functions never assign new splits."""
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SPLITS = ("train", "validation", "test")
ACQUISITION_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)-"
    r"(?P<scan_name>.+?)-\[(?P<sequence>\d+)\]-\[(?P<coordinates>[^\]]+)\]-"
    r"\[(?P<orientation>[^\]]+)\]-delta\[(?P<delta>[^\]]+)\]-avg(?P<average>\d+)\.txt$"
)


def parse_acquisition_metadata(filenames):
    """Filename-derived scan names/conditions, not authoritative session IDs."""
    result = filenames.str.extract(ACQUISITION_PATTERN)
    if result.isna().any().any():
        raise ValueError("Unrecognized acquisition filename; do not silently drop rows")
    result["sequence"] = result.sequence.astype(int)
    result["average"] = result.average.astype(int)
    result["timestamp"] = pd.to_datetime(result.timestamp, format="%Y-%m-%dT%H-%M-%S.%f")
    result["acquisition_date"] = result.timestamp.dt.strftime("%Y-%m-%d")
    return result


def group_overlap_table(frame, group_columns):
    """Count each existing split within a physical or diagnostic group."""
    if frame[[*group_columns, "split"]].isna().any().any():
        raise ValueError("Missing group or split metadata")
    if not frame.split.isin(SPLITS).all():
        raise ValueError("Unknown split name")
    table = frame.groupby([*group_columns, "split"], observed=True).size().unstack("split", fill_value=0)
    table = table.reindex(columns=SPLITS, fill_value=0)
    table["split_count"] = (table > 0).sum(axis=1)
    table["cross_split"] = table.split_count > 1
    return table.reset_index()


def spatial_train_proximity(frame):
    """Nearest training distance in recorded x/y units within the same source.

    This measures spatial dependence risk, not independent sample size or proof
    of a particular signal correlation. No learned model or split is produced.
    """
    rows = []
    for name, group in frame.groupby("source_dataset", sort=False):
        train = group.loc[group.split == "train", ["x", "y"]].to_numpy(dtype=float)
        tree = cKDTree(train) if len(train) else None
        for split in ("validation", "test"):
            part = group.loc[group.split == split]
            if part.empty:
                continue
            distances = tree.query(part[["x", "y"]].to_numpy(dtype=float))[0] if tree else np.full(len(part), np.inf)
            for instance_id, distance in zip(part.instance_id, distances):
                rows.append({"instance_id": instance_id, "source_dataset": name,
                             "split": split, "nearest_train_xy_distance": float(distance)})
    return pd.DataFrame(rows, columns=["instance_id", "source_dataset", "split", "nearest_train_xy_distance"])
