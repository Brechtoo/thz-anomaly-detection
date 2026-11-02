"""Restore ordered, disjoint splits by immutable measurement IDs."""

import hashlib
import json
from pathlib import Path

import pandas as pd

SPLIT_NAMES = ("train", "validation", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restore_splits(X, y, instance_ids, manifest_dir):
    """Return the create_splits tuple in saved order, independent of row order.

    Inputs must have identical unique indexes. Saved labels and complete ID
    coverage are checked, so missing/extra IDs or label changes fail closed.
    Feature-table content identity is checked separately by the experiment.
    """
    if not X.index.is_unique or not X.index.equals(y.index) or not X.index.equals(instance_ids.index):
        raise ValueError("X, y and instance_ids require identical unique indexes")
    if instance_ids.isna().any() or not instance_ids.is_unique:
        raise ValueError("instance_ids must be unique and non-null")
    directory = Path(manifest_dir)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("format_version") != 1:
        raise ValueError("Unsupported split manifest format")
    # IDs are strings on disk, never positional row numbers.
    ids = instance_ids.astype(str)
    if not ids.is_unique:
        raise ValueError("instance_ids must remain unique as strings")
    index_by_id = pd.Series(X.index, index=ids.to_numpy())
    seen = set()
    frames = []
    labels = []
    for name in SPLIT_NAMES:
        path = directory / f"{name}.csv"
        if sha256_file(path) != manifest["files"][path.name]:
            raise ValueError(f"Checksum mismatch: {path.name}")
        saved = pd.read_csv(path, dtype={"instance_id": str})
        if list(saved.columns) != ["instance_id", "label"]:
            raise ValueError(f"Invalid columns: {path.name}")
        if saved.instance_id.isna().any() or not saved.instance_id.is_unique:
            raise ValueError(f"Duplicate or null IDs in {name}")
        members = set(saved.instance_id)
        if members & seen:
            raise ValueError("Split IDs overlap")
        if not members <= set(ids):
            raise ValueError("Split contains unknown IDs")
        seen.update(members)
        positions = index_by_id.loc[saved.instance_id].to_list()
        split_y = y.loc[positions]
        if not split_y.reset_index(drop=True).equals(saved.label.rename(split_y.name)):
            # Accept equivalent numeric label dtypes, but never missing labels.
            if split_y.isna().any() or not (split_y.to_numpy() == saved.label.to_numpy()).all():
                raise ValueError(f"Labels changed in {name}")
        frames.append(X.loc[positions])
        labels.append(split_y)
    if seen != set(ids):
        raise ValueError("Split manifests do not cover all dataset IDs")
    return (*frames, *labels)
