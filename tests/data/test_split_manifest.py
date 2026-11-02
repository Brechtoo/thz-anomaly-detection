import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.split_manifest import restore_splits, sha256_file


@pytest.fixture
def dataset():
    return pd.DataFrame({"instance_id": list("abcdef"), "label": [0, 1, 2, 3, 0, 1], "f": [10, 11, 12, 13, 14, 15]})


def write_manifest(folder, groups):
    folder.mkdir()
    files = {}
    for name, rows in groups.items():
        p = folder / f"{name}.csv"
        pd.DataFrame(rows, columns=["instance_id", "label"]).to_csv(p, index=False)
        files[p.name] = sha256_file(p)
    (folder / "manifest.json").write_text(json.dumps({"format_version": 1, "files": files}))
    return folder


@pytest.fixture
def groups():
    return {"train": [("e", 0), ("b", 1)], "validation": [("d", 3), ("a", 0)], "test": [("f", 1), ("c", 2)]}


def restore(df, folder):
    return restore_splits(df[["f"]], df.label, df.instance_id, folder)


def test_restores_exact_order_despite_input_reordering(dataset, groups, tmp_path):
    folder = write_manifest(tmp_path / "ids", groups)
    parts = restore(dataset.sample(frac=1, random_state=7).reset_index(drop=True), folder)
    assert [part.f.tolist() for part in parts[:3]] == [[14, 11], [13, 10], [15, 12]]
    assert [part.tolist() for part in parts[3:]] == [[0, 1], [3, 0], [1, 2]]


@pytest.mark.parametrize("problem", ["duplicate", "overlap", "unknown", "missing", "label"])
def test_rejects_inconsistent_memberships(dataset, groups, tmp_path, problem):
    if problem == "duplicate":
        groups["train"].append(("e", 0))
    elif problem == "overlap":
        groups["validation"].append(("e", 0))
    elif problem == "unknown":
        groups["test"][0] = ("unknown", 1)
    elif problem == "missing":
        groups["test"].pop()
    else:
        groups["train"][0] = ("e", 2)
    folder = write_manifest(tmp_path / "ids", groups)
    with pytest.raises(ValueError):
        restore(dataset, folder)


@pytest.mark.parametrize("problem", ["duplicate_id", "null_id", "misaligned_y", "duplicate_index"])
def test_rejects_ambiguous_dataset(dataset, groups, tmp_path, problem):
    folder = write_manifest(tmp_path / "ids", groups)
    y = dataset.label.copy()
    if problem == "duplicate_id":
        dataset.loc[0, "instance_id"] = "b"
    elif problem == "null_id":
        dataset.loc[0, "instance_id"] = None
    elif problem == "misaligned_y":
        y.index = list(reversed(y.index))
    else:
        dataset.index = [0] * len(dataset)
    with pytest.raises(ValueError):
        restore_splits(dataset[["f"]], y, dataset.instance_id, folder)


def test_detects_tampered_manifest_file(dataset, groups, tmp_path):
    folder = write_manifest(tmp_path / "ids", groups)
    with (folder / "train.csv").open("a") as f:
        f.write("x,0\n")
    with pytest.raises(ValueError, match="Checksum"):
        restore(dataset, folder)


def test_frozen_baseline_ids_are_complete_disjoint_and_hash_verified():
    folder = Path(__file__).resolve().parents[2] / "configs/hgb_baseline/splits"
    manifest = json.loads((folder / "manifest.json").read_text())
    seen = set()
    for name, size in [("train", 34042), ("validation", 8511), ("test", 10639)]:
        p = folder / f"{name}.csv"
        assert sha256_file(p) == manifest["files"][p.name]
        frame = pd.read_csv(p)
        assert len(frame) == size
        assert frame.instance_id.is_unique
        assert not seen.intersection(frame.instance_id)
        seen.update(frame.instance_id)
    assert len(seen) == 53192
