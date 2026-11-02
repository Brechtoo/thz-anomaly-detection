import importlib
from pathlib import Path

import pandas as pd
import pytest

from src.data.labeling.label_data import build_labeled_dataset, extract_coordinates, label_dataset
from src.data.labeling.data_combining import combine_labeled_data
from src.data.labeling.rebuild_catalog import rebuild_catalog, DATASET_SPECS
from src.data.split_manifest import sha256_file


def make_source(root, annotation_text="0 0 1\n1 0 3\n"):
    scan = root / "scan"
    scan.mkdir(parents=True)
    # Reverse filename order relative to coordinates, as in a scan traversal.
    (scan / "a-[1.0,0.0,2.1]-[1,0,0,0].txt").write_text("0 1\n")
    (scan / "z-[0.0,0.0,2.2]-[1,0,0,0].txt").write_text("0 2\n")
    annotation = root / "labels.txt"
    annotation.write_text(annotation_text)
    return scan, annotation


def test_original_coordinate_rounding_is_preserved():
    assert extract_coordinates("scan-[0.5,-1.5,3.25]-[1,0,0,0].txt") == (0, -2, 3.25)
    assert extract_coordinates("no_coordinates.txt") == (None, None, None)


def test_annotation_assignment_and_order_ignore_enumeration_order(tmp_path, monkeypatch):
    scan, annotations = make_source(tmp_path)
    original_bytes = annotations.read_bytes()
    first = build_labeled_dataset(scan, annotations)
    original_rglob = Path.rglob
    def reverse_enumeration(path, pattern):
        return iter(reversed(list(original_rglob(path, pattern))))
    monkeypatch.setattr(Path, "rglob", reverse_enumeration)
    second = build_labeled_dataset(scan, annotations)
    pd.testing.assert_frame_equal(first, second)
    assert first.x.tolist() == [0, 1]
    assert first.label.tolist() == [1, 3]
    assert first.label_found.all()
    assert annotations.read_bytes() == original_bytes


@pytest.mark.parametrize("annotation_text", [
    "0 0 1\n0 0 2\n1 0 3\n",  # ambiguous annotation
    "0 0 1\n",                   # missing original label
    "0 0 1\n1 0 3\n2 0 1\n",  # annotation without measurement
    "0 0 4\n1 0 3\n",           # invalid label
    "0.4 0 1\n1 0 3\n",         # no silent truncation
])
def test_ambiguous_or_incomplete_annotations_fail(tmp_path, annotation_text):
    scan, annotations = make_source(tmp_path, annotation_text)
    with pytest.raises(ValueError):
        build_labeled_dataset(scan, annotations)


def test_two_raw_files_cannot_share_a_rounded_coordinate(tmp_path):
    scan, annotations = make_source(tmp_path)
    (scan / "another-[0.1,0.0,4.0].txt").write_text("0 3\n")
    with pytest.raises(ValueError, match="same rounded coordinates"):
        build_labeled_dataset(scan, annotations)


def test_combining_preserves_source_and_row_order_without_mutation(tmp_path):
    scan, annotation = make_source(tmp_path)
    frame = build_labeled_dataset(scan, annotation)
    original = frame.copy(deep=True)
    result = combine_labeled_data([("source", frame)])
    assert result.instance_id.tolist() == ["source_x0_y0", "source_x1_y0"]
    assert "label_found" not in result
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="unique"):
        combine_labeled_data([("source", frame), ("source", frame)])
    with pytest.raises(ValueError, match="Duplicate measurement"):
        combine_labeled_data([("a", frame), ("b", frame)])


def test_rebuild_is_byte_deterministic_and_never_overwrites(tmp_path):
    raw = tmp_path / "raw"
    annotation_hashes = {}
    for name, folder, filename in DATASET_SPECS:
        scan, annotation = make_source(raw / folder)
        target = annotation.with_name(filename)
        annotation.rename(target)
        annotation_hashes[target] = sha256_file(target)
    first = rebuild_catalog(raw, tmp_path / "first")
    second = rebuild_catalog(raw, tmp_path / "second")
    assert sha256_file(first) == sha256_file(second)
    df = pd.read_csv(first)
    assert df.instance_id.tolist() == [f"{name}_x{x}_y0" for name, _, _ in DATASET_SPECS for x in [0, 1]]
    assert df.label.tolist() == [1, 3] * 3
    with pytest.raises(FileExistsError):
        rebuild_catalog(raw, tmp_path / "first")
    assert all(sha256_file(p) == h for p, h in annotation_hashes.items())


def test_legacy_label_writer_refuses_existing_file(tmp_path):
    output = tmp_path / "existing.csv"
    output.write_text("preserve this\n")
    with pytest.raises(FileExistsError):
        label_dataset(tmp_path, tmp_path / "missing_labels", output)
    assert output.read_text() == "preserve this\n"


def test_combining_import_has_no_io(monkeypatch):
    from src.data.labeling import data_combining
    def unexpected(*args, **kwargs):
        pytest.fail("Import must not read/write dataset files")
    monkeypatch.setattr(pd, "read_csv", unexpected)
    monkeypatch.setattr(pd.DataFrame, "to_csv", unexpected)
    importlib.reload(data_combining)
