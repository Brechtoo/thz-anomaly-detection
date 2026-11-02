"""Rebuild features into a new directory, preserving input ID order.

Only explicit CLI execution reads measurements. No extraction runs on import.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.dataset import EXCLUDE_COLUMNS
from src.data.split_manifest import sha256_file
from src.experiments.provenance import capture_git_state, check_locked_environment
from src.features.create_feature_table import create_feature_table
from src.paths import PROJECT_ROOT


def rebuild_feature_table(input_csv, output_dir, *, old_raw_root=None, raw_root=None):
    """Use frozen input metadata; optional prefix relocation never changes IDs."""
    input_csv = Path(input_csv).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if (old_raw_root is None) != (raw_root is None):
        raise ValueError("old_raw_root and raw_root must be supplied together")
    lock = PROJECT_ROOT / "requirements-lock.txt"
    check_locked_environment(lock)
    state = capture_git_state(PROJECT_ROOT)
    input_hash = sha256_file(input_csv)
    df = pd.read_csv(input_csv)
    required = set(EXCLUDE_COLUMNS)
    if not required <= set(df):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(df))}")
    if df.instance_id.isna().any() or not df.instance_id.is_unique:
        raise ValueError("instance_id must be non-null and unique")
    if df.empty or not df.label.isin([0, 1, 2, 3]).all():
        raise ValueError("Nonempty input with original labels 0/1/2/3 required")
    paths = []
    for original in df.file_path:
        path = Path(original)
        if raw_root is not None:
            path = Path(raw_root) / path.relative_to(Path(old_raw_root))
        elif not path.is_absolute():
            path = input_csv.parent / path
        paths.append(path.resolve())
    # Missing inputs fail before creating output; retain source metadata order.
    before = [sha256_file(path) for path in paths]
    df = df.copy()
    df["file_path"] = [str(path) for path in paths]
    feature_df = create_feature_table(df)
    if input_hash != sha256_file(input_csv) or before != [sha256_file(path) for path in paths]:
        raise ValueError("Input metadata or measurement bytes changed during extraction")
    output_dir.mkdir(parents=True, exist_ok=False)
    table_path = output_dir / "feature_table.csv"
    feature_df.to_csv(table_path, index=False)
    pd.DataFrame({"instance_id": df.instance_id, "file_path": df.file_path, "sha256": before}).to_csv(
        output_dir / "raw_inputs.csv", index=False
    )
    sources = {str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
               for path in (PROJECT_ROOT / "src/features").glob("*.py")}
    manifest = {
        "format_version": 1, "input_csv_sha256": input_hash,
        "output_csv_sha256": sha256_file(table_path),
        "raw_inputs_sha256": sha256_file(output_dir / "raw_inputs.csv"),
        "dependency_lock_sha256": sha256_file(lock), "source_sha256": sources,
        "git_state": state, "row_count": len(feature_df),
        "feature_columns": [c for c in feature_df if c not in EXCLUDE_COLUMNS],
        "old_raw_root": str(old_raw_root), "raw_root": str(raw_root),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return table_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Frozen combined dataset CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Must not already exist")
    parser.add_argument("--old-raw-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    path = rebuild_feature_table(args.input, args.output_dir, old_raw_root=args.old_raw_root, raw_root=args.raw_root)
    print(path)


if __name__ == "__main__":
    main()
