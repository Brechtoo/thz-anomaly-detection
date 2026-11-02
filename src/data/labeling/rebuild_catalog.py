"""Deterministically reconstruct the catalog from existing raw files/annotations."""
import argparse
import json
from pathlib import Path
from src.data.labeling.label_data import build_labeled_dataset
from src.data.labeling.data_combining import combine_labeled_data
from src.data.split_manifest import sha256_file
from src.experiments.provenance import capture_git_state
from src.paths import PROJECT_ROOT

DATASET_SPECS = [
    ("messreihe_1", "Hohlraumtestk1_tr", "hohltestk1_tr_markArea_XYList2.txt"),
    ("messreihe_2", "Hohlraumtestk2_tr", "hohltestk2_tr_markArea_XYList.txt"),
    ("messreihe_3", "Hohlraumtestk3_tr", "hohltestk3_tr_markArea_XYList.txt"),
]


def rebuild_catalog(raw_root, output_dir):
    raw_root, output_dir = Path(raw_root).resolve(), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    annotation_hashes = {}
    datasets = []
    for name, folder, filename in DATASET_SPECS:
        annotation = raw_root / folder / filename
        annotation_hashes[str(annotation)] = sha256_file(annotation)
        datasets.append((name, build_labeled_dataset(raw_root / folder / "scan", annotation)))
    combined = combine_labeled_data(datasets)
    if any(sha256_file(path) != value for path, value in annotation_hashes.items()):
        raise ValueError("Annotations changed during reconstruction")
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, frame in datasets:
        frame.to_csv(output_dir / f"{name}_labeled.csv", index=False)
    target = output_dir / "combined_dataset.csv"
    combined.to_csv(target, index=False)
    source_hashes = {str(p.relative_to(PROJECT_ROOT)): sha256_file(p)
                     for p in (PROJECT_ROOT / "src/data/labeling").glob("*.py")}
    manifest = {"format_version": 1, "annotation_sha256": annotation_hashes,
                "catalog_sha256": sha256_file(target), "rows": len(combined),
                "source_sha256": source_hashes, "git_state": capture_git_state(PROJECT_ROOT),
                "source_order": [name for name, _, _ in DATASET_SPECS],
                "raw_root": str(raw_root), "coordinates": "Python round, unchanged",
                "labels": "Only original annotations; no inference or fallback",
                "ordering": "dataset specification order; within each dataset ascending x,y"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(rebuild_catalog(args.raw_root, args.output_dir))


if __name__ == "__main__":
    main()
