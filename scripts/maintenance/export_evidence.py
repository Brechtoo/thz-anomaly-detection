"""Copy an explicit small-result allowlist; never train or alter source runs.

CSV projections deliberately omit paths and error tracebacks. The manifest keeps
both source and exported hashes so a projection cannot be mistaken for an exact
copy. Invoke as a module from the repository root.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def export_evidence(root, specification, destination):
    root, destination = Path(root).resolve(), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    prepared, records = [], []
    targets = set()
    for item in specification:
        source = (root / item['source']).resolve()
        target = Path(item['destination'])
        if not source.is_relative_to(root) or target.is_absolute() or '..' in target.parts:
            raise ValueError('Evidence paths must stay within the declared roots')
        if target in targets or str(target) == 'manifest.json':
            raise ValueError('Duplicate or reserved evidence destination')
        targets.add(target)
        if source.suffix not in {'.csv', '.json', '.png'} or source.stat().st_size > 5_000_000:
            raise ValueError(f'Not a compact result: {item["source"]}')
        content = source.read_bytes()
        exported = content
        columns = item.get('columns')
        if columns:
            if source.suffix != '.csv':
                raise ValueError('Only CSV supports a column projection')
            reader = csv.DictReader(io.StringIO(content.decode()))
            if not set(columns) <= set(reader.fieldnames or []):
                raise ValueError(f'Missing selected columns: {item["source"]}')
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows({key: row[key] for key in columns} for row in reader)
            exported = stream.getvalue().encode()
        if source.suffix != '.png' and any(token in exported for token in [b'/Users/', b'/home/', b'file:///']):
            raise ValueError(f'Local path in selected evidence: {item["source"]}')
        prepared.append((target, exported))
        records.append(dict(source=item['source'], destination=target.as_posix(),
                            source_sha256=digest(content), exported_sha256=digest(exported),
                            bytes=len(exported), columns=columns))
    # Validate every input before creating any output directory.
    destination.mkdir(parents=True, exist_ok=False)
    for target, content in prepared:
        path = destination / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (destination / 'manifest.json').write_text(json.dumps(dict(format_version=1, files=records), indent=2) + '\n')
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--specification', type=Path, default=Path('configs/review_evidence.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = export_evidence(args.root, json.loads(args.specification.read_text()), args.output_dir)
    print(f'Exported {len(result)} small result files ({sum(r["bytes"] for r in result):,} bytes)')
