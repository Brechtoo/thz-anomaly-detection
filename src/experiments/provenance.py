"""Local provenance capture without opening or modifying MLflow runs."""

import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path


def snapshot_source_tree(source_root, destination):
    """Preserve sources without recursively copying local tracking stores/caches.

    Existing destinations are refused. Historical snapshots and artifact locations
    are never moved or rewritten by this helper.
    """
    return shutil.copytree(
        source_root, destination,
        ignore=shutil.ignore_patterns(
            '__pycache__', 'mlruns', 'mlartifacts', '.DS_Store',
            '*.pyc', '*.db', '*.db-shm', '*.db-wal',
        ),
    )


def capture_git_state(project_root):
    root = Path(project_root)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True)

    # Capture before creating output artifacts, with respect to the repo root.
    status = git("status", "--porcelain", "--untracked-files=all")
    return {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(status.strip()),
        "status": status,
        "tracked_diff": git("diff", "--binary", "HEAD"),
        "untracked_paths": git("ls-files", "--others", "--exclude-standard").splitlines(),
    }


def check_locked_environment(lock_path):
    """Fail before training on a version mismatch; never installs packages."""
    mismatches = []
    if platform.python_version() != "3.12.5":
        mismatches.append(f"Python: expected 3.12.5, got {platform.python_version()}")
    for line in Path(lock_path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            mismatches.append(f"{name}: expected {expected}, got {actual}")
    if mismatches:
        raise ValueError("Dependency lock mismatch: " + "; ".join(mismatches))
