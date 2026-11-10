import importlib.metadata

import pytest

from src.experiments import provenance


def test_git_capture_includes_staged_unstaged_and_untracked_state(monkeypatch, tmp_path):
    calls = []
    def git(command, text):
        assert command[:3] == ["git", "-C", str(tmp_path)]
        calls.append(command[3:])
        return {
            "status": " M tracked.py\nA  staged.py\n?? new.py\n",
            "rev-parse": "abc123\n", "diff": "actual tracked patch",
            "ls-files": "new.py\n",
        }[command[3]]
    monkeypatch.setattr(provenance.subprocess, "check_output", git)
    state = provenance.capture_git_state(tmp_path)
    assert state["git_commit"] == "abc123"
    assert state["git_dirty"] is True
    assert state["tracked_diff"] == "actual tracked patch"
    assert state["untracked_paths"] == ["new.py"]
    assert ["status", "--porcelain", "--untracked-files=all"] in calls
    assert ["diff", "--binary", "HEAD"] in calls


def test_clean_git_state(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance.subprocess, "check_output", lambda cmd, text: "abc\n" if "rev-parse" in cmd else "")
    assert provenance.capture_git_state(tmp_path)["git_dirty"] is False


@pytest.mark.parametrize("actual", ["1.0", "2.0", None])
def test_dependency_lock_is_checked_without_installing(monkeypatch, tmp_path, actual):
    lock = tmp_path / "lock.txt"
    lock.write_text("# example\npackage==1.0\n")
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "3.12.5")
    def version(name):
        if actual is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return actual
    monkeypatch.setattr(provenance.importlib.metadata, "version", version)
    if actual == "1.0":
        provenance.check_locked_environment(lock)
    else:
        with pytest.raises(ValueError, match="Dependency lock mismatch"):
            provenance.check_locked_environment(lock)


def test_python_version_is_checked(monkeypatch, tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_text("# empty\n")
    monkeypatch.setattr(provenance.platform, "python_version", lambda: "3.13.0")
    with pytest.raises(ValueError, match="Python"):
        provenance.check_locked_environment(lock)


def test_source_snapshot_excludes_nested_tracking_stores_and_caches(tmp_path):
    source = tmp_path / 'src'
    (source / 'experiments').mkdir(parents=True)
    original = source / 'experiments/run.py'
    original.write_bytes(b'"""Keep exact source bytes."""\n')
    for folder in ['mlruns', 'mlartifacts', '__pycache__']:
        target = source / 'experiments' / folder
        target.mkdir()
        (target / 'old_run.py').write_text('archived source must not recurse')
    (source / 'experiments/mlflow.db').write_bytes(b'unchanged tracking store')
    destination = tmp_path / 'snapshot'
    provenance.snapshot_source_tree(source, destination)
    assert (destination / 'experiments/run.py').read_bytes() == original.read_bytes()
    assert sorted(p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()) == ['experiments/run.py']
    assert (source / 'experiments/mlflow.db').read_bytes() == b'unchanged tracking store'
    with pytest.raises(FileExistsError):
        provenance.snapshot_source_tree(source, destination)
