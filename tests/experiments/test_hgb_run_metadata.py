"""Exercise the entry point with fake training and fake MLflow only."""

from contextlib import nullcontext
import json

import pandas as pd
import pytest

from src.data.split_manifest import sha256_file
from src.experiments import run_experiment as experiment


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    data = pd.DataFrame({
        "instance_id": [f"id{i}" for i in range(12)], "label": [0, 1, 2, 3] * 3,
        "f": [0, 1, 2, 3] * 3, "file_path": ["synthetic"] * 12,
        "x": [0] * 12, "y": [0] * 12, "source_dataset": ["synthetic"] * 12,
    })
    input_file = tmp_path / "data.csv"
    data.to_csv(input_file, index=False)
    folder = tmp_path / "splits"
    folder.mkdir()
    files = {}
    for i, name in enumerate(("train", "validation", "test")):
        p = folder / f"{name}.csv"
        data.iloc[i * 4:(i + 1) * 4][["instance_id", "label"]].to_csv(p, index=False)
        files[p.name] = sha256_file(p)
    (folder / "manifest.json").write_text(json.dumps({
        "format_version": 1, "feature_table_sha256": sha256_file(input_file),
        "feature_columns": ["f"], "files": files, "source_run_id": "reference",
        "sizes": {"train": 4, "validation": 4, "test": 4},
    }))
    lock = tmp_path / "requirements-lock.txt"
    lock.write_text("# synthetic environment\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text("# exact source snapshot\n")
    for attr, value in {
        "PROJECT_ROOT": tmp_path, "FEATURE_TABLE_ROOT": input_file,
        "SPLIT_MANIFEST_DIR": folder, "DEPENDENCY_LOCK": lock,
        "RESULTS_DIR": tmp_path / "outputs",
    }.items():
        monkeypatch.setattr(experiment, attr, value)
    monkeypatch.setattr(experiment, "capture_git_state", lambda root: {
        "git_commit": "abc123", "git_dirty": True, "tracked_diff": "patch", "status": " M example.py",
    })
    monkeypatch.setattr(experiment, "check_locked_environment", lambda path: None)
    calls = {"tags": [], "training": 0, "runs": 0}
    class FakeModel:
        def fit(self, *args, **kwargs):
            pytest.fail("Real fit must never run in this test")
        def predict(self, X):
            return X.f.to_numpy()
    monkeypatch.setattr(experiment, "create_hgb", FakeModel)
    def fake_train(model, X, y):
        calls["training"] += 1
        assert X.index.tolist() == [0, 1, 2, 3]
        return model
    monkeypatch.setattr(experiment, "train_hgb", fake_train)
    monkeypatch.setattr(experiment, "save_model", lambda *args: None)
    def start_run(**kwargs):
        calls["runs"] += 1
        return nullcontext()
    for attr in ("set_tracking_uri", "set_experiment", "log_artifacts", "log_params", "log_dict", "log_metrics"):
        monkeypatch.setattr(experiment.mlflow, attr, lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment.mlflow, "start_run", start_run)
    monkeypatch.setattr(experiment.mlflow, "active_run", lambda: None)
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setattr(experiment.mlflow, "set_tags", lambda tags: calls["tags"].append(tags))
    monkeypatch.setattr(experiment.mlflow.sklearn, "autolog", lambda **kwargs: None)
    return tmp_path, calls


def test_runs_log_provenance_and_use_separate_output_directories(fake_run):
    root, calls = fake_run
    experiment.run_experiment()
    experiment.run_experiment()  # two simulated runs; no MLflow writes or fits
    outputs = list((root / "outputs").iterdir())
    assert len(outputs) == 2
    assert calls["training"] == calls["runs"] == 2
    for tags in calls["tags"]:
        assert tags["git_commit"] == "abc123"
        assert tags["git_dirty"] == "True"
        assert tags["error_metrics_version"] == "four_label_v2"
        assert tags["split_source_run_id"] == "reference"
        assert len(tags["split_manifest_sha256"]) == 64
    for output in outputs:
        assert (output / "metadata/source/src/example.py").read_text() == "# exact source snapshot\n"
        assert (output / "metadata/splits/manifest.json").is_file()
        assert (output / "metadata/requirements-lock.txt").is_file()
        result = pd.read_csv(output / "validation_results.csv")
        assert result.accuracy.iloc[0] == 1.0


def test_changed_feature_table_fails_before_run_or_training(fake_run):
    root, calls = fake_run
    with (root / "data.csv").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="Feature-table hash"):
        experiment.run_experiment()
    assert calls["runs"] == calls["training"] == 0
    assert not (root / "outputs").exists()


@pytest.mark.parametrize("existing_run", ["environment", "active"])
def test_refuses_to_resume_an_existing_mlflow_run(fake_run, monkeypatch, existing_run):
    root, calls = fake_run
    if existing_run == "environment":
        monkeypatch.setenv("MLFLOW_RUN_ID", "historical-run")
    else:
        monkeypatch.setattr(experiment.mlflow, "active_run", lambda: object())
    with pytest.raises(ValueError, match="requires a new MLflow run"):
        experiment.run_experiment()
    assert calls["runs"] == calls["training"] == 0
    assert not (root / "outputs").exists()
