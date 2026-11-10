"""Paired class-2 ablation on frozen training-body folds; old holdouts stay closed.

Only the training label set changes. Three-class labels are reversibly encoded
for XGBoost's contiguous-label requirement; original labels stay in all outputs.
"""
import argparse
import importlib.metadata
import json
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from threadpoolctl import threadpool_limits

from src.data.reference_measurements import require_no_references
from src.data.split_manifest import sha256_file
from src.experiments.anchor_group_cv import SPLITS, fold_pipeline
from src.experiments.model_registry import build_models
from src.experiments.provenance import capture_git_state, snapshot_source_tree
from src.paths import PROJECT_ROOT

KNOWN = np.array([0, 1, 3])
ARMS = {"four_class": np.array([0, 1, 2, 3]), "three_class": KNOWN}


def training_indices(data, fold, classes):
    """Exclude both the held-out bodies and disallowed labels before any fit."""
    return np.flatnonzero((data.fold.to_numpy() != fold) & data.label.isin(classes).to_numpy())


def encode_labels(y, classes):
    y = np.asarray(y)
    if not np.isin(y, classes).all():
        raise ValueError("Label outside this training arm")
    return np.searchsorted(classes, y)


def original_probabilities(model, X, classes):
    """Return columns 0/1/2/3, with exactly zero probability for absent class 2."""
    encoded = np.asarray(model.classes_, dtype=int)
    p = np.zeros((len(X), 4))
    p[:, np.asarray(classes)[encoded]] = model.predict_proba(X)
    return p


def restricted_prediction(probabilities):
    """Diagnostic only: forbid output 2 without retraining the four-class model."""
    return KNOWN[np.argmax(np.asarray(probabilities)[:, KNOWN], axis=1)]


def known_metrics(y, pred):
    """Fixed 0/1/3 evaluation: output 2 is an error, never silently dropped.

    F1 averages only the three target classes, including their false negatives
    caused by output 2. Balanced accuracy is mean recall over present targets.
    Full 3x4 counts retain the extra output; no nonexistent fourth truth class
    is added to the primary macro average.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    if not np.isin(y, KNOWN).all():
        raise ValueError("Evaluation requires true labels 0/1/3 only")
    if not len(y):
        raise ValueError("Empty evaluation")
    _, recall, f1, support = precision_recall_fscore_support(y, pred, labels=KNOWN, zero_division=0)
    out = dict(points=len(y), accuracy=float(np.mean(y == pred)),
               macro_f1=float(np.mean(f1)),
               balanced_accuracy=float(np.mean(recall[support > 0])),
               output_2_count=int(np.sum(pred == 2)))
    for label, r, n in zip(KNOWN, recall, support):
        out[f"support_{label}"] = int(n)
        out[f"recall_{label}"] = float(r) if n else np.nan
    events = {"critical_1": (y == 1) & (pred == 3),
              "false_alarm_1": (y == 3) & (pred == 1),
              "false_alarm_2": (y == 3) & (pred == 2),
              "false_alarm": (y == 3) & np.isin(pred, [1, 2])}
    for key, mask in events.items():
        n = int(np.sum(y == (1 if key == "critical_1" else 3)))
        out[key + "_count"] = int(mask.sum())
        out[key + "_rate"] = float(mask.sum() / n) if n else np.nan
    return out


def load_inputs(source):
    source = Path(source)
    manifest = json.loads((source / "extraction_manifest.json").read_text())
    checks = {source / "cohort_folds.csv": manifest["cohort_sha256"],
              source / "features/early_legacy_windows.npz": manifest["features_sha256"]["early_legacy_windows.npz"],
              source / "plan.json": manifest["plan_sha256"],
              SPLITS / "train.csv": manifest["train_sha256"]}
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    data = pd.read_csv(source / "cohort_folds.csv")
    actual = data[["instance_id", "label"]].sort_values("instance_id").reset_index(drop=True)
    expected = pd.read_csv(SPLITS / "train.csv").sort_values("instance_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)
    if len(data) != 34235 or not data.instance_id.is_unique or data.specimen_id.nunique() != 23:
        raise ValueError("Unexpected original train cohort")
    if not data.instance_id.is_monotonic_increasing or set(data.fold) != {0, 1, 2}:
        raise ValueError("Unexpected frozen fold order")
    require_no_references(data)
    for fold in range(3):
        if set(data.specimen_id[data.fold == fold]) & set(data.specimen_id[data.fold != fold]):
            raise ValueError("Group leakage")
        for classes in ARMS.values():
            if set(data.label.iloc[training_indices(data, fold, classes)]) != set(classes):
                raise ValueError("Missing training class")
    features = json.loads((source / "plan.json").read_text())["features"]
    with np.load(source / "features/early_legacy_windows.npz") as saved:
        X = saved["X"].copy()
    if X.shape != (len(data), 52) or len(features) != 52 or np.isinf(X).any():
        raise ValueError("Unexpected feature schema")
    return data, X, features, checks


def prepare_output(output, source):
    if output.exists():
        raise FileExistsError(output)
    data, X, features, checks = load_inputs(source)
    git_state = capture_git_state(PROJECT_ROOT)
    # Hash existing artifacts without opening their tracking stores or holdout data.
    paths = [p for root in ["data", "configs", "results", "src"]
             for p in (PROJECT_ROOT / root).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    print(f"Protecting {len(paths)} existing files (hashes only)", flush=True)
    protected = {str(p): sha256_file(p) for p in sorted(paths)}
    output.mkdir(parents=True, exist_ok=False)
    for directory in ["models", "predictions", "metadata"]:
        (output / directory).mkdir()
    plan = dict(seed=42, source=str(source.resolve()), features=features,
                feature_variant="early_legacy_windows", folds="unchanged source/cohort_folds.csv",
                arms={k: v.tolist() for k, v in ARMS.items()},
                evaluation="identical OOF true 0/1/3; fixed target macro; output2 counts as error",
                false_alarm="true3 to1/2 primary; also report to1 and to2 separately",
                diagnostic="four-class output restricted to0/1/3 by probability argmax, no tuning",
                population_analysis="fold-training known-class median/scaler/PCA(5); GMM1/2 diagonal on train-class2; n_init5, reg_covar1e-4; held-out class2 log density; no model or threshold selection",
                uncertainty="uncalibrated probabilities; descriptive mixtures do not identify physical truth",
                scope="original 23 training bodies only; no old validation/test use")
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (output / "protected_inputs.json").write_text(json.dumps(protected, indent=2) + "\n")
    (output / "input_hashes.json").write_text(json.dumps({str(p): h for p, h in checks.items()}, indent=2) + "\n")
    data.to_csv(output / "cohort_folds.csv", index=False)
    snapshot_source_tree(PROJECT_ROOT / "src", output / "metadata/source")
    (output / "metadata/git.json").write_text(json.dumps(git_state, indent=2) + "\n")
    (output / "metadata/versions.json").write_text(json.dumps({d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}, indent=2) + "\n")
    return data, X


def summarize_predictions(output):
    rows = []
    for path in sorted((output / "predictions").glob("*.csv")):
        frame = pd.read_csv(path)
        arm, name = path.stem.split("__")
        for view in ["native", "restricted"] if arm == "four_class" else ["native"]:
            p = frame.prediction.to_numpy() if view == "native" else restricted_prediction(frame[[f"p_{i}" for i in range(4)]].to_numpy())
            known = frame.label.isin(KNOWN)
            part = frame.loc[known].copy()
            part["scored_prediction"] = p[known]
            cm = confusion_matrix(part.label, part.scored_prediction, labels=[0, 1, 2, 3])[[0, 1, 3]]
            pd.DataFrame(cm, index=KNOWN, columns=range(4)).to_csv(output / f"cm_{arm}_{view}_{name}.csv")
            groups = [("pooled", "all", part)]
            for field in ["fold", "specimen_id", "source_dataset"]:
                groups.extend((field, key, g) for key, g in part.groupby(field))
            for level, key, g in groups:
                rows.append(dict(arm=arm, view=view, model=name, level=level, group=key,
                                 **known_metrics(g.label, g.scored_prediction)))
    pd.DataFrame(rows).to_csv(output / "known_metrics.csv", index=False)


def run(source, output):
    source, output = Path(source), Path(output)
    data, X = prepare_output(output, source)
    mlflow.set_tracking_uri("sqlite:///" + str((output / "mlflow.db").resolve()))
    client = mlflow.MlflowClient()
    experiment = client.create_experiment("class2-training-bodies-group-cv", artifact_location=(output / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(experiment_id=experiment)
    mlflow.sklearn.autolog(disable=True)
    records = []
    oof = {(arm, name): np.full((len(data), 4), np.nan) for arm in ARMS for name in build_models()}
    predictions = {key: np.full(len(data), -1) for key in oof}
    for fold in range(3):
        va = np.flatnonzero(data.fold.to_numpy() == fold)
        for arm, classes in ARMS.items():
            tr = training_indices(data, fold, classes)
            y = encode_labels(data.label.to_numpy()[tr], classes)
            for name, factory in build_models(seed=42).items():
                print(f"Fold {fold + 1}/3 {arm} {name}: train={len(tr)}, validation={len(va)}", flush=True)
                model = fold_pipeline(factory)
                params = {k: str(v) for k, v in factory.get_params(deep=True).items()}
                with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                    warnings.simplefilter("always")
                    start = time.perf_counter()
                    model.fit(X[tr], y)
                    pred = classes[np.asarray(model.predict(X[va]), dtype=int)]
                    proba = original_probabilities(model, X[va], classes)
                elapsed = time.perf_counter() - start
                predictions[(arm, name)][va] = pred
                oof[(arm, name)][va] = proba
                known = data.label.to_numpy()[va] != 2
                metrics = known_metrics(data.label.to_numpy()[va][known], pred[known])
                tag = f"{arm}__{name}__fold{fold}"
                model_path = output / "models" / f"{tag}.joblib"
                joblib.dump(model, model_path)
                detail = dict(parameters=params, original_classes=classes.tolist(), training_points=len(tr),
                              training_groups=int(data.specimen_id.iloc[tr].nunique()), seconds=elapsed,
                              warnings=[str(w.message) for w in caught],
                              imputer_statistics=model.named_steps["imputer"].statistics_.tolist(),
                              retained_feature_indices=model.named_steps["constants"].get_support(indices=True).tolist())
                detail_path = model_path.with_suffix(".json")
                detail_path.write_text(json.dumps(detail, indent=2) + "\n")
                with mlflow.start_run(run_name=tag) as run_info:
                    mlflow.log_params(params)
                    mlflow.log_params(dict(arm=arm, original_classes=str(classes.tolist()), fold=fold, seed=42, train_points=len(tr)))
                    mlflow.set_tags(dict(scope="original_train_bodies_only", feature_variant="early_legacy_windows", no_old_holdout_used=True))
                    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if np.isfinite(v)})
                    for p in [model_path, detail_path, output / "plan.json", output / "cohort_folds.csv", output / "input_hashes.json"]:
                        mlflow.log_artifact(str(p), "experiment")
                    records.append(dict(fold=fold, arm=arm, model=name, run_id=run_info.info.run_id, seconds=elapsed, **metrics))
                pd.DataFrame(records).to_csv(output / "fold_runs.csv", index=False)
    for key, p in oof.items():
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
            raise ValueError("Incomplete OOF probabilities")
        frame = data.drop(columns="file_path").copy()
        frame["prediction"] = predictions[key]
        for label in range(4):
            frame[f"p_{label}"] = p[:, label]
        frame.to_csv(output / "predictions" / f"{key[0]}__{key[1]}.csv", index=False)
    summarize_predictions(output)
    print("All 42 fits complete. Existing validation/test bodies were not evaluated.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "results/experiments/anchor_group_cv_v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.source, args.output)
