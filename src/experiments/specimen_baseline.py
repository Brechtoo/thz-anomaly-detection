"""Seven unchanged registry models on a frozen specimen holdout.

Validation/test are evaluated once per fixed model, without tuning or ranking.
A separate local MLflow database preserves historical tracking stores. Metrics,
ID-linked predictions and fitted pipelines are also persisted independently.
"""
import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import time
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_info, threadpool_limits

from src.data.reference_measurements import require_no_references
from src.data.specimen_splits import load_membership
from src.data.split_manifest import restore_splits, sha256_file
from src.evaluation.metrics import (
    ERROR_METRICS_VERSION,
    calculate_error_counts,
    evaluate_predictions,
)
from src.experiments.model_registry import build_models
from src.experiments.model_serialization import trusted_types_for_model
from src.experiments.provenance import capture_git_state, snapshot_source_tree
from src.experiments.saving import save_model
from src.features.specimen_manual import validate_manual_schema
from src.paths import PROJECT_ROOT


def metrics_for_group(truth, prediction):
    """Reuse four-label error definitions; add fixed-label recall/support.

    A zero error rate with zero denominator is the legacy convention, not
    evidence of safety. Explicit supports make such group results identifiable.
    """
    result = evaluate_predictions(truth, prediction)
    result = {k: float(v) for k, v in result.items()
              if k not in ('confusion_matrix', 'classification_report')}
    result['macro_f1'] = f1_score(truth, prediction, labels=[0, 1, 2, 3], average='macro', zero_division=0)
    result.update(calculate_error_counts(truth, prediction))
    report = classification_report(truth, prediction, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)
    for label in range(4):
        result[f'recall_{label}'] = report[str(label)]['recall']
        result[f'support_{label}'] = report[str(label)]['support']
    result['critical_denominator'] = int(np.isin(truth, [1, 2]).sum())
    result['false_alarm_denominator'] = int(np.sum(np.asarray(truth) == 3))
    return result


def evaluate_holdout(metadata, prediction, output):
    """Persist predictions with group keys, full reports and subgroup metrics."""
    output.mkdir(parents=True, exist_ok=False)
    predictions = metadata[['instance_id', 'specimen_id', 'source_dataset', 'label']].copy()
    predictions['prediction'] = prediction
    predictions.to_csv(output / 'predictions.csv', index=False)
    truth = predictions.label.to_numpy()
    metrics = metrics_for_group(truth, prediction)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    pd.DataFrame(confusion_matrix(truth, prediction, labels=[0, 1, 2, 3]),
                 index=[f'true_{i}' for i in range(4)], columns=[f'pred_{i}' for i in range(4)]).to_csv(output / 'confusion_matrix.csv')
    pd.DataFrame(classification_report(truth, prediction, labels=[0, 1, 2, 3], output_dict=True, zero_division=0)).T.to_csv(output / 'classification_report.csv')
    for key in ('specimen_id', 'source_dataset'):
        rows = []
        for group, frame in predictions.groupby(key, sort=True):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # absent classes are explicit in supports
                rows.append({key: group, 'points': len(frame), **metrics_for_group(frame.label, frame.prediction)})
        pd.DataFrame(rows).to_csv(output / f'by_{key}.csv', index=False)
    return metrics


def run_baseline(split_dir, feature_dir, output_dir, seed=42):
    """Train fixed factories; fail closed on mismatched provenance or features."""
    split_dir, feature_dir, output_dir = map(Path, (split_dir, feature_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if mlflow.active_run() is not None or os.environ.get('MLFLOW_RUN_ID'):
        raise ValueError('Fresh MLflow runs required')
    feature_manifest = json.loads((feature_dir / 'manifest.json').read_text())
    inputs = json.loads((feature_dir / 'extraction_inputs.json').read_text())
    if (inputs['split_sha256'] != sha256_file(split_dir / 'manifest.json')
            or feature_manifest['features_sha256'] != sha256_file(feature_dir / 'features.csv')
            or feature_manifest['input_sha256'] != sha256_file(feature_dir / 'extraction_inputs.json')):
        raise ValueError('Feature provenance mismatch')
    table = pd.read_csv(feature_dir / 'features.csv')
    require_no_references(table)
    columns = feature_manifest['selected_features']
    feature_set = feature_manifest.get('feature_set', 'tsfresh')
    if feature_set == 'manual52':
        validate_manual_schema(columns)
        schema_path = feature_dir / 'feature_schema.json'
        schema = json.loads(schema_path.read_text())
        valid_columns = (columns == schema['features'] and len(set(columns)) == 52
                         and sha256_file(schema_path) == inputs['feature_schema_sha256'])
    else:
        valid_columns = feature_set == 'tsfresh' and all(c.startswith('y__') for c in columns)
    if set(table) != {'instance_id', 'label', *columns} or not valid_columns:
        raise ValueError('Unexpected features or metadata')
    X = table.loc[:, columns]
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Finite train-imputed features required')
    train, val, test, ytrain, yval, ytest = restore_splits(X, table.label, table.instance_id, split_dir)
    metadata = load_membership(split_dir).set_index('instance_id')
    models = build_models(seed=seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_dir = output_dir / 'models'
    model_dir.mkdir()
    provenance = output_dir / 'metadata'
    provenance.mkdir()
    shutil.copytree(split_dir, provenance / 'splits')
    snapshot_source_tree(PROJECT_ROOT / 'src', provenance / 'source')
    shutil.copy2(PROJECT_ROOT / 'requirements-lock.txt', provenance / 'requirements-lock.txt')
    for name in ['manifest.json', 'extraction_inputs.json', 'imputation.csv', 'raw_inputs.csv']:
        shutil.copy2(feature_dir / name, provenance / name)
    if feature_set == 'manual52':
        for name in ['verification.json', 'feature_schema.json', 'imputer.joblib']:
            shutil.copy2(feature_dir / name, provenance / name)
    state = capture_git_state(PROJECT_ROOT)
    (provenance / 'git_state.json').write_text(json.dumps(state, indent=2) + '\n')
    config = {'seed': seed, 'features': columns, 'feature_table_sha256': sha256_file(feature_dir / 'features.csv'),
              'split_manifest_sha256': sha256_file(split_dir / 'manifest.json'),
              'error_metrics_version': ERROR_METRICS_VERSION,
              'evaluation': 'fixed models; validation and test; no tuning; no refit on validation',
              'thread_limit': 1, 'python': platform.python_version(),
              'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
              'threadpools': threadpool_info()}
    (provenance / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    if feature_set == 'manual52':
        config.update({'feature_set': feature_set, 'validation_role': 'primary feature-set comparison',
                       'test_role': 'exploratory comparison only; not for model or feature-set selection'})
        (provenance / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    tracking = output_dir / 'mlflow.db'
    mlflow.set_tracking_uri('sqlite:///' + str(tracking.resolve()))
    mlflow.MlflowClient().create_experiment(
        'thz-specimen-holdout-v1', artifact_location=(output_dir / 'mlartifacts').resolve().as_uri())
    mlflow.set_experiment('thz-specimen-holdout-v1')
    mlflow.sklearn.autolog(disable=True)
    rows = []
    for name, model in models.items():
        print(f'Training {name}', flush=True)
        parameters = {k: str(v) for k, v in model.get_params(deep=True).items()}
        (provenance / f'{name}_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
        try:
            with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                warnings.simplefilter('always')
                start = time.perf_counter()
                model.fit(train, ytrain)
                duration = time.perf_counter() - start
                predictions = {'validation': model.predict(val), 'test': model.predict(test)}
            save_model(model, name, model_dir)
            (provenance / f'{name}_warnings.json').write_text(json.dumps([str(w.message) for w in caught], indent=2) + '\n')
            all_metrics = {}
            for split, frame in [('validation', val), ('test', test)]:
                ids = table.instance_id.loc[frame.index]
                meta = metadata.loc[ids].reset_index()
                metrics = evaluate_holdout(meta, predictions[split], output_dir / 'evaluations' / name / split)
                all_metrics.update({f'{split}_{k}': float(v) for k, v in metrics.items()})
                rows.append({'model': name, 'split': split, 'training_seconds': duration, **metrics})
            pd.DataFrame(rows).to_csv(output_dir / 'results.csv', index=False)
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params(parameters)
                mlflow.set_tags({'group_key': 'specimen_id', 'seed': seed,
                                 'feature_set': feature_set,
                                 'test_role': config.get('test_role', 'fixed holdout; no tuning'),
                                 'error_metrics_version': ERROR_METRICS_VERSION,
                                 'feature_table_sha256': config['feature_table_sha256'],
                                 'split_manifest_sha256': config['split_manifest_sha256']})
                mlflow.log_metrics({'training_seconds': duration, **all_metrics})
                mlflow.log_artifacts(str(provenance), 'metadata')
                mlflow.log_artifacts(str(output_dir / 'evaluations' / name), 'evaluations')
                mlflow.sklearn.log_model(model, name='model', serialization_format='skops',
                                        skops_trusted_types=trusted_types_for_model(name))
                (output_dir / f'{name}_run.json').write_text(json.dumps({'run_id': run.info.run_id,
                                                                          'tracking_uri': mlflow.get_tracking_uri()}, indent=2))
            print(f'{name} finished ({duration:.1f}s)', flush=True)
        except Exception:
            import traceback
            (output_dir / f'{name}_error.txt').write_text(traceback.format_exc())
            print(f'{name} failed; see error artifact', flush=True)
    results = pd.DataFrame(rows)
    if len(list(output_dir.glob('*_run.json'))) != len(models):
        raise RuntimeError('Some training or tracking stages failed; inspect saved errors')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split-dir', type=Path, required=True)
    p.add_argument('--feature-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    run_baseline(args.split_dir, args.feature_dir, args.output_dir)


if __name__ == '__main__':
    main()
