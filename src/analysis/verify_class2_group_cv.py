"""Reload-only audit of new ablation models, OOF predictions and protected files."""
import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.data.split_manifest import sha256_file
from src.experiments.class2_group_cv import (
    ARMS,
    load_inputs,
    original_probabilities,
    training_indices,
)
from src.experiments.model_registry import build_models


def verify(output):
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text())
    source = Path(plan['source'])
    data, X, _, _ = load_inputs(source)
    protected = json.loads((output / 'protected_inputs.json').read_text())
    print(f'Checking {len(protected)} protected files', flush=True)
    changed = [p for p, h in protected.items() if not Path(p).is_file() or sha256_file(p) != h]
    if changed:
        raise ValueError(f'Existing files changed: {changed[:10]}')
    runs = pd.read_csv(output / 'fold_runs.csv')
    if len(runs) != 42 or runs.run_id.nunique() != 42:
        raise ValueError('Expected 42 unique fits')
    count = 0
    for arm, classes in ARMS.items():
        for name, factory in build_models(seed=42).items():
            frame = pd.read_csv(output / 'predictions' / f'{arm}__{name}.csv')
            pd.testing.assert_frame_equal(frame[['instance_id', 'label', 'specimen_id', 'source_dataset', 'fold']],
                                          data[['instance_id', 'label', 'specimen_id', 'source_dataset', 'fold']])
            for fold in range(3):
                path = output / 'models' / f'{arm}__{name}__fold{fold}.joblib'
                model = joblib.load(path)
                detail = json.loads(path.with_suffix('.json').read_text())
                assert detail['parameters'] == {k: str(v) for k, v in factory.get_params(deep=True).items()}
                assert detail['original_classes'] == classes.tolist()
                tr = training_indices(data, fold, classes)
                va = np.flatnonzero(data.fold.to_numpy() == fold)
                np.testing.assert_allclose(model.named_steps['imputer'].statistics_, np.nanmedian(X[tr], axis=0), rtol=0, atol=0)
                transformed = model.named_steps['imputer'].transform(X[tr])
                variance = np.var(transformed, axis=0)
                np.testing.assert_array_equal(model.named_steps['constants'].get_support(), variance > 0)
                transformed = model.named_steps['constants'].transform(transformed)
                if 'scale' in model.named_steps:
                    np.testing.assert_allclose(model.named_steps['scale'].mean_, transformed.mean(axis=0), rtol=1e-13, atol=1e-13)
                with threadpool_limits(limits=1):
                    p = original_probabilities(model, X[va], classes)
                    pred = classes[np.asarray(model.predict(X[va]), dtype=int)]
                np.testing.assert_array_equal(pred, frame.prediction.iloc[va])
                np.testing.assert_allclose(p, frame.iloc[va][[f'p_{i}' for i in range(4)]], rtol=1e-12, atol=1e-14)
                count += 1
            if arm == 'four_class':
                old = pd.read_csv(source / 'cv/predictions' / f'early_legacy_windows_{name}.csv')
                np.testing.assert_array_equal(frame.instance_id, old.instance_id)
                np.testing.assert_array_equal(frame.prediction, old.prediction)
    mixture_count = 0
    for fold in range(3):
        path = output / f'mixture_fold{fold}.joblib'
        if not path.exists():
            continue
        fitted = joblib.load(path)
        training = data[data.fold != fold]
        assert set(fitted['transform_train_ids']) == set(training.instance_id[training.label != 2])
        assert set(fitted['mixture_train_ids']) == set(training.instance_id[training.label == 2])
        mixture_count += 1
    with sqlite3.connect(f'file:{output.resolve() / "mlflow.db"}?mode=ro', uri=True) as connection:
        statuses = connection.execute('SELECT status, COUNT(*) FROM runs GROUP BY status').fetchall()
    assert statuses == [('FINISHED', 42)], statuses
    result = dict(protected_files_unchanged=len(protected), models_verified=count,
                  imputer_variance_scaler_verified=count, probabilities_predictions_reproduced=count,
                  original_four_class_oof_exact_matches=7, mixture_fold_isolation_verified=mixture_count,
                  mlflow_finished_runs=42, original_holdouts_used=False)
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    verify(parser.parse_args().output)
