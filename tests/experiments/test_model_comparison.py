import json

import numpy as np
import pandas as pd
import pytest

from src.experiments import model_comparison as comparison
from src.experiments.model_registry import MODEL_ENABLED, build_models, create_model


def test_registry_and_train_only_preprocessing():
    assert not MODEL_ENABLED['rbf_svm']
    train = pd.DataFrame({'constant': [5.] * 8, 'signal': list(range(8))})
    model = create_model('logistic_regression')
    model.fit(train, [0, 1, 2, 3] * 2)
    np.testing.assert_array_equal(model.named_steps['constants'].get_support(), [False, True])
    assert model.named_steps['scale'].mean_[0] == 3.5
    model.predict(pd.DataFrame({'constant': [999.], 'signal': [1000.]}))
    assert model.named_steps['scale'].mean_[0] == 3.5
    assert 'scale' not in create_model('hgb').named_steps
    with pytest.raises(ValueError, match='Unknown'):
        build_models({'typo': True})


def test_input_hash_rejected(tmp_path):
    path = tmp_path / 'features.csv'
    path.write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        comparison.load_comparison_data({'feature_table': str(path), 'feature_table_sha256': 'wrong'})


@pytest.fixture
def comparison_setup(tmp_path, monkeypatch):
    import mlflow
    root = tmp_path
    # Auch MLflows standardmäßig relatives Artefaktverzeichnis bleibt im Testordner.
    monkeypatch.chdir(root)
    (root / 'src').mkdir()
    splits = root / 'splits'
    splits.mkdir()
    (splits / 'manifest.json').write_text('{}')
    features = root / 'features.csv'
    features.with_suffix('.json').write_text('{}')
    config = root / 'config.json'
    config.write_text(json.dumps({'seed': 42, 'split_dir': str(splits),
                                 'feature_table': str(features), 'feature_table_sha256': 'fixture'}))
    X = pd.DataFrame({'a': np.arange(16.), 'constant': np.ones(16)})
    y = pd.Series([0, 1, 2, 3] * 4)
    monkeypatch.setattr(comparison, 'load_comparison_data', lambda c: (X, X.copy(), y, y.copy()))
    monkeypatch.setattr(comparison, 'PROJECT_ROOT', root)
    monkeypatch.setattr(comparison, 'EXPERIMENTS_ROOT', root / 'experiments')
    monkeypatch.setattr(comparison, 'capture_git_state', lambda p: {'git_commit': 'fixture', 'git_dirty': False})

    previous_uri = mlflow.get_tracking_uri()
    yield config, root
    mlflow.set_tracking_uri(previous_uri)


@pytest.mark.parametrize("failure_stage", ["fit", "predict", "metrics", "local_save", "mlflow"])
def test_comparison_preserves_progress_and_continues(comparison_setup, monkeypatch, failure_stage):
    config, root = comparison_setup
    first = create_model('logistic_regression')
    second = create_model('logistic_regression')

    def fail(*args, **kwargs):
        raise RuntimeError('deliberate failure')

    # Nur das erste Modell scheitert, damit der nächste Lauf geprüft wird.
    if failure_stage in ('fit', 'predict'):
        monkeypatch.setattr(first, failure_stage, fail)
    else:
        target, attr = {
            'metrics': (comparison, 'evaluate_predictions'),
            'local_save': (comparison, 'save_model'),
            'mlflow': (comparison.mlflow.sklearn, 'log_model'),
        }[failure_stage]
        original = getattr(target, attr)
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                fail()
            return original(*args, **kwargs)

        monkeypatch.setattr(target, attr, fail_once)
    monkeypatch.setattr(comparison, 'build_models', lambda *a: {
        'logistic_regression': first, 'rbf_svm': second})
    result = comparison.run_model_comparison(config)
    directory, = (root / 'experiments').iterdir()
    saved = pd.read_csv(directory / 'validation_results.csv')
    individual = pd.read_csv(directory / 'logistic_regression_results.csv').iloc[0]
    failed = result.iloc[0]
    phase = {'fit': 'training', 'predict': 'evaluation', 'metrics': 'evaluation'}.get(failure_stage, failure_stage)
    assert failed[f'{phase}_status'] == 'FAILED'
    assert failed[f'{phase}_error_type'] == 'RuntimeError'
    assert result.iloc[1]['status'] == 'FINISHED'
    assert saved.status.tolist() == result.status.tolist()
    assert 'deliberate failure' in (directory / f'logistic_regression_{phase}_error.txt').read_text()
    if failure_stage != 'fit':
        assert failed.training_status == 'FINISHED'
        assert failed.training_seconds >= 0
    if failure_stage in ('local_save', 'mlflow'):
        assert failed.evaluation_status == 'FINISHED'
        assert failed.status == 'PARTIAL'
        for metric in ('accuracy', 'critical_errors', 'false_alarms', 'prediction_seconds'):
            assert individual[metric] == pytest.approx(failed[metric])
            assert saved.iloc[0][metric] == pytest.approx(failed[metric])
    if failure_stage == 'mlflow':
        assert failed.local_save_status == 'FINISHED'
        assert (directory / 'models/logistic_regression.joblib').exists()
        run = comparison.mlflow.MlflowClient().get_run(failed.run_id)
        assert run.info.status == 'FAILED'
        assert run.data.tags['evaluation_status'] == 'FINISHED'
        assert run.data.tags['mlflow_status'] == 'FAILED'

    # Die hochgeladene CSV muss den endgültigen Status und die Run-ID enthalten.
    success = result.iloc[1]
    artifact = comparison.mlflow.MlflowClient().download_artifacts(
        success.run_id, 'rbf_svm_results.csv', str(root))
    uploaded = pd.read_csv(artifact).iloc[0]
    assert uploaded['status'] == uploaded.mlflow_status == 'FINISHED'
    assert uploaded.run_id == success.run_id


@pytest.mark.parametrize('name', list(MODEL_ENABLED))
def test_pipeline_mlflow_roundtrip(comparison_setup, name):
    import mlflow
    from threadpoolctl import threadpool_limits

    from src.experiments.model_serialization import trusted_types_for_model

    # Kleine synthetische Daten statt des echten tsfresh-Trainings.
    if name in ('xgboost', 'lightgbm'):
        pytest.importorskip(name)
    _, root = comparison_setup
    model = create_model(name)
    params = {
        'random_forest': {'model__n_estimators': 3},
        'extra_trees': {'model__n_estimators': 3},
        'hgb': {'model__max_iter': 3, 'model__min_samples_leaf': 2},
        'xgboost': {'model__n_estimators': 3},
        'lightgbm': {'model__n_estimators': 3, 'model__min_child_samples': 2},
        'mlp': {'model__hidden_layer_sizes': (4,), 'model__max_iter': 3},
    }
    model.set_params(**params.get(name, {}))
    X = pd.DataFrame({'signal': np.arange(32.), 'constant': np.ones(32)})
    y = np.array([0, 1, 2, 3] * 8)
    mlflow.set_tracking_uri('sqlite:///' + str(root / 'roundtrip.db'))
    mlflow.set_experiment('roundtrip')
    with threadpool_limits(limits=1):
        model.fit(X, y)
        expected = model.predict(X)
        with mlflow.start_run():
            info = mlflow.sklearn.log_model(
                model, name='pipeline', serialization_format='skops',
                skops_trusted_types=trusted_types_for_model(name),
                pip_requirements=[],  # Keine Paket-Inferenz für diesen lokalen Funktionstest.
            )
        loaded = mlflow.sklearn.load_model(info.model_uri)
        np.testing.assert_array_equal(loaded.predict(X), expected)
        np.testing.assert_array_equal(
            loaded.named_steps['constants'].get_support(), [True, False])


def test_type_allowlist_rejects_unknown_names():
    from src.experiments.model_serialization import trusted_types_for_model
    with pytest.raises(KeyError):
        trusted_types_for_model('unknown')
    types = trusted_types_for_model('random_forest')
    assert types == ['sklearn.tree._tree.Tree']
    types.append('unreviewed.Type')
    assert trusted_types_for_model('random_forest') == ['sklearn.tree._tree.Tree']


def unreviewed_transform(X):
    """Absichtlich nicht freigegebener Typ für den Sicherheitstest."""
    return X


def test_serialization_does_not_trust_unreviewed_types(tmp_path):
    import mlflow.sklearn
    from mlflow.exceptions import MlflowException
    from sklearn.preprocessing import FunctionTransformer

    from src.experiments.model_serialization import trusted_types_for_model

    # Auch ein ansonsten unterstütztes Modell darf keine zusätzlichen,
    # ungeprüften Python-Funktionen automatisch freigeben.
    model = create_model('logistic_regression')
    model.steps.insert(0, ('unreviewed', FunctionTransformer(unreviewed_transform)))
    X = pd.DataFrame({'signal': [0., 1., 2., 3.]})
    model.fit(X, [0, 1, 0, 1])
    with pytest.raises(MlflowException, match='untrusted types'):
        mlflow.sklearn.save_model(
            model, tmp_path / 'untrusted', serialization_format='skops',
            skops_trusted_types=trusted_types_for_model('logistic_regression'),
            pip_requirements=[],
        )
