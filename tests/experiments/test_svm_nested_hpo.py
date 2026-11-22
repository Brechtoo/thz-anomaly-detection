import mlflow
import numpy as np
import pandas as pd

from src.experiments import svm_nested_hpo as hpo


def cohort():
    rows = [
        (f'{fold}-{specimen}-{label}-{rep}', f'{fold}-{specimen}', 'synthetic', label, fold)
        for fold in range(3)
        for specimen in range(6)
        for label in range(4)
        for rep in range(4)
    ]
    return pd.DataFrame(rows, columns=hpo.KEYS)


def test_inner_assignments_ignore_outer_labels():
    train = cohort()
    baseline = hpo.inner_assignments(train)
    modified = train.copy()
    modified.loc[modified.fold.eq(1), 'label'] = (modified.loc[modified.fold.eq(1), 'label'] + 1) % 4
    new = hpo.inner_assignments(modified)
    pd.testing.assert_frame_equal(baseline.loc[baseline.outer.eq(1)], new.loc[new.outer.eq(1)])
    assert baseline.groupby(['outer', 'specimen_id']).fold.nunique().eq(1).all()
    assert baseline.original_fold.ne(baseline.outer).all()
    assert len(hpo.candidate_grid()) == 32


def test_inner_refit_and_resume(tmp_path, monkeypatch):
    train = cohort()
    assignments = hpo.inner_assignments(train).query('outer == 0')
    combo = 'manual'
    candidates = hpo.candidate_grid()[:2]
    run_dir = tmp_path / 'output'
    work = run_dir / 'search' / combo / 'outer_0'
    work.mkdir(parents=True)
    rng = np.random.default_rng(172)
    values = np.eye(4)[train.label] * 12 + rng.normal(size=(len(train), 4)) * .01
    seen = []

    class Features:
        def prepare(self, fit, assessment, names):
            assert names == ('manual',)
            assert set(fit.specimen_id).isdisjoint(assessment.specimen_id)
            seen.append((set(fit.fold), set(assessment.specimen_id)))
            return values[fit.index], values[assessment.index], {'names': list(names)}

    uri = 'sqlite:///' + str(tmp_path / 'mlflow.db')
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri)
    monkeypatch.setattr(mlflow.tracking._tracking_service.utils, '_tracking_uri', uri)
    monkeypatch.setattr(mlflow.tracking.fluent, '_active_experiment_id', None)
    client = mlflow.MlflowClient(tracking_uri=uri)
    client.create_experiment('synthetic_hpo', artifact_location=(tmp_path / 'artifacts').as_uri())
    experiment = mlflow.set_experiment('synthetic_hpo')
    with mlflow.start_run() as parent:
        hpo.fit_inner(run_dir, work, train, assignments, Features(), combo, 0, candidates, 1)
        hpo.fit_inner(run_dir, work, train, assignments, Features(), combo, 0, candidates, 1)
        assert len(seen) == 3
        assert all(0 not in fit_folds and all(not body.startswith('0-') for body in eval_bodies)
                   for fit_folds, eval_bodies in seen)
        picks, oof = hpo.choose_inner(work, assignments, train, combo, 0, candidates, client, parent)
        assert all(len(frame) == train.fold.ne(0).sum() for frame in oof.values())
        fit, held = train.loc[train.fold.ne(0)], train.loc[train.fold.eq(0)]
        hpo.finish_outer(run_dir, work, fit, held, Features(), combo, 0, candidates, picks, oof, 1, client, parent)
        hpo.finish_outer(run_dir, work, fit, held, Features(), combo, 0, candidates, {}, {}, 1, client, parent)
        assert len(seen) == 4
        hpo.update_aggregate(run_dir)

    result = pd.read_parquet(run_dir / 'outer_oof' / 'manual.parquet')
    pd.testing.assert_frame_equal(result[hpo.KEYS], held[hpo.KEYS].reset_index(drop=True))
    for prefix in ('', 'regular_'):
        np.testing.assert_allclose(result[[f'{prefix}p_{i}' for i in range(4)]].sum(axis=1), 1)
    assert result.prediction_adapted.isin(range(4)).all()
    assert result.prediction_review.isin(range(-1, 4)).all()
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4 and all(run.info.status == 'FINISHED' for run in runs)
