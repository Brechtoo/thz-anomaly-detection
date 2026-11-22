import json

import pandas as pd

from src.analysis.feature_grid_report import write_report


def cohort():
    return pd.DataFrame([
        {
            'instance_id': f'{fold}-{label}', 'specimen_id': f'body-{fold}',
            'source_dataset': 'source', 'label': label, 'fold': fold,
        }
        for fold in range(3) for label in range(4)
    ])


def write_checkpoint(path, data, predictions, *, reused=False):
    path.mkdir(parents=True)
    frame = data.assign(prediction=predictions)
    # Equal scores are intentional. Reporting uses the saved decision, not argmax.
    for label in range(4):
        frame[f'p_{label}'] = .25
    frame.iloc[::-1].to_parquet(path / 'predictions.parquet')
    (path / 'metrics.json').write_text(json.dumps({
        'training_seconds': 1.5, 'prediction_seconds': .25, 'run_id': 'run', 'reused': reused,
    }))


def test_ranking_and_paired_comparison_follow_safety_priority(tmp_path):
    data = cohort()
    combinations = {
        'manual': ['manual'],
        'manual+physical_v1': ['manual', 'physical_v1'],
        'manual+physical_v2': ['manual', 'physical_v2'],
    }
    for name, predictions in zip(combinations, ([0, 3, 2, 3], [0, 1, 3, 2], [0, 1, 2, 2]), strict=True):
        for fold in range(3):
            write_checkpoint(tmp_path / 'fits' / name / 'model' / f'fold_{fold}',
                             data.loc[data.fold.eq(fold)], predictions, reused=name == 'manual')
    pooled, status = write_report(tmp_path, combinations, ['model'], data)
    assert pooled.combination.tolist() == ['manual+physical_v2', 'manual', 'manual+physical_v1']
    assert pooled.critical_1_errors.tolist() == [0, 3, 0]
    assert pooled.critical_2_errors.tolist() == [0, 0, 3]
    assert pooled.false_alarms.tolist() == [3, 0, 3]
    assert pooled.reused_folds.tolist() == [0, 3, 0]
    assert pooled.training_seconds.tolist() == [4.5] * 3
    assert set(status.status) == {'finished'}
    deltas = pd.read_csv(tmp_path / 'report/paired_deltas.csv').set_index('change')
    assert deltas.loc['physical_v1_to_v2', 'delta_critical_errors'] == -3
    assert deltas.loc['add_physical_v2', 'delta_false_alarms'] == 3
    assert deltas.loc['physical_v1_to_v2', 'outcome'] == 'win'
    assert deltas.loc['add_physical_v1', 'outcome'] == 'loss'
    summary = (tmp_path / 'report/status.md').read_text()
    assert 'beste abgeschlossene Einzelset ist **manual / model**' in summary
    assert 'beste abgeschlossene Kombination ist **manual+physical_v2 / model**' in summary
    assert '| Physikalisch V1 durch V2 ersetzen | 1 | 1 | 0 | 0 |' in summary
    matrix = pd.read_csv(tmp_path / 'report/confusion_matrices.csv')
    assert len(matrix) == 48
    assert matrix.groupby('combination')['count'].sum().tolist() == [12] * 3


def test_unfinished_corrupt_and_failed_folds_stay_out_of_ranking(tmp_path):
    data = cohort()
    fold = data.loc[data.fold.eq(0)].copy()
    fold.loc[fold.label.eq(1), 'specimen_id'] = 'wrong-body'
    write_checkpoint(tmp_path / 'fits/manual/model/fold_0', fold, [0, 1, 2, 3])
    failures = tmp_path / 'failures/manual/model'
    failures.mkdir(parents=True)
    (failures / 'fold_1.json').write_text('{"error":"fit failed"}')
    pooled, status = write_report(tmp_path, {'manual': ['manual']}, ['model'], data)
    assert pooled.empty
    assert status.status.tolist() == ['invalid', 'failed', 'pending']
    assert len(pd.read_csv(tmp_path / 'report/fold_status.csv')) == 3


def test_probability_columns_must_sum_to_one(tmp_path):
    data = cohort()
    path = tmp_path / 'fits/manual/model/fold_0'
    write_checkpoint(path, data.loc[data.fold.eq(0)], [0, 1, 2, 3])
    predictions = path / 'predictions.parquet'
    frame = pd.read_parquet(predictions)
    frame.assign(p_1=.5).to_parquet(predictions)
    pooled, status = write_report(tmp_path, {'manual': ['manual']}, ['model'], data)
    assert pooled.empty
    assert status.status.iloc[0] == 'invalid'
    assert 'class probabilities' in status.detail.iloc[0]


def test_holdouts_document_recorded_candidates_without_changing_oof_ranking(tmp_path):
    data = cohort()
    combinations = {'manual': ['manual'], 'tsfel': ['tsfel']}
    for combo, predictions in zip(combinations, ([0, 1, 2, 3], [0, 3, 2, 3]), strict=True):
        for fold in range(3):
            write_checkpoint(tmp_path / 'fits' / combo / 'model' / f'fold_{fold}',
                             data.loc[data.fold.eq(fold)], predictions)
    (tmp_path / 'shortlist.json').write_text(json.dumps({
        'pairs': [['manual', 'model'], ['tsfel', 'model']], 'source': 'pooled OOF only',
    }))
    (tmp_path / 'test_selection.json').write_text(json.dumps({
        'pair': ['tsfel', 'model'], 'source': 'validation shortlist, no test scores',
    }))
    test = tmp_path / 'holdout/tsfel/model/test'
    test.mkdir(parents=True)
    (test / 'metrics.json').write_text(json.dumps({
        'critical_errors': 0, 'critical_error_rate': 0, 'false_alarms': 0, 'false_alarm_rate': 0,
        'macro_f1': 1, 'balanced_accuracy': 1,
    }))
    failures = tmp_path / 'failures/manual/model'
    failures.mkdir(parents=True)
    (failures / 'validation.json').write_text('{"error":"holdout fit failed"}')

    pooled, _ = write_report(tmp_path, combinations, ['model'], data)
    assert pooled.combination.tolist() == ['manual', 'tsfel']
    holdouts = pd.read_csv(tmp_path / 'report/holdout_metrics.csv')
    assert holdouts.status.tolist() == ['failed', 'pending', 'finished']
    summary = (tmp_path / 'report/status.md').read_text()
    assert 'Die OOF-Shortlist enthält 2 Kandidaten.' in summary
    assert 'Vor der Testausführung wurde **tsfel / model** ausgewählt.' in summary
    assert '| Test | tsfel | model | finished | 0 / 0.00 | 0.00 | 1.0000 | 1.0000 |' in summary
