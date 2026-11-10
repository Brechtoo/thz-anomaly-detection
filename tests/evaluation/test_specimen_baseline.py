import pandas as pd

from src.experiments.specimen_baseline import evaluate_holdout


def test_id_linked_group_metrics_and_fixed_confusion_matrix(tmp_path):
    data = pd.DataFrame({'instance_id': list('abcd'), 'label': [0, 1, 2, 3],
                         'specimen_id': ['a', 'a', 'b', 'b'], 'source_dataset': ['s'] * 4})
    metrics = evaluate_holdout(data, [0, 3, 3, 1], tmp_path / 'evaluation')
    assert metrics['critical_errors'] == 2
    assert metrics['critical_error_rate'] == 100
    assert metrics['false_alarms'] == 1
    assert metrics['false_alarm_rate'] == 100
    assert metrics['recall_0'] == 1
    matrix = pd.read_csv(tmp_path / 'evaluation/confusion_matrix.csv', index_col=0)
    assert matrix.shape == (4, 4) and matrix.to_numpy().sum() == 4
    saved = pd.read_csv(tmp_path / 'evaluation/predictions.csv')
    assert saved.instance_id.tolist() == list('abcd')
    groups = pd.read_csv(tmp_path / 'evaluation/by_specimen_id.csv')
    assert groups.points.sum() == 4
