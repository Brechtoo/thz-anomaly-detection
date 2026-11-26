import numpy as np
import pandas as pd
import pytest

from src.analysis.tabular_nested_report import write_report
from src.analysis.tabular_nested_robustness import specimen_sensitivity, write_analysis
from src.analysis.tabular_nested_summary import write_summary


def test_specimen_concentration_and_paired_intervals():
    svm = np.tile(np.eye(4, dtype=int) * 4, (4, 1, 1))
    model = svm.copy()
    model[0, 1, 1], model[0, 1, 3] = 0, 4
    folds = np.array([0, 0, 1, 1])
    intervals, bodies = specimen_sensitivity(model, svm, folds, draws=200)
    critical = intervals.set_index('metric').loc['critical_error_rate']
    assert critical.delta == 12.5
    assert critical.delta_lower_95 == 0 and critical.delta_upper_95 == 25
    np.testing.assert_allclose(bodies.delta_without_critical_error_rate, [0, 100 / 6, 100 / 6, 100 / 6])
    assert bodies.delta_errors_on_specimen.tolist() == [4, 0, 0, 0]
    assert intervals.set_index('metric').loc['false_alarm_rate', 'delta_upper_95'] == 0
    with pytest.raises(ValueError, match='true class counts'):
        specimen_sensitivity(model + 1, svm, folds)


def test_checkpoint_report_documents_only_fully_assessed_feature_sets(tmp_path):
    baseline, evaluated = {}, {}
    for fold in range(3):
        data = pd.DataFrame({
            'instance_id': [f'{fold}-{label}' for label in range(4)], 'specimen_id': f'pk_{fold}',
            'source_dataset': 'train', 'fold': fold, 'label': range(4),
            'prediction_adapted': range(4), 'prediction_regular': range(4),
        })
        baseline['manual', fold] = data
        evaluated['xgboost', 'manual', fold] = data.assign(prediction_hard=data.label)
    write_report(tmp_path, evaluated, baseline, completed=False)
    write_analysis(tmp_path)
    write_summary(tmp_path)
    text = (tmp_path / 'robustness/vergleich.md').read_text()
    assert 'Zwischenstand mit 1 von 15' in text
    assert '12 Messungen der 3 Trainingsprüfkörper' in text
    assert '| XGBoost | M | 0/3/0 |' in text
    assert '## Folgerung' not in text
