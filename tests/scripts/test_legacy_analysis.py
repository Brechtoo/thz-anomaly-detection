from importlib import import_module

import matplotlib.pyplot as plt
import pandas as pd
import pytest


@pytest.mark.parametrize('module', [
    'src.generalization.transformer_external_test',
    'src.generalization.unsupervised_external_test',
    'src.analysis.boxplots',
    'src.analysis.basic_stats_and_correlation',
])
def test_import_does_not_read_measurements(module, monkeypatch):
    def unexpected_read(*args, **kwargs):
        pytest.fail('Import attempted to read measurements')
    monkeypatch.setattr(pd, 'read_csv', unexpected_read)
    assert callable(import_module(module).main)


def test_descriptive_report_handles_no_correlated_pairs(tmp_path, monkeypatch):
    from src.analysis import basic_stats_and_correlation as stats

    inputs = tmp_path / 'features.csv'
    pd.DataFrame({'label': [1, 3, 3], 'amplitude': [1., 2., 3.]}).to_csv(inputs, index=False)
    report = tmp_path / 'report.txt'
    monkeypatch.setattr(stats, 'INPUT_PATH', inputs)
    monkeypatch.setattr(stats, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(stats, 'OUTPUT_PATH', report)
    monkeypatch.setattr(plt, 'show', lambda: None)
    stats.main()
    assert 'amplitude' in report.read_text()
    assert (tmp_path / f'{stats.HEATMAP_NAME}.png').is_file()
