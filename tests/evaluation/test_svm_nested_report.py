import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import svm_nested_report as report
from src.data.split_manifest import sha256_file


@pytest.fixture
def partial_run(tmp_path, monkeypatch):
    cohort = pd.DataFrame([
        {
            'instance_id': f'{fold}-{label}', 'specimen_id': f'body-{fold}',
            'source_dataset': 'source', 'label': label, 'fold': fold,
        }
        for fold in range(3) for label in range(4)
    ])
    monkeypatch.setattr(report, 'frozen_train', lambda source: pd.read_parquet(source / 'cohort.parquet'))
    source = tmp_path / 'source'
    checkpoint = source / 'fits/manual/rbf_svm/fold_0'
    checkpoint.mkdir(parents=True)
    cohort.to_parquet(source / 'cohort.parquet')
    fold = cohort.loc[cohort.fold.eq(0)].copy()
    path = checkpoint / 'predictions.parquet'
    fold.assign(prediction=[3, 1, 3, 3]).iloc[::-1].to_parquet(path)
    (checkpoint / 'metrics.json').write_text(json.dumps({'predictions_sha256': sha256_file(path)}))
    output = tmp_path / 'output'
    (output / 'outer_oof').mkdir(parents=True)
    outer = fold.assign(prediction_regular=[0, 3, 3, 1], prediction_hard=[0, 3, 3, 3],
                        prediction_adapted=[0, 2, 2, 3], prediction_review=[0, -1, -1, 3])
    for column in report.PROBA + report.REGULAR_PROBA:
        outer[column] = .25
    outer.iloc[::-1].to_parquet(output / 'outer_oof/manual.parquet')
    return output, source, path


def test_nested_report_pairs_same_outer_folds_and_separates_abstention(partial_run):
    output, source, _ = partial_run
    tables = report.write_report(output, source)
    decision = tables['decision_metrics.csv'].query('scope == "pooled"').set_index('method')
    assert decision.points.tolist() == [4] * 5
    assert decision.critical_errors.tolist() == [1, 2, 2, 0, 0]
    assert decision.false_alarms.tolist() == [0, 1, 0, 0, 0]
    assert not decision.complete.any()
    review = decision.loc['Explizite Prüfung']
    assert review.review_rate == 50 and review.true3_review_rate == 0
    assert np.isnan(review.macro_f1)
    deltas = tables['paired_deltas.csv'].query(
        'scope == "pooled" & reference == "Regulär" & variant == "Adaptiert"',
    )
    assert deltas.delta_critical_errors.iloc[0] == -2
    matrix = tables['confusion_matrices.csv'].query('scope == "pooled" & method == "Explizite Prüfung"')
    assert len(matrix) == 20 and matrix.loc[matrix.decision.eq(-1), 'count'].sum() == 2
    by_class = tables['outcomes_by_true_class.csv'].query('scope == "pooled"').set_index(['method', 'true_label'])
    assert by_class.loc[('Adaptiert', 2), ['precision', 'recall', 'f1']].tolist() == pytest.approx([.5, 1, 2 / 3])
    assert np.isnan(by_class.loc[('Adaptiert', 1), 'precision'])
    assert by_class.loc[('Adaptiert', 1), 'f1'] == 0
    assert by_class.loc['Explizite Prüfung', ['precision', 'recall', 'f1']].isna().all(axis=None)
    assert tables['probability_metrics.csv'].log_loss.iloc[0] == pytest.approx(np.log(4))
    summary = (output / 'report/status.md').read_text()
    assert 'Zwischenstand' in summary and 'explorativ' in summary
    for graphic in ['operating_points.png', 'specimen_deltas.png']:
        assert (output / 'report' / graphic).stat().st_size > 1000


def test_reference_checkpoint_hash_must_match(partial_run):
    output, source, path = partial_run
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        report.write_report(output, source)


def test_pending_report_renders_before_first_checkpoint(partial_run):
    output, source, _ = partial_run
    (output / 'outer_oof/manual.parquet').unlink()
    assert report.write_report(output, source) == {}
    assert 'Es liegen noch keine äußeren OOF-Checkpoints vor.' in (output / 'report/status.md').read_text()


def test_search_progress_counts_checkpoints_and_preserves_failure_paths(partial_run):
    output, source, _ = partial_run
    (output / 'outer_oof/manual.parquet').unlink()
    (output / 'plan.json').write_text(json.dumps({'features': {'manual': []}, 'candidates': [{'candidate': 'c0'}]}))
    inner = output / 'search/manual/outer_0/inner_1'
    (inner / 'c0').mkdir(parents=True)
    (inner / 'c0/fit.json').touch()
    (inner / 'failures').mkdir()
    (inner / 'failures/c0.json').touch()
    progress = report.write_report(output, source)['search_progress.csv'].set_index('outer_fold')
    assert progress.inner_fits_checkpointed.tolist() == [1, 0, 0]
    assert progress.inner_fits_expected.tolist() == [3, 3, 3]
    assert progress.full_candidates.tolist() == [0, 0, 0]
    assert progress.loc[0, 'failure_paths'] == 'search/manual/outer_0/inner_1/failures/c0.json'
    assert '1/9 innere Fits gespeichert' in (output / 'report/status.md').read_text()


def test_paired_specimen_percentiles_resample_within_fold():
    rows = pd.DataFrame({
        'features': 'manual', 'reference': 'reference', 'variant': 'variant', 'scope': 'specimen_id',
        'group': ['a', 'b', 'c'], 'fold': [0, 0, 1], 'complete': False,
        'delta_critical_errors': [0, -2, -4], 'critical_support': 10,
        'delta_critical_1_errors': 0, 'critical_1_support': 0,
        'delta_critical_2_errors': [0, -2, -4], 'critical_2_support': 10,
        'delta_false_alarms': 1, 'true3_support': 5,
    })
    intervals = report.paired_specimen_intervals(rows).set_index('metric')
    pd.testing.assert_frame_equal(intervals, report.paired_specimen_intervals(rows.iloc[::-1]).set_index('metric'))
    overall = intervals.loc['critical_error_rate']
    assert overall[['delta_pct_points', 'p025_pct_points', 'p975_pct_points']].tolist() == pytest.approx(
        [-20, -800 / 30, -400 / 30],
    )
    assert overall.valid_draws == 2000
    assert intervals.loc['critical_1_rate', 'valid_draws'] == 0
    assert np.isnan(intervals.loc['critical_1_rate', 'p025_pct_points'])


def test_missing_error_denominators_have_no_rate():
    values = report.decision_metrics(np.array([0, 3]), np.array([0, -1]), review=True)
    assert np.isnan(values['critical_error_rate'])
    assert values['true3_nonrelease_rate'] == 100
    assert values['review_rate'] == 50
    assert 'macro_f1' not in values
