import numpy as np
import pandas as pd

from src.analysis.nested_release_review import budget_threshold, review, thresholds


def test_strict_target_and_specimen_fold_constraint_with_ties():
    inner = pd.DataFrame({'label': [1] * 100, 'fold': np.repeat([0, 1], [10, 90]),
                          'base': [3, 3] + [2] * 98, 'p3': [.9, .8] + [1.] * 98})
    gates = thresholds(inner, .5)
    assert gates['pooled'] == np.nextafter(.8, np.inf)
    assert gates['each_fold'] == np.nextafter(.9, np.inf)
    assert thresholds(inner, .99) == dict.fromkeys(gates, .99)
    inner.loc[1, 'p3'] = .9
    assert thresholds(inner, .5)['pooled'] == np.nextafter(.9, np.inf)


def test_budget_comparator_preserves_nonrelease_and_routes_ties_together():
    base = np.array([0, 1, 2, 3, 3, 3])
    scores = np.array([1., 1., 1., .9, .8, .8])
    for count, expected in [(0, 0), (1, 1), (2, 1), (3, 3)]:
        threshold = budget_threshold(base, scores, count)
        pred = review(base, scores >= threshold)
        assert (pred == 3).sum() == expected
        np.testing.assert_array_equal(pred[:3], base[:3])
        assert (pred == -1).sum() == 3 - expected


def test_consensus_veto_and_budget_use_inner_rows_only(monkeypatch):
    from src.analysis import nested_release_review as experiment

    inner = pd.DataFrame({
        'instance_id': range(4), 'specimen_id': ['a', 'a', 'b', 'b'], 'source_dataset': 's',
        'label': [1, 3, 2, 3], 'fold': [0, 0, 1, 1], 'base': [1, 3, 2, 3], 'p3': [.1, .8, .1, .9],
    })
    held = inner.assign(instance_id=range(4, 8), specimen_id='c', fold=2, base=[0, 2, 3, 3])

    def scores(model, combo, outer, train, assignments):
        val = inner.copy()
        out = train.copy()
        if model == 'mlp':
            val.loc[1, 'base'] = 1
            out.loc[3, 'base'] = 2
        return val, out, {'candidate': 'fixed', 'threshold': 0., 'gate': 'predict', 'calibration': 'ovr'}

    monkeypatch.setattr(experiment, 'load_scores', scores)
    first, second = [], []
    a = experiment.assess_fold('manual', 2, held, None, first)
    b = experiment.assess_fold('manual', 2, held.assign(label=3), None, second)
    pd.testing.assert_frame_equal(a.drop(columns='label'), b.drop(columns='label'))
    assert first == second
    assert a['all_six:pooled'].tolist() == [0, 2, 3, -1]
    assert a['svm_budget_all_six:pooled'].tolist() == [0, 2, -1, 3]
