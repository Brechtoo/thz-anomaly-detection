import numpy as np
import pandas as pd

from src.analysis.nested_release_pareto import (
    COUNTS,
    KEY,
    Counts,
    add_rates,
    choose,
    pareto_mask,
    risk_cutoffs,
)


def test_strict_cutoffs_keep_ties_and_apply_fold_and_class_constraints():
    labels = np.repeat([1, 2], [25, 175])
    folds = np.repeat([0, 1], [50, 150])
    scores = np.full(200, -1.)
    scores[[0, 1, 100]] = [.9, .9, .8]
    cuts = risk_cutoffs(scores, labels, folds, 100)
    assert cuts['pooled'] == np.nextafter(.9, np.inf)
    scores[1] = -1.
    cuts = risk_cutoffs(scores, labels, folds, 100)
    assert cuts['pooled'] == np.nextafter(.8, np.inf)
    assert cuts['class'] == cuts['fold'] == np.nextafter(.9, np.inf)
    assert risk_cutoffs(scores, labels, folds, 50)['pooled'] == cuts['fold']


def test_dominance_keeps_ties_and_tradeoffs_and_selection_is_strict():
    frame = pd.DataFrame([
        [1000, 100, 100, 600, 1, 0, 10, 50, 500],
        [1000, 100, 100, 600, 0, 1, 10, 50, 500],
        [1000, 100, 100, 600, 0, 1, 20, 50, 500],
        [1000, 100, 100, 600, 0, 2, 5, 25, 600],
        [1000, 100, 100, 600, 0, 0, 2, 20, 0],
    ], columns=COUNTS)
    for col in KEY:
        frame[col] = range(len(frame))
    assert pareto_mask(frame).tolist() == [True, True, False, True, True]
    assert choose(frame, 100, 'release').name == 0
    assert choose(frame, 200, 'review').name == 3


def test_reviews_use_lead_and_conditional_rates_use_class_support():
    y = np.array([0, 1, 1, 2, 3, 3])
    base = np.array([0, 3, 3, 3, 1, 3])
    counter = Counts(y, np.array([0, 0, 0, 1, 1, 1]), {'lead': base})
    mask = np.array([0, 0, 1, 0, 0, 1], dtype=bool)
    result = add_rates(pd.DataFrame(counter.rows(counter.count(mask), 'lead'), columns=COUNTS))
    assert result.class1_rate.iloc[0] == 50
    assert np.isnan(result.class2_rate.iloc[0])
    assert result.class2_rate.iloc[1] == 0
    assert result.alarm_rate.iloc[1] == 50
    assert result.reviews.tolist() == [1, 1]
