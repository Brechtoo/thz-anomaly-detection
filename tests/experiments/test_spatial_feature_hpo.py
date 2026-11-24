import numpy as np
import pandas as pd

from src.evaluation.svm_nested_selection import raw_metrics
from src.experiments.spatial_feature_hpo import POINT, screen_radius, search_choices


def metrics(errors, *, collapse=False):
    labels = np.tile(np.repeat(np.arange(4), 50), 3)
    predictions = np.full(len(labels), 2) if collapse else labels.copy()
    predictions[np.flatnonzero(labels == 1)[:errors]] = 3
    return raw_metrics(pd.DataFrame({
        'label': labels, 'prediction': predictions, 'fold': np.repeat(range(3), 200),
        'specimen_id': np.repeat(['a', 'b', 'c'], 200),
    }))


def test_radius_screen_keeps_a_spatial_arm_if_point_is_better():
    choice = screen_radius({POINT: metrics(0), 'tight': metrics(8), 'wide': metrics(0, collapse=True)})

    assert choice['arm'] == 'tight'
    assert choice['feasible']


def test_nested_selection_preserves_frozen_fit_on_tie_and_allows_point_fallback():
    rows = pd.DataFrame([
        {'arm': 'wide', 'candidate': 'frozen', **metrics(4)},
        {'arm': 'wide', 'candidate': 'equivalent', **metrics(4)},
        {'arm': 'wide', 'candidate': 'poor-quality', **metrics(0, collapse=True)},
    ])
    best, selected = search_choices(rows, {'arm': POINT, 'candidate': 'frozen', **metrics(0)})

    assert best['candidate'] == 'frozen'
    assert selected['arm'] == POINT

    best, selected = search_choices(rows, {'arm': POINT, 'candidate': 'frozen', **metrics(12)})

    assert selected == best
