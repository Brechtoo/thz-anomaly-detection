import numpy as np
import pandas as pd
import pytest

from src.features.specimen_fusion import fuse_features


def test_fusion_filters_only_on_training_and_keeps_manual_representative():
    manual = pd.DataFrame({'pulse': [-0.0, 2.0, np.nan, 1000.0]}, index=[10, 20, 30, 40])
    tsfresh = pd.DataFrame({
        'copy': [0.0, 2.0, 1.0, -900.0],
        'varying_only_in_holdout': [4.0, 4.0, 4.0, 999.0],
        'almost_copy': [0.0, 2.0, 1.000001, -900.0],
        'new': [1.0, 8.0, 4.0, 1.0],
    }, index=manual.index)
    training = manual.index[:3]

    transformed, details = fuse_features(manual, tsfresh, training)

    assert transformed.columns.tolist() == ['manual__pulse', 'tsfresh__almost_copy', 'tsfresh__new']
    assert transformed['manual__pulse'].tolist() == [0.0, 2.0, 1.0, 1000.0]
    assert details['manual_medians'] == {'manual__pulse': 1.0}
    assert details['removed_duplicates'] == {'tsfresh__copy': 'manual__pulse'}
    assert details['removed_constants'] == ['tsfresh__varying_only_in_holdout']


def test_fusion_rejects_misaligned_inputs():
    manual = pd.DataFrame({'pulse': [0.0, 2.0]})
    with pytest.raises(ValueError, match='aligned'):
        fuse_features(manual, manual.iloc[::-1], manual.index)
