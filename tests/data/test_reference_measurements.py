import pandas as pd
import pytest

from src.data.dataset import EXCLUDE_COLUMNS, prepare_dataset
from src.data.reference_measurements import (
    exclude_references,
    reference_mask,
    require_no_references,
)
from src.data.specimen_mapping import identify_specimens


def test_reference_ids_and_boundaries():
    frame = pd.DataFrame({'instance_id': ['messreihe_1_x-28_y102', 'messreihe_2_x-9_y120',
                                         'messreihe_3_x-8_y112', 'messreihe_1_x-28_y101',
                                         'messreihe_3_x-8_y113', 'other_x-28_y102']}, index=[8, 2, 1, 3, 6, 4])
    assert reference_mask(frame).tolist() == [True, True, True, False, False, False]
    assert exclude_references(frame).index.tolist() == [3, 6, 4]
    with pytest.raises(ValueError, match='3 reference measurements'):
        require_no_references(frame)
    require_no_references(exclude_references(frame))


def test_confirmed_reference_mapping_and_training_guard():
    frame = pd.DataFrame([[f'messreihe_1_x{x}_y{y}', 'messreihe_1', x, y]
                          for x in range(-28, -8) for y in range(102, 120)],
                         columns=['instance_id', 'source_dataset', 'x', 'y'])
    mapping, summary = identify_specimens(frame)
    assert mapping.measurement_role.eq('reference').all()
    assert not mapping.training_eligible.any()
    assert mapping.assignment_status.eq('confirmed_reference').all()
    assert not summary.review_required.any()
    for c in EXCLUDE_COLUMNS:
        if c not in frame:
            frame[c] = 0
    with pytest.raises(ValueError, match='reference measurements'):
        prepare_dataset(frame)
