import numpy as np
import pandas as pd
import pytest

from src.analysis.specimen_ood_review import DETECTORS, assess, budget_cutoff, novelty


def test_budget_keeps_ties_and_ignores_ineligible_rows():
    score = np.array([.4, .5, .5, .7, 1.])
    release = np.array([True, True, True, True, False])
    for bp, expected in [(0, 0), (2000, 1), (4000, 1), (8000, 4)]:
        flags = release & (score > budget_cutoff(score, release, bp))
        assert flags.sum() == expected
        assert not flags[-1]


def test_detector_choice_ignores_outer_labels():
    frame = pd.DataFrame(dict(specimen_id='a', label=[1, 3, 3, 2], release=[True, True, True, False],
                              review=[False, False, False, True], alarm=False))
    for name in DETECTORS:
        frame[name] = [1., .3, .1, 1.] if name == 'isolation' else [.1, .3, 1., 1.]
    choices = []
    groups = assess(frame, frame.assign(label=3), 2500, 200, 0, choices)
    selected = [r for r in choices if r['selected']]
    assert [r['detector'] for r in selected] == ['isolation']
    pooled = pd.concat(groups).query("detector == 'selected'").iloc[0]
    assert pooled.extra == 1 and pooled.caught1 == 0 and pooled.reviews == 2


def test_novelty_excludes_held_body_and_reproduces_direct_distance():
    fit = pd.DataFrame({'label': np.tile([1, 3], 128), 'specimen_id': 'a'}, index=range(256))
    held = pd.DataFrame({'instance_id': ['id'], 'specimen_id': ['b']}, index=[256])
    x = pd.DataFrame({'a': np.arange(257, dtype=float), 'b': [1.] * 256 + [2.]})
    result = novelty(fit, held, x)
    direct = np.hypot((256 - np.arange(256)) / np.arange(256).std(), 1)
    assert result.distance.iloc[0] == pytest.approx(np.sort(direct)[14])
    with pytest.raises(ValueError, match='overlap'):
        novelty(fit, held.assign(specimen_id='a'), x)


def test_annotation_check_uses_target_coordinates_and_rejects_mismatch(tmp_path):
    from src.analysis.specimen_ood_diagnosis import check_annotations

    path = tmp_path / 'training_coordinates.txt'
    path.write_text('# x y label\n65 15 1\n65 14 3\n')
    check_annotations(path, {(65, 15): 1})
    with pytest.raises(ValueError, match='inconsistent'):
        check_annotations(path, {(65, 15): 3})
