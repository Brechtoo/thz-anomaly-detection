"""Diagnostics must expose fixed-window limitations without changing data."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.surface_anchor_review import annotation_check, inspect_wave


def test_early_impulse_is_missed_and_window_displacement_is_explicit():
    x=np.arange(950.,1150.,.05)
    y=-4-np.exp(-((x-975)/.2)**2)+.1*np.exp(-((x-1005)/3)**2)
    result=inspect_wave(x,y)
    assert result['current_x']==pytest.approx(1005,abs=.05)
    assert result['broad_extremum_x']==pytest.approx(975,abs=.05)
    assert result['delta_x']==pytest.approx(30,abs=.05)
    assert result['roi_overlap_fraction']==pytest.approx(47/77,abs=.002)
    shifted=inspect_wave(x,2*y+10)
    assert shifted['current_x']==result['current_x']
    assert shifted['broad_extremum_x']==result['broad_extremum_x']


def test_incomplete_signal_is_rejected():
    with pytest.raises(ValueError,match='search regions'):
        inspect_wave(np.arange(0.,10.),np.ones(10))


def test_annotation_audit_counts_disagreement_without_relabeling(tmp_path):
    folder=tmp_path/'scan';folder.mkdir()
    annotations=tmp_path/'hohltestk1_tr_markArea_XYList2.txt'
    annotations.write_text('# x y label\n1 2 2\n2 2 3\n')
    before=annotations.read_bytes()
    catalog=pd.DataFrame({'file_path':[str(folder/'a.txt'),str(folder/'b.txt')],
                          'source_dataset':['messreihe_1']*2,'x':[1,2],'y':[2,2],'label':[2,1]})
    unchanged=catalog.copy(deep=True)
    report,_=annotation_check(catalog)
    assert report.label_mismatches.iloc[0]==1
    assert annotations.read_bytes()==before
    pd.testing.assert_frame_equal(catalog,unchanged)
