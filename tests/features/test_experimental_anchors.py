"""Parity, fold isolation and transparent failure handling for anchor experiments."""
import numpy as np
import pandas as pd

from src.experiments.anchor_group_cv import fold_pipeline, make_folds
from src.experiments.model_registry import create_model
from src.features.experimental_anchor_calculator import extension_at_anchor
from src.features.experimental_anchors import (
    adapted_start,
    early_index,
    extract_variant,
    fit_template,
    template_index,
)
from src.features.extract_extension_block1_features import (
    extract_extension_block1_features,
)


def signal(path,shift=0):
    x=np.arange(950,1180,.05)
    y=-4-np.exp(-((x-975-shift)/.25)**2)+.08*np.exp(-((x-1005-shift)/3)**2)+.01*np.sin(x*.4)
    np.savetxt(path,np.column_stack([x,y]))
    return x,y


def test_experimental_calculator_matches_unchanged_legacy(tmp_path):
    path=tmp_path/'wave.txt';x,y=signal(path)
    original=extract_extension_block1_features(path)
    # Read through the same loader to avoid parser-rounding differences.
    from src.features.loading import load_measurement
    w=load_measurement(path);x,y=w.x.to_numpy(),w.y.to_numpy()
    ix=np.flatnonzero((x>=980)&(x<=1010));anchor=ix[np.argmax(abs((y-np.median(y))[ix]))]
    experimental=extension_at_anchor(x,y,anchor)
    assert set(experimental)==set(original) and len(original)==35
    np.testing.assert_allclose(list(experimental.values()),list(original.values()),rtol=0,atol=0)
    early=early_index(x,y)
    assert abs(x[early]-975)<.1
    assert 6<=adapted_start(x,y,early)<=9


def test_template_cannot_use_excluded_waveforms_or_labels(tmp_path):
    rows=[]
    for i in range(8):
        path=tmp_path/f'{i}.txt';signal(path,shift=(i%3)*.1)
        rows.append([str(i),str(i//2),str(path),i%4])
    data=pd.DataFrame(rows,columns=['instance_id','specimen_id','file_path','label'])
    training=data.iloc[:6];first=fit_template(training,per_specimen=2)
    signal(tmp_path/'6.txt',shift=30);signal(tmp_path/'7.txt',shift=40)
    altered=training.copy();altered['label']=99
    second=fit_template(altered,per_specimen=2)
    assert set(first.member_ids).isdisjoint(set(data.iloc[6:].instance_id))
    np.testing.assert_array_equal(first.values,second.values)
    path=tmp_path/'new.txt';x,y=signal(path,shift=1)
    anchor,score=template_index(x,y,first)
    assert abs(x[anchor]-976)<.3 and score>.8


def test_failed_extraction_preserves_nan_extension(tmp_path):
    path=tmp_path/'flat.txt';np.savetxt(path,np.column_stack([np.arange(950,1180,.05),np.ones(4600)]))
    features,diagnostic=extract_variant(path,'early_adapted',[str(i) for i in range(35)])
    assert np.isnan(features).all() and diagnostic['status']=='flat_signal'


def test_cv_is_group_disjoint_and_order_independent():
    data=pd.DataFrame([[f'{g:02d}_{label}',f'g{g}',label] for g in range(23) for label in range(4)],columns=['instance_id','specimen_id','label'])
    ordered,folds=make_folds(data);other,again=make_folds(data.sample(frac=1,random_state=7))
    pd.testing.assert_frame_equal(ordered,other);np.testing.assert_array_equal(folds,again)
    assert set(folds)=={0,1,2}
    for fold in range(3):
        assert not set(ordered.specimen_id[folds==fold])&set(ordered.specimen_id[folds!=fold])


def test_imputation_and_scaler_fit_only_fold_training():
    factory=create_model('logistic_regression');before=factory.named_steps['model'].get_params()
    pipeline=fold_pipeline(factory)
    X=np.array([[0,np.nan],[2,4],[4,6],[6,8]],dtype=float)
    pipeline.fit(X,[0,1,0,1]);state=pipeline.named_steps['imputer'].statistics_.copy()
    np.testing.assert_allclose(state,[3,6])
    np.testing.assert_allclose(pipeline.named_steps['scale'].mean_,[3,6])
    pipeline.predict([[1e9,np.nan]])
    np.testing.assert_array_equal(state,pipeline.named_steps['imputer'].statistics_)
    assert before==pipeline.named_steps['model'].get_params()
