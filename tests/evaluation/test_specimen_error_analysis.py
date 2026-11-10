"""Deterministic diagnostics: denominators, alignment, conditioning and train scales."""
import numpy as np
import pandas as pd
import pytest

from src.analysis.specimen_error_analysis import (
    assemble_points,
    distribution_shift,
    error_associations,
    grouped_errors,
    overlap_table,
)


def saved_data():
    rows=[]
    for split in ['validation','test']:
        for label in range(4):
            rows.append([f'{split}_{label}',f'body_{split}','source',label,split,label,0,'unused'])
    data=pd.DataFrame(rows,columns=['instance_id','specimen_id','source_dataset','label','split','x','y','file_path'])
    predictions={}
    for model in range(7):
        predictions[f'm{model}']={}
        for split in ['validation','test']:
            frame=data.loc[data.split.eq(split),['instance_id','specimen_id','source_dataset','label']].copy()
            frame['prediction']=frame.label if model==0 else 3
            predictions[f'm{model}'][split]=frame
    return data,predictions


def test_critical_votes_exclude_label_zero_and_denominators_are_explicit():
    data,predictions=saved_data()
    points,models=assemble_points(data.drop(columns=['x','y','file_path']),data,predictions)
    assert points.loc[points.label.eq(0),'critical_votes'].eq(0).all()
    assert points.loc[points.label.isin([1,2]),'critical_votes'].eq(6).all()
    assert not points.all7_critical.any()
    grouped=grouped_errors(points,models,['split'])
    assert grouped.critical_support.tolist()==[2,2]
    assert grouped.model_error_events.tolist()==[12,12]
    assert np.allclose(grouped.mean_model_critical_rate,100*12/14)
    overlap=overlap_table(points,models)
    assert overlap.query("model_a=='m0' and model_b=='m1'").jaccard.eq(0).all()
    assert overlap.query("model_a=='m1' and model_b=='m2'").jaccard.eq(1).all()
    shuffled,_=assemble_points(data.drop(columns=['x','y','file_path']).sample(frac=1,random_state=42),data,predictions)
    pd.testing.assert_frame_equal(points,shuffled)


@pytest.mark.parametrize('problem',['missing','label','group'])
def test_saved_predictions_must_match_frozen_metadata(problem):
    data,predictions=saved_data()
    p=predictions['m0']['validation']
    if problem=='missing': predictions['m0']['validation']=p.iloc[:-1]
    elif problem=='label': p.loc[p.index[0],'label']=1
    else: p.loc[p.index[0],'specimen_id']='wrong'
    with pytest.raises(ValueError): assemble_points(data.drop(columns=['x','y','file_path']),data,predictions)


def test_shift_is_class_conditioned_and_scaled_by_training_only():
    rows=[]
    for split in ['train','validation','test']:
        for label in range(4):
            for value in [0.,2.]:
                rows.append([f'{split}_{label}_{value}',split,label,value+(10 if split=='validation' else 0)])
    data=pd.DataFrame(rows,columns=['instance_id','split','label','f'])
    shift=distribution_shift(data[['instance_id','f']],data[['instance_id','split','label']],['f'])
    assert shift.query("split=='test'").w1_train_iqr.eq(0).all()
    assert shift.query("split=='validation' and true_class=='all'").w1_train_iqr.iloc[0]==5
    assert shift.query("split=='validation' and true_class=='1'").w1_train_iqr.iloc[0]==10


def test_feature_effects_keep_body_and_true_class_separate():
    points=pd.DataFrame({'instance_id':list('abcd'),'split':['validation']*4,
                        'specimen_id':['A','A','B','B'],'label':[1]*4,
                        'critical_truth':[True]*4,'all7_critical':[True,False,True,False],
                        'critical_votes':[7,0,7,0]})
    features=pd.DataFrame({'instance_id':list('abcd')+['t1','t2'],'f':[2.,0.,100.,102.,0.,4.]})
    membership=pd.DataFrame({'instance_id':['t1','t2'],'split':['train']*2,'label':[1,1]})
    effects=error_associations(points,features,membership,['f'],min_n=1)
    assert effects.specimen_id.tolist()==['A','B']
    assert effects.delta_train_class_iqr.tolist()==[1.,-1.]
