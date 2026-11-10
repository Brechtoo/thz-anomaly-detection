"""Isolated, fixed-hypothesis Group-CV on original TRAIN bodies only.

No original validation/test CSV, waveform, prediction or score is consumed.
Templates fit inside outer training folds; all imputation/scaling/variance removal
fits there too. Fixed-variant OOF comparisons are exploratory, not an unbiased
post-selection estimate of a winning variant. Existing artifacts are read-only.
"""
import argparse
import importlib.metadata
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.data.reference_measurements import require_no_references
from src.data.split_manifest import sha256_file
from src.evaluation.metrics import calculate_error_counts, evaluate_predictions
from src.experiments.model_registry import build_models
from src.experiments.provenance import capture_git_state, snapshot_source_tree
from src.features.experimental_anchor_calculator import extension_at_anchor
from src.features.experimental_anchors import VARIANTS, extract_variant, fit_template
from src.features.extract_extension_block1_features import (
    extract_extension_block1_features,
)
from src.features.loading import load_measurement
from src.paths import PROJECT_ROOT

SPLITS=PROJECT_ROOT/'configs/specimen_holdout_v1/splits'
SOURCE_TABLE=PROJECT_ROOT/'results/feature_tables/feature_table.csv'


def train_cohort():
    """Read train IDs first; stream/discard other rows before any statistics."""
    manifest=json.loads((SPLITS/'manifest.json').read_text())
    if sha256_file(SPLITS/'train.csv')!=manifest['files']['train.csv']:raise ValueError('Train manifest changed')
    train=pd.read_csv(SPLITS/'train.csv');ids=set(train.instance_id)
    if len(train)!=34235 or not train.instance_id.is_unique:raise ValueError('Unexpected original train cohort')
    meta=pd.read_csv(SPLITS/'membership.csv',usecols=['instance_id','specimen_id','source_dataset'])
    cohort=train.merge(meta,on='instance_id',validate='one_to_one').sort_values('instance_id').reset_index(drop=True)
    require_no_references(cohort)
    schema=json.loads((PROJECT_ROOT/'configs/specimen_holdout_v1/manual_features.json').read_text())
    if sha256_file(SOURCE_TABLE)!=schema['source_table_sha256']:raise ValueError('Original feature table changed')
    columns=schema['features'];frames=[]
    for chunk in pd.read_csv(SOURCE_TABLE,chunksize=6000,usecols=['instance_id','file_path',*columns]):
        frames.append(chunk.loc[chunk.instance_id.isin(ids)])
    raw=pd.concat(frames).set_index('instance_id').loc[cohort.instance_id]
    cohort['file_path']=raw.file_path.to_numpy()
    X=raw[columns].to_numpy()
    if cohort.specimen_id.nunique()!=23 or not np.isfinite(X).all():raise ValueError('Incomplete train features')
    extension=columns[17:]
    if len(extension)!=35:raise ValueError('Expected original 17+35 schema')
    return cohort,columns,extension,X


def make_folds(cohort,n_splits=3):
    """Canonical order makes SGKF independent of incoming table order."""
    ordered=cohort.sort_values('instance_id').reset_index(drop=True)
    splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=42)
    folds=np.full(len(ordered),-1)
    for fold,(train,val) in enumerate(splitter.split(ordered,ordered.label,ordered.specimen_id)):
        if set(ordered.specimen_id.iloc[train])&set(ordered.specimen_id.iloc[val]):raise ValueError('Group leakage')
        if set(ordered.label.iloc[train])!={0,1,2,3} or set(ordered.label.iloc[val])!={0,1,2,3}:raise ValueError('Missing class in fold')
        folds[val]=fold
    return ordered,folds


def work(task):
    path,variant,columns,template=task
    return extract_variant(path,variant,columns,template)


def extract_matrix(cohort,base,variant,columns,template,out,pool):
    records=list(pool.map(work,[(p,variant,columns,template) for p in cohort.file_path],chunksize=32))
    matrix=base.copy();matrix[:,17:]=np.array([r[0] for r in records])
    pd.DataFrame([dict(instance_id=i,**r[1]) for i,r in zip(cohort.instance_id,records)]).to_csv(out.with_suffix('.csv'),index=False)
    np.savez_compressed(out,X=matrix)
    print(f'Extracted {variant}: failures={sum(r[1]["status"]!="ok" for r in records)}/{len(cohort)}',flush=True)


def prepare(output,workers=4):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    cohort,columns,extension,base=train_cohort();cohort,folds=make_folds(cohort)
    output.mkdir(parents=True,exist_ok=False);(output/'features').mkdir();(output/'templates').mkdir();(output/'metadata').mkdir()
    # Freeze the full protocol before computing alternatives or training models.
    plan={'seed':42,'folds':3,'variants':list(VARIANTS),'features':columns,'cohort':'original train IDs only',
          'early':'first abs lobe height>=.35*max, prominence>=.20*max, search960..1030',
          'template':'fold-train only, 12 IDs/specimen, median sign-normalized pulse [-2.5,4.5], abs NCC>=.45, amplitude gate .35',
          'adapted':'post start clip(packet_end+1,6,9), smoothed abs level .20, packet cap8; end80; noise -10..-5; correlation +/-2.5 unchanged',
          'failure':'retain all IDs; 35 extension values NaN; fold-train median imputation',
          'selection':'no tuning, no old holdout use; report class1/2 critical errors separately, false alarms, macro-F1',
          'limitation':'Hypotheses were motivated by earlier holdout reviews; this is not a fresh confirmatory study.'}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    cohort.assign(fold=folds).to_csv(output/'cohort_folds.csv',index=False)
    snapshot_source_tree(PROJECT_ROOT/'src',output/'metadata/source')
    (output/'metadata/git.json').write_text(json.dumps(capture_git_state(PROJECT_ROOT),indent=2)+'\n')
    (output/'metadata/versions.json').write_text(json.dumps({d.metadata['Name']:d.version for d in importlib.metadata.distributions()},indent=2)+'\n')
    raw_hashes={p:sha256_file(p) for p in cohort.file_path}
    (output/'raw_sha256.json').write_text(json.dumps(raw_hashes,indent=2)+'\n')
    # Legacy equality on fixed training-only IDs guards the isolated formula fork.
    parity=[]
    for index in np.linspace(0,len(cohort)-1,64,dtype=int):
        path=cohort.file_path.iloc[index];wave=load_measurement(path);x,y=wave.x.to_numpy(),wave.y.to_numpy()
        ix=np.flatnonzero((x>=980)&(x<=1010));anchor=ix[np.argmax(abs((y-np.median(y))[ix]))]
        expected=extract_extension_block1_features(path);actual=extension_at_anchor(x,y,anchor)
        if not all(np.isclose(expected[k],actual[k],rtol=1e-12,atol=1e-13) for k in extension):raise ValueError('Experimental formula parity failed')
        if not np.allclose([expected[k] for k in extension],base[index,17:],rtol=1e-11,atol=1e-13):raise ValueError('Legacy cache/raw mismatch')
        parity.append(cohort.instance_id.iloc[index])
    (output/'legacy_parity_ids.json').write_text(json.dumps(parity,indent=2)+'\n')
    np.savez_compressed(output/'features/legacy.npz',X=base)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for variant in ['early_legacy_windows','early_adapted']:
            extract_matrix(cohort,base,variant,extension,None,output/'features'/f'{variant}.npz',pool)
        for fold in range(3):
            train=cohort.loc[folds!=fold]
            template=fit_template(train)
            assert set(template.member_ids)<=set(train.instance_id)
            np.savez_compressed(output/'templates'/f'fold_{fold}.npz',values=template.values,member_ids=template.member_ids)
            extract_matrix(cohort,base,'template_adapted',extension,template,output/'features'/f'template_fold_{fold}.npz',pool)
    if any(sha256_file(p)!=h for p,h in raw_hashes.items()):raise ValueError('Raw input changed')
    manifest={'source_table_sha256':sha256_file(SOURCE_TABLE),'train_sha256':sha256_file(SPLITS/'train.csv'),
              'plan_sha256':sha256_file(output/'plan.json'),'cohort_sha256':sha256_file(output/'cohort_folds.csv'),
              'features_sha256':{p.name:sha256_file(p) for p in (output/'features').iterdir()},
              'templates_sha256':{p.name:sha256_file(p) for p in (output/'templates').iterdir()}}
    (output/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def scores(y,pred):
    result=evaluate_predictions(y,pred)
    result={k:float(v) for k,v in result.items() if k not in ['confusion_matrix','classification_report']}
    result.update(calculate_error_counts(y,pred))
    for label in [1,2]:
        n=int(np.sum(np.asarray(y)==label));errors=int(np.sum((np.asarray(y)==label)&(np.asarray(pred)==3)))
        result[f'critical_{label}_support']=n;result[f'critical_{label}_errors']=errors
        result[f'critical_{label}_rate']=100*errors/n if n else np.nan
    return result


def fold_pipeline(factory):
    """Only new step: fold-training median imputation before unchanged registry steps."""
    return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),*factory.steps])


def train(output):
    output=Path(output);destination=output/'cv'
    if destination.exists():raise FileExistsError(destination)
    manifest=json.loads((output/'extraction_manifest.json').read_text())
    if sha256_file(output/'cohort_folds.csv')!=manifest['cohort_sha256']:raise ValueError('Cohort changed')
    for name,h in manifest['features_sha256'].items():
        if sha256_file(output/'features'/name)!=h:raise ValueError('Features changed')
    data=pd.read_csv(output/'cohort_folds.csv');folds=data.fold.to_numpy();y=data.label.to_numpy()
    destination.mkdir();(destination/'models').mkdir();(destination/'predictions').mkdir()
    mlflow.set_tracking_uri('sqlite:///'+str((destination/'mlflow.db').resolve()))
    client=mlflow.MlflowClient();client.create_experiment('anchor-training-bodies-group-cv',artifact_location=(destination/'mlartifacts').resolve().as_uri())
    mlflow.set_experiment('anchor-training-bodies-group-cv')
    mlflow.sklearn.autolog(disable=True)
    rows=[];oof={(variant,name):np.full(len(data),-1) for variant in VARIANTS for name in build_models()}
    for fold in range(3):
        tr,va=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        for variant in VARIANTS:
            filename=f'template_fold_{fold}.npz' if variant=='template_adapted' else f'{variant}.npz'
            with np.load(output/'features'/filename) as saved:X=saved['X']
            for name,factory in build_models(seed=42).items():
                print(f'Fold {fold+1}/3 {variant} {name}',flush=True)
                model=fold_pipeline(factory);params={k:str(v) for k,v in factory.get_params(deep=True).items()}
                with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=1):
                    warnings.simplefilter('always');start=time.perf_counter();model.fit(X[tr],y[tr]);prediction=model.predict(X[va]);elapsed=time.perf_counter()-start
                oof[(variant,name)][va]=prediction
                metrics=scores(y[va],prediction);row=dict(fold=fold,variant=variant,model=name,train_groups=data.specimen_id.iloc[tr].nunique(),val_groups=data.specimen_id.iloc[va].nunique(),seconds=elapsed,**metrics)
                rows.append(row);pd.DataFrame(rows).to_csv(destination/'fold_metrics.csv',index=False)
                tag=f'{variant}_{name}_fold{fold}';path=destination/'models'/f'{tag}.joblib';joblib.dump(model,path)
                diagnostic={'parameters':params,'warnings':[str(w.message) for w in caught],
                            'imputer_statistics':model.named_steps['imputer'].statistics_.tolist(),
                            'retained_feature_indices':model.named_steps['constants'].get_support(indices=True).tolist()}
                detail=destination/'models'/f'{tag}.json';detail.write_text(json.dumps(diagnostic,indent=2)+'\n')
                with mlflow.start_run(run_name=tag) as run:
                    mlflow.log_params(params);mlflow.set_tags({'variant':variant,'fold':fold,'scope':'original_train_bodies_only','no_original_holdout_used':True})
                    mlflow.log_metrics({k:float(v) for k,v in metrics.items() if np.isfinite(v)})
                    for artifact in [path,detail,output/'plan.json',output/'cohort_folds.csv',output/'extraction_manifest.json']:
                        mlflow.log_artifact(str(artifact),'experiment')
                    if variant=='template_adapted':mlflow.log_artifact(str(output/'templates'/f'fold_{fold}.npz'),'template')
                    row['run_id']=run.info.run_id
                pd.DataFrame(rows).to_csv(destination/'fold_metrics.csv',index=False)
    pooled=[];bodies=[]
    for (variant,name),prediction in oof.items():
        if (prediction<0).any():raise ValueError('Incomplete OOF')
        frame=data[['instance_id','specimen_id','source_dataset','label','fold']].copy();frame['prediction']=prediction
        frame.to_csv(destination/'predictions'/f'{variant}_{name}.csv',index=False)
        pooled.append(dict(variant=variant,model=name,**scores(y,prediction)))
        for body,part in frame.groupby('specimen_id'):
            bodies.append(dict(variant=variant,model=name,specimen_id=body,points=len(part),**scores(part.label,part.prediction)))
        pd.DataFrame(confusion_matrix(y,prediction,labels=[0,1,2,3])).to_csv(destination/'predictions'/f'{variant}_{name}_cm.csv',index=False)
    pd.DataFrame(pooled).to_csv(destination/'pooled_oof_metrics.csv',index=False)
    pd.DataFrame(bodies).to_csv(destination/'specimen_oof_metrics.csv',index=False)
    print('All 84 fold fits complete; no original validation/test evaluated.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','train']);parser.add_argument('--output-dir',required=True,type=Path);parser.add_argument('--workers',default=4,type=int)
    args=parser.parse_args()
    prepare(args.output_dir,args.workers) if args.stage=='prepare' else train(args.output_dir)
