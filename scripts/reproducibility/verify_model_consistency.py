"""Reusable entry point for the preserved historical audit (no training)."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--reference-run-id", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    root = project_root
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Output directory must not exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    """Read-only numerical/model audit; no fit, feature extraction or MLflow writes."""
    import json,sys,shutil,time,hashlib,subprocess
    from datetime import datetime,timezone
    from uuid import uuid4
    ROOT=root
    sys.path.insert(0,str(ROOT))
    import numpy as np
    import pandas as pd
    import joblib
    from src.data.dataset import prepare_dataset
    from src.data.split_manifest import restore_splits,sha256_file
    from src.evaluation.metrics import evaluate_predictions,calculate_error_counts
    from src.experiments.provenance import check_locked_environment

    RTOL=1e-11
    ATOL=1e-13
    BASE=args.rebuilt.parent.parent
    REFERENCE=args.reference
    REBUILT=args.rebuilt
    SPLITS=ROOT/'configs/hgb_baseline/splits'
    preflight={'reference_run_id':args.reference_run_id}
    MODEL=args.model
    OUT=args.output_dir

    shutil.copy2(__file__,OUT/'verify_model_consistency.py')
    check_locked_environment(ROOT/'requirements-lock.txt')
    start=time.monotonic()
    protected={str(p):sha256_file(p) for p in [REFERENCE,REBUILT,MODEL,ROOT/'mlflow.db']}
    a=pd.read_csv(REFERENCE);b=pd.read_csv(REBUILT)
    assert list(a.columns)==list(b.columns)
    assert a.instance_id.tolist()==b.instance_id.tolist()
    X,y=prepare_dataset(a);new_X,new_y=prepare_dataset(b)
    assert y.equals(new_y)
    assert list(X.columns)==list(new_X.columns)
    cols=list(X.columns);av=X.to_numpy();bv=new_X.to_numpy()
    assert np.array_equal(np.isfinite(av),np.isfinite(bv))
    assert np.array_equal(np.isnan(av),np.isnan(bv))
    delta=np.abs(bv-av)
    exact=(av==bv)|(np.isnan(av)&np.isnan(bv))
    relative=np.zeros_like(delta)
    nonzero=av!=0
    np.divide(delta,np.abs(av),out=relative,where=nonzero)
    relative[(av==0)&(bv!=0)]=np.inf
    within=np.isclose(bv,av,rtol=RTOL,atol=ATOL,equal_nan=True)
    feature_rows=[]
    for j,name in enumerate(cols):
        feature_rows.append({'feature':name,'different_cells':int((~exact[:,j]).sum()),
          'max_abs_delta':float(np.nanmax(delta[:,j])),
          'max_relative_delta_reference_denominator':float(np.nanmax(relative[:,j])),
          'reference_zero_rebuilt_nonzero_cells':int(((av[:,j]==0)&(bv[:,j]!=0)).sum()),
          'outside_proposed_tolerance':int((~within[:,j]).sum())})
    pd.DataFrame(feature_rows).to_csv(OUT/'per_feature_differences.csv',index=False)
    _,val,_,_,val_y,_=restore_splits(X,y,a.instance_id,SPLITS)
    _,new_val,_,_,new_val_y,_=restore_splits(new_X,new_y,b.instance_id,SPLITS)
    assert val.index.tolist()==new_val.index.tolist()
    assert val_y.equals(new_val_y)
    model=joblib.load(MODEL)
    assert list(model.feature_names_in_)==list(val.columns)
    old_prediction=model.predict(val)
    new_prediction=model.predict(new_val)
    old_proba=model.predict_proba(val)
    new_proba=model.predict_proba(new_val)
    saved=pd.read_csv(args.predictions)
    assert np.array_equal(val_y,saved.y_true)
    old_metrics=evaluate_predictions(val_y,old_prediction)
    new_metrics=evaluate_predictions(new_val_y,new_prediction)
    pd.DataFrame({'instance_id':a.loc[val.index,'instance_id'].to_numpy(),'y_true':val_y.to_numpy(),
     'reference_prediction':old_prediction,'rebuilt_prediction':new_prediction,
     'same':old_prediction==new_prediction}).to_csv(OUT/'validation_prediction_comparison.csv',index=False)
    absolute_location=np.unravel_index(np.nanargmax(delta),delta.shape)
    relative_location=np.unravel_index(np.nanargmax(relative),relative.shape)
    result={
     'output_dir':str(OUT),'reference_model_path':str(MODEL),'reference_mlflow_run_id':preflight['reference_run_id'],
     'git_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
     'reference_sha256':protected[str(REFERENCE)],'rebuilt_sha256':protected[str(REBUILT)],'model_sha256':protected[str(MODEL)],
     'rows':len(a),'feature_count':len(cols),'different_features':sum(r['different_cells']>0 for r in feature_rows),
     'exact_features':[r['feature'] for r in feature_rows if r['different_cells']==0],
     'numeric_cell_count':int(av.size),'different_cells':int((~exact).sum()),
     'max_abs_delta':float(np.nanmax(delta)),
     'max_abs_location':{'instance_id':a.instance_id.iloc[absolute_location[0]],'feature':cols[absolute_location[1]]},
     'max_relative_delta':float(np.nanmax(relative)),
     'max_relative_location':{'instance_id':a.instance_id.iloc[relative_location[0]],'feature':cols[relative_location[1]],
      'reference_value':float(av[relative_location]),'rebuilt_value':float(bv[relative_location]),'absolute_delta':float(delta[relative_location])},
     'reference_zero_rebuilt_nonzero_cells':int(((av==0)&(bv!=0)).sum()),
     'rtol':RTOL,'atol':ATOL,'tolerance_passes':bool(within.all()),'outside_tolerance':int((~within).sum()),
     'max_tolerance_budget_fraction':float(np.nanmax(delta/(ATOL+RTOL*np.abs(av)))),
     'validation_count':len(val),'prediction_mismatches':int(np.count_nonzero(old_prediction!=new_prediction)),
     'reference_predictions_match_saved':bool(np.array_equal(old_prediction,saved.y_pred)),
     'rebuilt_predictions_match_saved':bool(np.array_equal(new_prediction,saved.y_pred)),
     'probabilities_exact_equal':bool(np.array_equal(old_proba,new_proba)),
     'max_probability_delta':float(np.max(np.abs(old_proba-new_proba))),
     'metrics':{k:{'reference':old_metrics[k],'rebuilt':new_metrics[k]} for k in ['accuracy','balanced_accuracy','macro_f1','weighted_f1']},
     'confusion_matrix_equal':bool(np.array_equal(old_metrics['confusion_matrix'],new_metrics['confusion_matrix'])),
     'confusion_matrix':old_metrics['confusion_matrix'].tolist(),
     'safety_metrics_equal':calculate_error_counts(val_y,old_prediction)==calculate_error_counts(new_val_y,new_prediction),
     'safety_metrics':calculate_error_counts(val_y,new_prediction),
     'protected_inputs_unchanged':all(sha256_file(p)==h for p,h in protected.items()),
     'seconds':time.monotonic()-start,
    }
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'per_feature_differences.md').write_text('| Feature | Abweichende Zellen | Max. absolut | Max. relativ (Referenz) |\n| --- | ---: | ---: | ---: |\n'+''.join(f'| {r["feature"]} | {r["different_cells"]} | {r["max_abs_delta"]:.8g} | {r["max_relative_delta_reference_denominator"]:.8g} |\n' for r in feature_rows))
    print(json.dumps(result,indent=2),flush=True)

if __name__ == "__main__":
    main()
