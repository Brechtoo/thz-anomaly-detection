"""Reusable entry point for the preserved historical audit (no training)."""
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    root = project_root
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Output directory must not exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import json,time,resource,shutil,sys
    import numpy as np,pandas as pd
    root=project_root;sys.path.insert(0,str(root))
    from src.features.extract_all_features import extract_all_features
    from src.data.dataset import EXCLUDE_COLUMNS
    out=args.output_dir
    combined=pd.read_csv(args.catalog)
    reference=pd.read_csv(args.reference)
    paths=[Path(p) for p in combined.file_path]
    sizes=[p.stat().st_size for p in paths]
    columns=[c for c in reference if c not in EXCLUDE_COLUMNS]
    sample=combined.iloc[np.linspace(0,len(combined)-1,64,dtype=int)]
    reference_by_id=reference.set_index('instance_id')
    start=time.monotonic();rows=[]
    for p in sample.file_path:rows.append(extract_all_features(p))
    duration=time.monotonic()-start
    actual=pd.DataFrame(rows)[columns].to_numpy()
    expected=reference_by_id.loc[sample.instance_id,columns].to_numpy()
    delta=np.abs(actual-expected)
    # Historical NumPy trapz formula for 1-D y/x, matching our two call sites.
    def legacy_trapz(y,x):
     return (np.diff(x)*(y[1:]+y[:-1])/2.0).sum()
    modern=np.trapezoid
    try:
     np.trapezoid=legacy_trapz
     legacy=pd.DataFrame([extract_all_features(p) for p in sample.file_path])[columns].to_numpy()
    finally:
     np.trapezoid=modern
    result={'rows':len(combined),'raw_total_bytes':sum(sizes),'raw_max_bytes':max(sizes),
     'reference_csv_bytes':(args.reference).stat().st_size,
     'free_disk_bytes':shutil.disk_usage(root).free,'probe_rows':len(sample),'probe_seconds':duration,
     'extraction_estimate_seconds':duration/len(sample)*len(combined),
     'probe_process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
     'estimated_output_budget_bytes':1024**3,'estimated_peak_memory_budget_bytes':2*1024**3,
     'catalog_id_order_equals_reference':combined.instance_id.tolist()==reference.instance_id.tolist(),
     'probe_exact_equal':bool(np.array_equal(actual,expected,equal_nan=True)),
     'probe_close_rtol1e-12_atol1e-12':bool(np.allclose(actual,expected,rtol=1e-12,atol=1e-12,equal_nan=True)),
     'probe_max_abs_delta':float(np.nanmax(delta)),
     'probe_trapezoid_equals_legacy_formula':bool(np.array_equal(actual,legacy,equal_nan=True)),
     'output_dir':str(out/'feature_rebuild'),'output_target_exists':(out/'feature_rebuild').exists()}
    (out/'feature_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame({'instance_id':sample.instance_id.to_numpy(),'raw_bytes':[Path(p).stat().st_size for p in sample.file_path]}).to_csv(out/'probe_samples.csv',index=False)
    print(json.dumps(result,indent=2),flush=True)

if __name__ == "__main__":
    main()
