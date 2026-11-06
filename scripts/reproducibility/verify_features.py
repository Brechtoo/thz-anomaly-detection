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
    """Full feature rebuild audit; no model loading/training or MLflow writes."""
    import json
    import resource
    import sys
    import time

    import numpy as np
    import pandas as pd
    ROOT=root
    sys.path.insert(0,str(ROOT))
    from src.data.dataset import EXCLUDE_COLUMNS
    from src.data.split_manifest import sha256_file
    from src.experiments.provenance import check_locked_environment
    from src.features.rebuild_feature_table import rebuild_feature_table
    OUT=args.output_dir
    original_trapezoid=np.trapezoid
    calls=0; formula_mismatches=0

    def audited_trapezoid(y,x):
        nonlocal calls,formula_mismatches
        current=original_trapezoid(y,x)
        legacy=(np.diff(x)*(y[1:]+y[:-1])/2.0).sum()
        calls+=1
        formula_mismatches+=int(np.asarray(current).tobytes()!=np.asarray(legacy).tobytes())
        return current

    check_locked_environment(ROOT/'requirements-lock.txt')
    start=time.monotonic()
    np.trapezoid=audited_trapezoid
    try:
        path=rebuild_feature_table(args.catalog,OUT/'feature_rebuild')
    finally:
        np.trapezoid=original_trapezoid
    extraction_seconds=time.monotonic()-start
    reference_path=args.reference
    # Normal parsing mirrors the baseline; round-trip parsing isolates decimal I/O.
    ref=pd.read_csv(reference_path)
    new=pd.read_csv(path)
    columns=[c for c in ref if c not in EXCLUDE_COLUMNS]
    a=ref[columns].to_numpy();b=new[columns].to_numpy()
    exact=(a==b)|(np.isnan(a)&np.isnan(b))
    close=np.isclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True)
    per_feature=[]
    for j,c in enumerate(columns):
        diff=np.abs(a[:,j]-b[:,j])
        per_feature.append({'feature':c,'different_cells':int((~exact[:,j]).sum()),'outside_tolerance':int((~close[:,j]).sum()),'max_abs_delta':float(np.nanmax(diff))})
    pd.DataFrame(per_feature).to_csv(OUT/'feature_differences.csv',index=False)
    metadata_columns=[c for c in EXCLUDE_COLUMNS if c in ref and c in new]
    metadata_equal=ref[metadata_columns].equals(new[metadata_columns])
    ref_roundtrip=pd.read_csv(reference_path,float_precision='round_trip')
    new_roundtrip=pd.read_csv(path,float_precision='round_trip')
    ar=ref_roundtrip[columns].to_numpy();br=new_roundtrip[columns].to_numpy()
    roundtrip_exact=(ar==br)|(np.isnan(ar)&np.isnan(br))
    # Quantify the normal parser's own rounding error without another extraction.
    ref_parser_equal=(a==ar)|(np.isnan(a)&np.isnan(ar))
    new_parser_equal=(b==br)|(np.isnan(b)&np.isnan(br))
    result={
        'rows':len(new),'columns':len(new.columns),'feature_count':len(columns),
        'schema_equal':list(ref.columns)==list(new.columns),
        'dtypes_equal':ref.dtypes.astype(str).to_dict()==new.dtypes.astype(str).to_dict(),
        'id_order_equal':ref.instance_id.tolist()==new.instance_id.tolist(),
        'metadata_equal':metadata_equal,
        'exact_numeric_equal_default_parser':bool(exact.all()),
        'different_numeric_cells_default_parser':int((~exact).sum()),
        'numeric_equal_rtol1e-12_atol1e-12':bool(close.all()),
        'cells_outside_tolerance':int((~close).sum()),
        'max_abs_delta':float(np.nanmax(np.abs(a-b))),
        'max_scaled_delta':float(np.nanmax(np.abs(a-b)/(1+np.abs(a)))),
        'roundtrip_exact_numeric_equal':bool(roundtrip_exact.all()),
        'roundtrip_different_numeric_cells':int((~roundtrip_exact).sum()),
        'reference_cells_changed_by_parser_choice':int((~ref_parser_equal).sum()),
        'new_cells_changed_by_parser_choice':int((~new_parser_equal).sum()),
        'nan_masks_equal':bool(np.array_equal(np.isnan(a),np.isnan(b))),
        'reference_sha256':sha256_file(reference_path),'rebuilt_sha256':sha256_file(path),
        'sha256_equal':sha256_file(reference_path)==sha256_file(path),
        'trapezoid_calls':calls,'trapezoid_vs_legacy_formula_bitwise_mismatches':formula_mismatches,
        'extraction_and_hash_seconds':extraction_seconds,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'python_executable':sys.executable,'feature_path':str(path),
    }
    # A few differing serialized cells document exact decimal strings independently.
    samples=[]
    for j,c in enumerate(columns):
        for i in np.flatnonzero(~roundtrip_exact[:,j])[:2]:
            samples.append({'instance_id':ref.instance_id.iloc[i],'feature':c,'reference_value_roundtrip':float(ar[i,j]),'rebuilt_value_roundtrip':float(br[i,j])})
    result['difference_samples']=samples[:20]
    (OUT/'feature_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPARISON',json.dumps(result),flush=True)

if __name__ == "__main__":
    main()
