"""Reusable entry point for the preserved historical audit (no training)."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--prior-check-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    root = project_root
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Output directory must not exist")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    """Reconstruct catalog and audit existing splits. No training or MLflow writes."""
    import json
    import shutil
    import sys
    import time

    import numpy as np
    import pandas as pd
    ROOT=root;sys.path.insert(0,str(ROOT))
    from src.data.dataset import prepare_dataset
    from src.data.labeling.rebuild_catalog import DATASET_SPECS, rebuild_catalog
    from src.data.split_manifest import restore_splits, sha256_file
    from src.evaluation.leakage import (
        group_overlap_table,
        parse_acquisition_metadata,
        spatial_train_proximity,
    )
    RAW=args.raw_root
    OLD=args.prior_check_dir
    OUT=args.output_dir

    shutil.copy2(__file__,OUT/'audit_catalog_leakage.py')
    started=time.monotonic()
    protected=json.loads((OLD/'initial_state.json').read_text())['protected_sha256']
    annotation_hashes={str(RAW/folder/name):sha256_file(RAW/folder/name) for _,folder,name in DATASET_SPECS}
    split_hashes={str(p):sha256_file(p) for p in (ROOT/'configs/hgb_baseline/splits').glob('*') if p.is_file()}
    first=rebuild_catalog(RAW,OUT/'catalog_rebuild_1')
    second=rebuild_catalog(RAW,OUT/'catalog_rebuild_2')
    a=pd.read_csv(ROOT/'results/measurements/combined_dataset.csv');b=pd.read_csv(first)
    by_id=a.set_index('instance_id');new_by_id=b.set_index('instance_id')
    ids_equal=set(a.instance_id)==set(b.instance_id)
    reconstruction={
     'rows_reference':len(a),'rows_rebuilt':len(b),'columns_equal':a.columns.tolist()==b.columns.tolist(),
     'ids_equal':ids_equal,'id_order_equal':a.instance_id.tolist()==b.instance_id.tolist(),
     'labels_equal_by_id':bool(ids_equal and by_id.label.sort_index().equals(new_by_id.label.sort_index())),
     'all_catalog_fields_equal':a.equals(b),'different_cells_by_column':{c:int((a[c]!=b[c]).sum()) for c in a.columns},
     'reference_sha256':sha256_file(ROOT/'results/measurements/combined_dataset.csv'),
     'rebuilt_sha256':sha256_file(first),'repeated_rebuild_sha256_equal':sha256_file(first)==sha256_file(second),
     'byteidentical_to_reference':sha256_file(first)==sha256_file(ROOT/'results/measurements/combined_dataset.csv'),
     'annotation_hashes':annotation_hashes,'new_labels_assigned':0,
    }
    # No positional re-splitting: exact saved IDs determine all memberships.
    xt,xv,xe,yt,yv,ye=restore_splits(a[['x','y']],a.label,a.instance_id,ROOT/'configs/hgb_baseline/splits')
    d=a.copy();d['split']=None
    for name,part in [('train',xt),('validation',xv),('test',xe)]:d.loc[part.index,'split']=name
    metadata=parse_acquisition_metadata(d.filename)
    d=pd.concat([d,metadata.drop(columns='coordinates')],axis=1)
    d['block_x_10']=np.floor(d.x/10).astype(int);d['block_y_10']=np.floor(d.y/10).astype(int)
    # Counter resets are diagnostic candidates, not asserted physical session IDs.
    chronological=d.sort_values(['source_dataset','timestamp'])
    chronological['counter_reset_candidate']=(chronological.groupby('source_dataset').sequence.diff()<=0)
    chronological['session_candidate']=chronological.groupby('source_dataset').counter_reset_candidate.cumsum().astype(int)+1
    d['session_candidate']=chronological.session_candidate
    specs={
     'source_dataset':['source_dataset'],'scan_name':['scan_name'],
     'source_date':['source_dataset','acquisition_date'],
     'source_condition':['source_dataset','orientation','average'],
     'source_counter_reset_session_candidates':['source_dataset','session_candidate'],
     'source_spatial_blocks_10':['source_dataset','block_x_10','block_y_10'],
    }
    groups={}
    for name,cols in specs.items():
     table=group_overlap_table(d,cols);table.to_csv(OUT/f'groups_{name}.csv',index=False)
     groups[name]={'groups':len(table),'cross_split_groups':int(table.cross_split.sum()),
                   'groups_in_all_three_splits':int((table.split_count==3).sum())}
    d.groupby(['source_dataset','split','label']).size().rename('count').reset_index().to_csv(OUT/'source_split_label_counts.csv',index=False)
    proximity=spatial_train_proximity(d)
    proximity.to_csv(OUT/'spatial_train_proximity.csv',index=False)
    spatial=[]
    for split,part in proximity.groupby('split'):
     distances=part.nearest_train_xy_distance
     spatial.append({'split':split,'rows':len(part),'within_one_xy_unit':int((distances<=1+1e-12).sum()),
      'within_one_xy_unit_percent':float(100*(distances<=1+1e-12).mean()),
      'minimum_distance':float(distances.min()),'median_distance':float(distances.median()),'max_distance':float(distances.max())})
    # Verify and reuse the complete previous fresh feature extraction for an efficient
    # exact-signal duplicate screen. Identical parsed x/y signals necessarily yield
    # identical deterministic feature vectors; only vector-collision candidates need parsing.
    feature_dir=OLD/'feature_rebuild';fm=json.loads((feature_dir/'manifest.json').read_text())
    assert sha256_file(feature_dir/'raw_inputs.csv')==fm['raw_inputs_sha256']
    assert sha256_file(feature_dir/'feature_table.csv')==fm['output_csv_sha256']
    assert all(sha256_file(ROOT/p)==h for p,h in fm['source_sha256'].items())
    raw_manifest=pd.read_csv(feature_dir/'raw_inputs.csv')
    assert raw_manifest.instance_id.tolist()==a.instance_id.tolist()
    assert raw_manifest.file_path.tolist()==a.file_path.tolist()
    raw_current=[]
    for i,row in enumerate(raw_manifest.itertuples(index=False),1):
     current=sha256_file(row.file_path)
     if current!=row.sha256:raise ValueError(f'Raw data changed since full feature extraction: {row.instance_id}')
     raw_current.append(current)
     if i%10000==0:print('RAW_HASHES_VERIFIED',i,flush=True)
    d['raw_sha256']=raw_current
    raw_dups=d[d.raw_sha256.duplicated(keep=False)]
    raw_dups[['instance_id','split','file_path','raw_sha256','label']].to_csv(OUT/'byte_duplicate_candidates.csv',index=False)
    fresh=pd.read_csv(feature_dir/'feature_table.csv')
    feature_duplicate_mask=fresh[fm['feature_columns']].duplicated(keep=False)
    feature_dups=fresh.loc[feature_duplicate_mask]
    feature_dups.to_csv(OUT/'exact_feature_duplicate_candidates.csv',index=False)
    # This data has no candidate vectors; fail rather than claim unsupported absence
    # if future data requires direct numeric-signal verification of candidates.
    if len(feature_dups):
     raise ValueError('Exact feature-duplicate candidates require direct signal verification')
    raw_duplicate_groups=raw_dups.groupby('raw_sha256').agg(rows=('instance_id','size'),splits=('split','nunique'))
    features=pd.read_csv(ROOT/'results/feature_tables/feature_table.csv')
    X,y=prepare_dataset(features)
    leakage={
     'split_sizes':d.split.value_counts().to_dict(),'group_overlaps':groups,
     'source_split_counts':pd.crosstab(d.source_dataset,d.split).to_dict(),
     'duplicate_id_extra_rows':int(a.instance_id.duplicated().sum()),
     'duplicate_path_extra_rows':int(a.file_path.duplicated().sum()),
     'duplicate_source_xy_extra_rows':int(a.duplicated(['source_dataset','x','y']).sum()),
     'duplicate_source_sequence_extra_rows':int(d.duplicated(['source_dataset','sequence']).sum()),
     'byte_duplicate_rows':len(raw_dups),'byte_duplicate_groups':len(raw_duplicate_groups),
     'byte_duplicate_cross_split_groups':int((raw_duplicate_groups.splits>1).sum()),
     'fresh_exact_feature_duplicate_rows':len(feature_dups),
     'exact_numeric_xy_signal_duplicates_excluded_by_deterministic_feature_screen':True,
     'raw_files_verified_against_previous_full_rebuild':len(raw_manifest),
     'near_or_rescaled_signal_duplicates_exhaustively_checked':False,
     'spatial_proximity':spatial,'model_feature_count':len(X.columns),
     'forbidden_metadata_features':sorted(set(X.columns)&{'label','x','y','z','instance_id','file_path','filename','source_dataset'}),
     'source_label_counts':a.groupby(['source_dataset','label']).size().unstack(fill_value=0).to_dict(),
     'important_scope':'Source/scan overlap is confirmed. Bias depends on whether the target is known-scan interpolation or generalization to independent new scans/specimens.',
    }
    reconstruction['annotations_unchanged']=all(sha256_file(p)==h for p,h in annotation_hashes.items())
    checks={'historical_protected_files_unchanged':all(sha256_file(p)==h for p,h in protected.items()),
            'split_manifests_unchanged':all(sha256_file(p)==h for p,h in split_hashes.items()),
            'annotations_unchanged':reconstruction['annotations_unchanged']}
    assert all(checks.values())
    (OUT/'reconstruction.json').write_text(json.dumps(reconstruction,indent=2)+'\n')
    (OUT/'leakage.json').write_text(json.dumps(leakage,indent=2,default=str)+'\n')
    (OUT/'preservation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps({'output_dir':str(OUT),'seconds':time.monotonic()-started,'reconstruction':reconstruction,'leakage':leakage,'preservation':checks},indent=2,default=str),flush=True)

if __name__ == "__main__":
    main()
