"""Read-only signal/annotation review; candidates are NOT physical ground truth.

No feature extractor is changed or called, no ML feature matrix is generated,
and no model is fitted or inferred. Alternative markers/windows are diagnostic
counterfactuals only. Correctness uses original labels, not absence of critical errors.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split_manifest import sha256_file
from src.features.loading import load_measurement

MODELS=['logistic_regression','random_forest','hgb','xgboost','lightgbm','extra_trees','mlp']


def inspect_wave(x,y):
    """Reproduce the existing marker; audit two broad-window markers and ROI overlap."""
    centered=y-np.median(y)
    narrow=np.flatnonzero((x>=980)&(x<=1010));wide=np.flatnonzero((x>=960)&(x<=1030))
    if not len(narrow) or len(wide)<3: raise ValueError('Signal does not cover diagnostic search regions')
    current=narrow[np.argmax(abs(centered[narrow]))]
    candidate=wide[np.argmax(abs(centered[wide]))]
    gradient=np.gradient(y,x);slope=wide[np.argmax(abs(gradient[wide]))]
    a,b=x[current],x[candidate]
    noise=(x>=a+70)&(x<=a+80);alternate=(x>=b+70)&(x<=b+80)
    if noise.sum()<4 or alternate.sum()<4: raise ValueError('Noise audit window missing')
    midpoint=a+75;first=noise&(x<midpoint);second=noise&(x>=midpoint)
    std=np.std(y[noise]);half_change=abs(y[first].mean()-y[second].mean())
    return dict(current_x=a,broad_extremum_x=b,slope_marker_x=x[slope],delta_x=a-b,
                current_at_boundary=min(a-980,1010-a)<=.1,
                roi_overlap_fraction=max(0.,77-abs(a-b))/77,
                current_noise_mean=y[noise].mean(),alternative_noise_mean=y[alternate].mean(),
                current_noise_std=std,alternative_noise_std=np.std(y[alternate]),
                half_window_mean_change_over_std=half_change/std if std>0 else np.nan)


def annotation_check(catalog):
    """Check catalog assignments against original coordinate lists without relabeling."""
    files={'messreihe_1':'hohltestk1_tr_markArea_XYList2.txt',
           'messreihe_2':'hohltestk2_tr_markArea_XYList.txt','messreihe_3':'hohltestk3_tr_markArea_XYList.txt'}
    rows=[];paths=[]
    for source,g in catalog.groupby('source_dataset'):
        path=Path(g.file_path.iloc[0]).parent.parent/files[source];paths.append(path)
        original=pd.read_csv(path,sep=r'\s+',comment='#',names=['x','y','original_label'])
        if original.duplicated(['x','y']).any(): raise ValueError('Ambiguous original annotation')
        matched=g.merge(original,on=['x','y'],how='left',validate='one_to_one')
        if len(original)!=len(g) or matched.original_label.isna().any(): raise ValueError('Annotation coverage differs')
        rows.append(dict(source_dataset=source,points=len(g),label_mismatches=int(matched.label.ne(matched.original_label).sum())))
    return pd.DataFrame(rows),paths


def run_review(output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    prior=Path('results/analysis/herai_error_analysis_20261110')
    root=Path('results/experiments/specimen_manual52_v1')
    inputs=[p for folder in [root,prior,Path('configs/specimen_holdout_v1/splits')] for p in folder.rglob('*') if p.is_file()]
    catalog_path=Path('data/measurements/combined_dataset.csv');mapping_path=Path('data/metadata_reference_excluded/specimen_mapping.csv')
    catalog=pd.read_csv(catalog_path);mapping=pd.read_csv(mapping_path)
    annotations,annotation_paths=annotation_check(catalog)
    inputs += [catalog_path,mapping_path,*annotation_paths,Path('src/features/extract_extension_block1_features.py')]
    before={str(p):sha256_file(p) for p in inputs}
    points=pd.read_csv(prior/'point_votes.csv')
    points['correct_votes']=points[MODELS].eq(points.label,axis=0).sum(axis=1)
    points['outcome']=np.select([points.critical_votes.ge(4),points.correct_votes.ge(4)],['critical_majority','correct_majority'],default='other')
    target=points.loc[points.label.isin([1,2])].copy()
    refs=catalog.merge(mapping[['instance_id','measurement_role']],on='instance_id').query("measurement_role=='reference'")
    reference_sample=pd.concat([g.sort_values('instance_id').iloc[np.linspace(0,len(g)-1,25,dtype=int)] for _,g in refs.groupby('source_dataset')])
    output.mkdir(parents=True,exist_ok=False)
    annotations.to_csv(output/'annotation_consistency.csv',index=False)
    rows=[];raw_hashes={};reference_waves={}
    for role,frame in [('heldout',target),('reference',reference_sample)]:
        for i,r in enumerate(frame.itertuples(index=False),1):
            raw_hashes[r.file_path]=sha256_file(r.file_path)
            w=load_measurement(r.file_path);x,y=w.x.to_numpy(),w.y.to_numpy();d=inspect_wave(x,y)
            rows.append(dict(instance_id=r.instance_id,role=role,source_dataset=r.source_dataset,
                             label=r.label,split=r.split if role=='heldout' else 'reference_only',
                             specimen_id=r.specimen_id if role=='heldout' else 'metal_reference',
                             outcome=r.outcome if role=='heldout' else 'reference_only',**d))
            if role=='reference': reference_waves.setdefault(r.source_dataset,[]).append((x,y))
            if i%1000==0: print(f'Anchor review {role}: {i}/{len(frame)}',flush=True)
    diagnostics=pd.DataFrame(rows);diagnostics.to_csv(output/'anchor_window_diagnostics.csv',index=False)
    diagnostics.groupby(['role','split','source_dataset','label','outcome']).median(numeric_only=True).to_csv(output/'diagnostic_medians.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(source,waves) in zip(axes,reference_waves.items()):
        grid=np.linspace(960,1040,1601);values=[]
        for x,y in waves:
            z=y-np.median(y);values.append(np.interp(grid,x,z)/max(abs(z).max(),1e-15))
        values=np.array(values);ax.plot(grid,np.median(values,axis=0),color='black',label='Median 25 Metallmessungen')
        ax.fill_between(grid,*np.quantile(values,[.1,.9],axis=0),alpha=.2)
        ax.axvspan(980,1010,color='orange',alpha=.2,label='bestehendes Suchfenster')
        ax.set(title=source,xlabel='Aufgezeichnetes Signal-X',ylabel='Medianzentrierte Amplitude / max. |Amplitude|',xlim=(965,1030));ax.legend(fontsize=7)
    fig.savefig(output/'metal_reference_pulses.png',dpi=170);plt.close(fig)
    # Deterministic representative pairs; actual correct majority, not zero critical votes.
    focuses=[('validation','messreihe_1_pk_07',1),('validation','messreihe_1_pk_11',2),
             ('validation','messreihe_2_pk_02',2),('validation','messreihe_3_pk_04',1),
             ('test','messreihe_1_pk_10',2),('test','messreihe_2_pk_14',1)]
    selected=[];fig,axes=plt.subplots(3,2,figsize=(15,11),layout='constrained')
    for ax,(split,body,label) in zip(axes.ravel(),focuses):
        g=target.loc[target.split.eq(split)&target.specimen_id.eq(body)&target.label.eq(label)]
        for outcome,color in [('critical_majority','crimson'),('correct_majority','navy')]:
            candidates=g.loc[g.outcome.eq(outcome)].sort_values('instance_id')
            if candidates.empty: continue
            # Median-nearest current anchor; ties by existing ID, no outcome-score tuning.
            anchors=diagnostics.set_index('instance_id').loc[candidates.instance_id,'current_x'].to_numpy()
            r=candidates.iloc[np.argmin(abs(anchors-np.median(anchors)))]
            w=load_measurement(r.file_path);x,y=w.x.to_numpy(),w.y.to_numpy();d=inspect_wave(x,y)
            z=y-np.median(y);ax.plot(x,z/max(abs(z).max(),1e-15),color=color,lw=1,label=f'{outcome}, korrekt {r.correct_votes}/7, kritisch {r.critical_votes}/7')
            ax.axvline(d['current_x'],color=color,ls='--');ax.plot(d['broad_extremum_x'],np.interp(d['broad_extremum_x'],x,z)/max(abs(z).max(),1e-15),marker='o',color=color,ms=4)
            selected.append(dict(instance_id=r.instance_id,split=split,specimen_id=body,label=label,outcome=outcome,correct_votes=r.correct_votes,critical_votes=r.critical_votes,candidate_n=len(candidates)))
        ax.axvspan(980,1010,color='gray',alpha=.1);ax.set(title=f'{split}: {body}, Klasse {label}',xlim=(965,1040),xlabel='Signal-X',ylabel='Medianzentriert / max. |Amplitude|');ax.legend(fontsize=7)
    fig.savefig(output/'representative_raw_pairs.png',dpi=170);plt.close(fig)
    pd.DataFrame(selected).to_csv(output/'representative_ids.csv',index=False)
    # B-scans on fixed, informative spatial rows with original annotations/predictions.
    for body in ['messreihe_1_pk_07','messreihe_1_pk_11','messreihe_2_pk_02']:
        g=points.loc[points.specimen_id.eq(body)]
        scores=g.groupby('y').apply(lambda a:min(sum(a.outcome.eq('critical_majority')),sum(a.outcome.eq('correct_majority'))),include_groups=False)
        yrow=scores.sort_index().idxmax();line=g[g.y.eq(yrow)].sort_values('x')
        grid=np.linspace(965,1050,1701);matrix=[];anchors=[];broad=[]
        for r in line.itertuples(index=False):
            w=load_measurement(r.file_path);x,y=w.x.to_numpy(),w.y.to_numpy();z=y-np.median(y);d=inspect_wave(x,y)
            matrix.append(np.interp(grid,x,z)/max(abs(z).max(),1e-15));anchors.append(d['current_x']);broad.append(d['broad_extremum_x']);raw_hashes[r.file_path]=sha256_file(r.file_path)
        fig,axes=plt.subplots(2,1,figsize=(12,6),height_ratios=[5,1],layout='constrained',sharex=True)
        image=axes[0].pcolormesh(line.x,grid,np.array(matrix).T,cmap='RdBu_r',vmin=-1,vmax=1,shading='auto')
        axes[0].plot(line.x,anchors,color='lime',label='bestehender Anker');axes[0].plot(line.x,broad,color='black',ls=':',label='breiter Extremum-Marker (nicht verifiziert)')
        axes[0].axhspan(980,1010,alpha=.1,color='gray');axes[0].set(ylabel='Signal-X',title=f'{body}, räumliche Zeile Y={yrow}; pro Signal normiert');axes[0].legend(fontsize=8)
        axes[1].scatter(line.x,line.label,c=line.critical_votes,cmap='viridis',vmin=0,vmax=7,s=35)
        axes[1].set(yticks=range(4),ylabel='Originalklasse',xlabel='Mess-X (nur Diagnose)');fig.colorbar(image,ax=axes[0],label='Normierte Amplitude')
        fig.savefig(output/f'transect_{body}.png',dpi=170);plt.close(fig)
        line[['instance_id','x','y','label','correct_votes','critical_votes']].assign(current_anchor=anchors,broad_marker=broad).to_csv(output/f'transect_{body}.csv',index=False)
    # One concrete sensitivity example: original vs diagnostic window locations.
    d=diagnostics.query("split=='validation' and specimen_id=='messreihe_1_pk_07' and label==1 and outcome=='critical_majority'").sort_values('instance_id')
    r=d.iloc[np.argmin(abs(d.current_x-d.current_x.median()))];path=target.set_index('instance_id').loc[r.instance_id,'file_path'];w=load_measurement(path)
    fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained')
    for ax,a,title in [(axes[0],r.current_x,'Bestehender Anker'),(axes[1],r.broad_extremum_x,'Alternativer Marker: nur Sensitivität, nicht physikalisch bestätigt')]:
        ax.plot(w.x,w.y,color='black',lw=1);ax.axvline(a,color='red');ax.axvspan(a+3,a+80,color='steelblue',alpha=.18,label='Post-Peak-Fenster 3–80');ax.axvspan(a+70,a+80,color='orange',alpha=.35,label='Rauschfenster 70–80')
        ax.set(title=title+f', X={a:.2f}',xlabel='Signal-X',ylabel='Rohamplitude');ax.legend()
    fig.suptitle(r.instance_id);fig.savefig(output/'window_sensitivity.png',dpi=170);plt.close(fig)
    if any(sha256_file(p)!=h for p,h in before.items()) or any(sha256_file(p)!=h for p,h in raw_hashes.items()): raise ValueError('Input changed')
    manifest=dict(no_feature_implementation=True,no_training=True,no_inference=True,raw_signal_count=len(raw_hashes),
                  input_files_unchanged=len(before),input_sha256=before,raw_sha256=raw_hashes,source_sha256=sha256_file(__file__),
                  interpretation='All alternative markers, normalization views and windows are diagnostic, not physical ground truth or a new extractor.',
                  output_sha256={p.name:sha256_file(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Review complete:',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    run_review(parser.parse_args().output_dir)
