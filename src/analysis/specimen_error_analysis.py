"""Read-only HeRAI error diagnosis, with validation/test kept separate.

No model fit, tuning, feature-table update or split change occurs. Feature shifts
and error associations are descriptive: spatial points are not independent tests.
All normalizations use training data. Zero critical votes does not imply a correct
four-class prediction. Output paths must be new.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from src.data.specimen_splits import load_membership
from src.data.split_manifest import sha256_file
from src.features.extract_extension_block1_features import SURFACE_END, SURFACE_START
from src.features.loading import load_measurement


def assemble_points(membership, catalog, predictions):
    """Validate full ID/label/group alignment before comparing model errors."""
    held = membership.loc[membership.split.ne('train')].sort_values('instance_id').copy()
    if not catalog.instance_id.is_unique:
        raise ValueError('Duplicate catalog IDs')
    held = held.merge(catalog[['instance_id', 'x', 'y', 'file_path']], on='instance_id', validate='one_to_one')
    if held[['x', 'y', 'file_path']].isna().any().any():
        raise ValueError('Missing coordinates or measurement path')
    models = list(predictions)
    if len(models) != 7:
        raise ValueError('Seven saved model prediction sets required')
    for model, by_split in predictions.items():
        for split in ['validation', 'test']:
            saved = by_split[split]
            expected = held.loc[held.split.eq(split)].set_index('instance_id')
            if saved.instance_id.isna().any() or not saved.instance_id.is_unique:
                raise ValueError('Duplicate or null prediction IDs')
            if set(saved.instance_id) != set(expected.index):
                raise ValueError('Prediction ID coverage differs')
            saved = saved.set_index('instance_id').loc[expected.index]
            if not np.array_equal(saved.label, expected.label):
                raise ValueError('Prediction labels differ from frozen membership')
            for key in ['specimen_id', 'source_dataset']:
                if not saved[key].equals(expected[key]):
                    raise ValueError('Prediction group metadata differs')
            if not saved.prediction.isin([0, 1, 2, 3]).all():
                raise ValueError('Invalid predicted class')
            values = saved.prediction.to_dict()
            mask = held.split.eq(split)
            held.loc[mask, model] = held.loc[mask, 'instance_id'].map(values)
    held[models] = held[models].astype(int)
    critical = held.label.isin([1, 2])
    held['critical_votes'] = held[models].eq(3).sum(axis=1).where(critical, 0)
    held['critical_truth'] = critical
    held['all7_critical'] = critical & held.critical_votes.eq(7)
    held['any_critical'] = critical & held.critical_votes.gt(0)
    return held, models


def grouped_errors(points, models, keys):
    """Counts are model-error events; denominators explicitly include seven models."""
    rows = []
    for values, group in points.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        eligible = group.loc[group.critical_truth]
        denominator = len(eligible)
        row = dict(zip(keys, values))
        row.update(points=len(group), critical_support=denominator,
                   any_critical=int(eligible.any_critical.sum()), all7_critical=int(eligible.all7_critical.sum()),
                   model_error_events=int(eligible.critical_votes.sum()),
                   mean_model_critical_rate=100 * eligible.critical_votes.sum() / (len(models) * denominator) if denominator else np.nan)
        for model in models:
            n = int(eligible[model].eq(3).sum())
            row[model + '_errors'] = n
            row[model + '_rate'] = 100 * n / denominator if denominator else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def overlap_table(points, models):
    """Jaccard compares error sets, not agreement dominated by true negatives."""
    rows = []
    for split, group in points.loc[points.critical_truth].groupby('split'):
        for a in models:
            for b in models:
                left, right = group[a].eq(3), group[b].eq(3)
                shared, union = int((left & right).sum()), int((left | right).sum())
                rows.append(dict(split=split, model_a=a, model_b=b, shared=shared, union=union,
                                 only_a=int((left & ~right).sum()), only_b=int((right & ~left).sum()),
                                 jaccard=shared / union if union else np.nan))
    return pd.DataFrame(rows)


def distribution_shift(features, membership, columns):
    """Wasserstein / training IQR, also conditional on class; no pointwise p-values."""
    data = membership.merge(features[['instance_id', *columns]], on='instance_id', validate='one_to_one')
    rows = []
    for label in ['all', 0, 1, 2, 3]:
        population = data if label == 'all' else data.loc[data.label.eq(label)]
        train = population.loc[population.split.eq('train')]
        for feature in columns:
            values = train[feature].to_numpy()
            q01, q25, q75, q99 = np.quantile(values, [.01, .25, .75, .99])
            scale = q75 - q25
            for split in ['validation', 'test']:
                held = population.loc[population.split.eq(split), feature].to_numpy()
                if not len(held):
                    continue
                rows.append(dict(split=split, true_class=str(label), feature=feature,
                                 train_n=len(values), held_n=len(held),
                                 w1_train_iqr=wasserstein_distance(values, held) / scale if scale > 0 else np.nan,
                                 outside_train_01_99_pct=100 * np.mean((held < q01) | (held > q99)),
                                 train_median=np.median(values), held_median=np.median(held)))
    return pd.DataFrame(rows)


def error_associations(points, features, membership, columns, min_n=5):
    """Same specimen AND true class controls; equally weight strata, not pixels.

    Contrast all-seven critical cases against zero-critical-vote cases. The latter
    may still have other classification errors. Scale is the true-class training
    IQR; no model importance, causal interpretation or significance is implied.
    """
    data = points.merge(features[['instance_id', *columns]], on='instance_id', validate='one_to_one')
    training = membership.loc[membership.split.eq('train')].merge(features[['instance_id', *columns]], on='instance_id')
    rows = []
    for (split, specimen, label), group in data.loc[data.critical_truth].groupby(['split', 'specimen_id', 'label']):
        miss = group.loc[group.all7_critical]
        control = group.loc[group.critical_votes.eq(0)]
        if len(miss) < min_n or len(control) < min_n:
            continue
        reference = training.loc[training.label.eq(label), columns]
        scale = reference.quantile(.75) - reference.quantile(.25)
        effect = (miss[columns].median() - control[columns].median()) / scale.replace(0, np.nan)
        for feature in columns:
            rows.append(dict(split=split, specimen_id=specimen, true_class=int(label), feature=feature,
                             miss_n=len(miss), control_n=len(control), miss_median=miss[feature].median(),
                             control_median=control[feature].median(), delta_train_class_iqr=effect[feature]))
    return pd.DataFrame(rows)


def signal_diagnostics(file_path):
    """Raw-signal audit only: existing surface/noise windows and omitted late energy."""
    signal = load_measurement(file_path)
    x, y = signal.x.to_numpy(), signal.y.to_numpy()
    region = np.flatnonzero((x >= SURFACE_START) & (x <= SURFACE_END))
    surface = region[np.argmax(np.abs((y - np.median(y))[region]))]
    t = x - x[surface]
    noise_mask = (t >= 70) & (t <= 80)
    baseline = np.mean(y[noise_mask])
    centered = y - baseline
    post, late = (t >= 3), (t > 80)
    pre = (t >= -15) & (t <= -5)
    noise = np.std(y[noise_mask])
    early_noise = np.std(y[pre]) if pre.sum() >= 5 else np.nan
    denominator = np.sum(centered[post] ** 2)
    diagnostics = dict(surface_time=x[surface], surface_abs_amp=abs(centered[surface]),
                       noise_std_70_80=noise, pre_noise_std=early_noise,
                       noise_ratio_late_pre=noise / early_noise if early_noise > 0 else np.nan,
                       last_relative_x=t.max(), late_sample_count=int(late.sum()),
                       late_energy_fraction=np.sum(centered[late] ** 2) / denominator if denominator > 0 else np.nan)
    return diagnostics, t, centered


def plots(points, models, body, overlap, shifts, associations, features, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    short = ['LR', 'RF', 'HGB', 'XGB', 'LGBM', 'ET', 'MLP']
    plt.rcParams.update({'font.size': 9})
    for split in ['validation', 'test']:
        selected = body.loc[body.split.eq(split)].set_index('specimen_id').sort_index()
        rates = selected[[model + '_rate' for model in models]].to_numpy()
        counts = selected[[model + '_errors' for model in models]].to_numpy()
        fig, ax = plt.subplots(figsize=(11, 3 + len(selected)*.35), layout='constrained')
        image = ax.imshow(rates, vmin=0, vmax=100, cmap='YlOrRd', aspect='auto')
        for i in range(len(selected)):
            for j in range(7):
                ax.text(j, i, f'{counts[i,j]:.0f}\n{rates[i,j]:.1f}%', ha='center', va='center', fontsize=8,
                        color='white' if rates[i,j] > 65 else 'black')
        ax.set(xticks=range(7), xticklabels=short, yticks=range(len(selected)),
               yticklabels=[f'{sid} (n1/2={int(n)})' for sid,n in zip(selected.index, selected.critical_support)],
               title=f'{split}: kritische Fehler pro Körper (Anzahl / Rate)')
        fig.colorbar(image, ax=ax, label='% der wahren Klassen 1/2')
        fig.savefig(output / f'{split}_specimen_errors.png', dpi=170); plt.close(fig)
        group = points.loc[points.split.eq(split)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
        for j, source in enumerate(sorted(points.source_dataset.unique())):
            g = group.loc[group.source_dataset.eq(source)]
            for label, color in [(0, 'gray'), (1, 'red'), (2, 'orange'), (3, 'green')]:
                a = g.loc[g.label.eq(label)]
                axes[0,j].scatter(a.x, a.y, s=5, color=color, label=str(label))
            axes[0,j].set_title(source + ': Originalklasse')
            axes[1,j].scatter(g.x, g.y, s=3, color='lightgray')
            a = g.loc[g.critical_truth]
            image = axes[1,j].scatter(a.x, a.y, s=7, c=a.critical_votes, cmap='viridis',
                                       norm=BoundaryNorm(np.arange(-.5, 8.5), 256))
            axes[1,j].set_title('Anzahl Modelle mit kritischer Freigabe')
            for sid, a in g.groupby('specimen_id'):
                for ax in axes[:,j]:
                    ax.text(a.x.mean(), a.y.mean(), sid.split('_pk_')[-1], ha='center',
                            bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
            for ax in axes[:,j]:
                ax.set(xlabel='Mess-X', ylabel='Mess-Y', aspect='equal')
        axes[0,0].legend(title='Klasse', loc='best')
        fig.colorbar(image, ax=axes[1,:], ticks=range(8), label='Kritische Stimmen (nur wahre 1/2)')
        fig.suptitle(split + ': räumliche Fehlerverteilung; Nummern = pk-ID innerhalb Messreihe')
        fig.savefig(output / f'{split}_spatial.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for ax, split in zip(axes, ['validation', 'test']):
        matrix = overlap.loc[overlap.split.eq(split)].pivot(index='model_a', columns='model_b', values='jaccard').loc[models,models].to_numpy()
        image=ax.imshow(matrix, vmin=0, vmax=1, cmap='Blues')
        for i in range(7):
            for j in range(7): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>.65 else 'black')
        ax.set(xticks=range(7), xticklabels=short, yticks=range(7), yticklabels=short, title=split)
    fig.colorbar(image,ax=axes,label='Jaccard der kritischen Fehlerpunkte')
    fig.savefig(output/'model_overlap.png',dpi=170);plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,split in zip(axes,['validation','test']):
        for label,color,offset in [(1,'tomato',-.2),(2,'goldenrod',.2)]:
            a=points.loc[points.split.eq(split)&points.label.eq(label)]
            hist=a.critical_votes.value_counts().reindex(range(8),fill_value=0)
            ax.bar(np.arange(8)+offset,100*hist/len(a),width=.4,color=color,label=f'Klasse {label}, n={len(a)}')
        ax.set(xticks=range(8),xlabel='Anzahl kritischer Modellstimmen',ylabel='% innerhalb wahrer Klasse',title=split);ax.legend()
    fig.savefig(output/'consensus_by_class.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    for ax,label in zip(axes,['all','1','2']):
        a=shifts.loc[shifts.true_class.eq(label)].pivot(index='feature',columns='split',values='w1_train_iqr')
        a=a.loc[a.max(axis=1).nlargest(10).index].sort_values('validation')
        a[['validation','test']].plot.barh(ax=ax)
        ax.set(title='Alle Klassen' if label=='all' else 'Wahre Klasse '+label,xlabel='Wasserstein / Trainings-IQR')
    fig.savefig(output/'feature_shift.png',dpi=170);plt.close(fig)
    # Four deterministic, same-body/same-class illustrative pairs from validation.
    strata=associations.loc[associations.split.eq('validation')].drop_duplicates(['specimen_id','true_class']).sort_values(['true_class','specimen_id'])
    merged=points.merge(features,on='instance_id',suffixes=('','_feature'))
    chosen=[]
    fig,axes=plt.subplots(2,2,figsize=(14,8),layout='constrained')
    for ax,row in zip(axes.ravel(),strata.head(4).itertuples()):
        group=merged.loc[merged.split.eq('validation')&merged.specimen_id.eq(row.specimen_id)&merged.label.eq(row.true_class)]
        subset=['second_peak_snr','second_peak_to_surface_ratio','post_peak_std','second_peak_surface_corr']
        center=group[subset].median();scale=(group[subset].quantile(.75)-group[subset].quantile(.25)).replace(0,1)
        for mask,color,tag in [(group.all7_critical,'crimson','7/7 kritisch'),(group.critical_votes.eq(0),'navy','0/7 kritisch')]:
            candidates=group.loc[mask].sort_values('instance_id')
            distance=((candidates[subset]-center)/scale).abs().sum(axis=1)
            sample=candidates.loc[distance.idxmin()]
            _,t,y=signal_diagnostics(sample.file_path)
            ax.plot(t,y/max(np.max(np.abs(y)),1e-15),color=color,label=tag+' '+sample.instance_id,linewidth=1)
            chosen.append({'instance_id':sample.instance_id,'specimen_id':sample.specimen_id,'label':int(sample.label),'group':tag})
        ax.axvspan(3,80,color='gray',alpha=.1);ax.axvspan(70,80,color='orange',alpha=.15)
        ax.set(title=f'{row.specimen_id}, wahre Klasse {row.true_class}',xlabel='Signal-X relativ zum Oberflächenpeak',ylabel='Amplitude / max. |Amplitude|')
        ax.legend(fontsize=7)
    fig.savefig(output/'validation_signal_pairs.png',dpi=170);plt.close(fig)
    pd.DataFrame(chosen).to_csv(output/'illustrative_signal_ids.csv',index=False)


def run_analysis(run_dir, split_dir, catalog_path, feature_dir, output_dir):
    run_dir,split_dir,feature_dir,output_dir=map(Path,[run_dir,split_dir,feature_dir,output_dir])
    if output_dir.exists(): raise FileExistsError(output_dir)
    paths=[p for p in run_dir.rglob('*') if p.is_file()]
    paths += [p for p in feature_dir.iterdir() if p.is_file()]
    paths += [p for p in split_dir.iterdir() if p.is_file()]+[Path(catalog_path)]
    before={str(p):sha256_file(p) for p in paths}
    membership=load_membership(split_dir)
    columns=json.loads((feature_dir/'manifest.json').read_text())['selected_features']
    if len(columns)!=52: raise ValueError('Expected the most recent manual52 models')
    features=pd.read_csv(feature_dir/'features.csv')
    if not features.instance_id.is_unique or set(features.instance_id)!=set(membership.instance_id): raise ValueError('Feature coverage differs')
    models=pd.read_csv(run_dir/'results.csv').model.unique().tolist()
    predictions={model:{split:pd.read_csv(run_dir/'evaluations'/model/split/'predictions.csv') for split in ['validation','test']} for model in models}
    points,models=assemble_points(membership,pd.read_csv(catalog_path),predictions)
    output_dir.mkdir(parents=True,exist_ok=False)
    points.to_csv(output_dir/'point_votes.csv',index=False)
    body=grouped_errors(points,models,['split','source_dataset','specimen_id'])
    body.to_csv(output_dir/'specimen_errors.csv',index=False)
    grouped_errors(points,models,['split','source_dataset']).to_csv(output_dir/'source_errors.csv',index=False)
    grouped_errors(points,models,['split','label']).to_csv(output_dir/'class_errors.csv',index=False)
    grouped_errors(points,models,['split','source_dataset','specimen_id','label']).to_csv(output_dir/'specimen_class_errors.csv',index=False)
    overlap=overlap_table(points,models);overlap.to_csv(output_dir/'model_overlap.csv',index=False)
    shifts=distribution_shift(features,membership,columns);shifts.to_csv(output_dir/'feature_shift.csv',index=False)
    associations=error_associations(points,features,membership,columns)
    associations.to_csv(output_dir/'within_specimen_class_feature_associations.csv',index=False)
    associations.groupby(['split','true_class','feature']).agg(strata=('specimen_id','nunique'),
        median_delta_train_class_iqr=('delta_train_class_iqr','median'),
        min_delta=('delta_train_class_iqr','min'),max_delta=('delta_train_class_iqr','max')).to_csv(output_dir/'feature_association_summary.csv')
    membership.groupby(['split','source_dataset','label']).size().rename('points').to_csv(output_dir/'class_priors.csv')
    rows=[];raw_hashes={}
    for i,row in enumerate(points.loc[points.critical_truth].itertuples(index=False),1):
        path=row.file_path;raw_hashes[path]=sha256_file(path)
        d,_,_=signal_diagnostics(path)
        rows.append({'instance_id':row.instance_id,'split':row.split,'specimen_id':row.specimen_id,'label':row.label,
                     'critical_votes':row.critical_votes,**d})
        if i%1000==0: print(f'Raw-signal audit {i}/4092',flush=True)
    raw=pd.DataFrame(rows);raw.to_csv(output_dir/'raw_signal_diagnostics.csv',index=False)
    raw.groupby(['split','specimen_id','label','critical_votes']).median(numeric_only=True).to_csv(output_dir/'raw_signal_group_medians.csv')
    plots(points,models,body,overlap,shifts,associations,features,output_dir)
    if any(sha256_file(p)!=h for p,h in before.items()) or any(sha256_file(p)!=h for p,h in raw_hashes.items()):
        raise ValueError('Input changed during read-only analysis')
    manifest={'models':models,'feature_set':'manual52','no_model_training':True,'validation':'primary diagnosis',
              'test':'exploratory diagnosis only; not selection or tuning', 'input_sha256':before,
              'raw_signal_sha256':raw_hashes,'source_sha256':sha256_file(__file__),
              'input_files_unchanged':len(before),'critical_signal_audits':len(raw),
              'methods':'Descriptive counts, Jaccard, train-IQR-normalized Wasserstein, within-body/class median contrasts; no pointwise significance tests',
              'output_sha256':{p.name:sha256_file(p) for p in output_dir.iterdir() if p.is_file()}}
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Read-only error analysis complete:',output_dir,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['run-dir','split-dir','catalog','feature-dir','output-dir']: parser.add_argument('--'+key,required=True,type=Path)
    args=parser.parse_args()
    run_analysis(args.run_dir,args.split_dir,args.catalog,args.feature_dir,args.output_dir)


if __name__=='__main__': main()
