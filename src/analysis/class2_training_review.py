"""Descriptive OOF review of the fixed class-2 ablation; no old holdout access.

The auxiliary mixtures are density diagnostics, not relabeling models. Their
transforms and density fits exclude each displayed body's entire CV fold.
"""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.data.split_manifest import sha256_file
from src.experiments.class2_group_cv import KNOWN, load_inputs, training_indices
from src.features.experimental_anchors import early_index
from src.features.loading import load_measurement


def positions(data):
    xy = data.instance_id.str.extract(r'_x(?P<x>-?\d+(?:\.\d+)?)_y(?P<y>-?\d+(?:\.\d+)?)$').astype(float)
    if xy.isna().any().any():
        raise ValueError('Unknown coordinate ID format')
    return xy


def neighbor_context(data):
    """Local annotation context only; neighbors do not certify hidden class-2 truth."""
    records = []
    for body, group in data.groupby('specimen_id'):
        known = group[group.label.isin([1, 3])]
        uncertain = group[group.label == 2]
        if not len(uncertain):
            continue
        tree = cKDTree(known[['x', 'y']]) if len(known) else None
        for row in uncertain.itertuples():
            fraction = np.nan
            if tree is not None:
                _, ix = tree.query([row.x, row.y], k=min(8, len(known)))
                fraction = float(np.mean(known.label.to_numpy()[np.atleast_1d(ix)] == 1))
            distances = {}
            for label in [1, 3]:
                part = known[known.label == label]
                distances[f'distance_to_{label}'] = float(cKDTree(part[['x', 'y']]).query([row.x, row.y])[0]) if len(part) else np.nan
            records.append(dict(instance_id=row.instance_id, neighbor_cavity_fraction=fraction, **distances))
    return pd.DataFrame(records)


def mixture_diagnostics(output, data, X):
    """Fixed 5-PC diagonal GMM diagnostic, fit on training-fold class 2 only."""
    all_rows, summaries, body_scores, components = [], [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {1: '#c44e52', 2: '#8172b3', 3: '#4c72b0'}
    for fold in range(3):
        known = training_indices(data, fold, KNOWN)
        train2 = np.flatnonzero((data.fold != fold) & (data.label == 2))
        val2 = np.flatnonzero((data.fold == fold) & (data.label == 2))
        va = np.flatnonzero(data.fold == fold)
        transform = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler(), PCA(n_components=5, svd_solver='full'))
        with threadpool_limits(limits=1):
            transform.fit(X[known])
            ztrain = transform.transform(X[train2])
            zval = transform.transform(X[val2])
            gmms = [GaussianMixture(n_components=k, covariance_type='diag', n_init=5, random_state=42, reg_covar=1e-4).fit(ztrain) for k in [1, 2]]
        one, two = gmms
        difference = two.score_samples(zval) - one.score_samples(zval)
        # Ordering is stable within this fold only; PCs/components have no physical identity.
        order = np.argsort(two.means_[:, 0])
        posterior = two.predict_proba(zval)[:, order]
        cluster = posterior.argmax(axis=1)
        for i, ix in enumerate(val2):
            all_rows.append(dict(instance_id=data.instance_id.iloc[ix], fold=fold,
                                 pc1=zval[i, 0], pc2=zval[i, 1], component=int(cluster[i]),
                                 component_max_probability=float(posterior[i].max()),
                                 log_density_gain=float(difference[i])))
        summary = dict(fold=fold, training_class2=len(train2), validation_class2=len(val2),
                       log_density_gain_mean=float(difference.mean()), log_density_gain_median=float(np.median(difference)),
                       bic_train_1=float(one.bic(ztrain)), bic_train_2=float(two.bic(ztrain)),
                       weight_component0=float(two.weights_[order[0]]), weight_component1=float(two.weights_[order[1]]),
                       pca_explained_variance=float(transform[-1].explained_variance_ratio_.sum()))
        summaries.append(summary)
        for component, original in enumerate(order):
            row = dict(fold=fold, component=component, train_weight=float(two.weights_[original]),
                       diagonal_variance_trace=float(two.covariances_[original].sum()))
            for label in [1, 3]:
                reference = transform.transform(X[known[data.label.iloc[known].to_numpy() == label]])
                row[f'distance_to_train_class{label}_median'] = float(np.linalg.norm(two.means_[original] - np.median(reference, axis=0)))
            components.append(row)
        for body in sorted(data.specimen_id.iloc[val2].unique()):
            selected = data.specimen_id.iloc[val2].to_numpy() == body
            body_scores.append(dict(fold=fold, specimen_id=body, points=int(selected.sum()),
                                    mean_log_density_gain=float(difference[selected].mean()),
                                    component0_fraction=float(np.mean(cluster[selected] == 0))))
        joblib.dump(dict(transform=transform, one=one, two=two, component_order=order,
                         transform_train_ids=data.instance_id.iloc[known].tolist(),
                         mixture_train_ids=data.instance_id.iloc[train2].tolist()), output / f'mixture_fold{fold}.joblib')
        z = transform.transform(X[va])
        for label in [3, 1, 2]:
            sel = data.label.iloc[va].to_numpy() == label
            axes[fold].scatter(z[sel, 0], z[sel, 1], s=3, alpha=.25 if label != 2 else .7, c=colors[label], label=f'Klasse {label}', rasterized=True)
        # Fixed quantile view clips the picture only; no density/evaluation observations are dropped.
        axes[fold].set_xlim(*np.quantile(z[:, 0], [.005, .995]))
        axes[fold].set_ylim(*np.quantile(z[:, 1], [.005, .995]))
        axes[fold].set(title=f'OOF-Fold {fold}: trainbasierte PCA', xlabel='PC1', ylabel='PC2')
    axes[0].legend(markerscale=3)
    fig.tight_layout(); fig.savefig(output / 'class2_feature_space.png', dpi=160); plt.close(fig)
    pd.DataFrame(all_rows).to_csv(output / 'class2_mixture_oof.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'mixture_fold_summary.csv', index=False)
    pd.DataFrame(body_scores).to_csv(output / 'mixture_body_summary.csv', index=False)
    pd.DataFrame(components).to_csv(output / 'mixture_component_summary.csv', index=False)


def representative_waves(output, context, source):
    """Deterministic within-body shape illustration, explicitly prediction-selected."""
    counts = context.groupby('specimen_id')[['votes_1', 'votes_3']].agg(lambda x: int((x == 7).sum()))
    counts['paired'] = counts.min(axis=1)
    bodies = counts[counts.paired >= 2].sort_values(['paired', 'specimen_id'], ascending=[False, True]).head(3).index
    if not len(bodies):
        return
    hashes = json.loads((source / 'raw_sha256.json').read_text())
    fig, axes = plt.subplots(1, len(bodies), figsize=(5 * len(bodies), 4), squeeze=False)
    records = []
    grid = np.linspace(-10, 80, 901)
    for ax, body in zip(axes.flat, bodies):
        for label, color in [(1, '#c44e52'), (3, '#4c72b0')]:
            candidates = context[(context.specimen_id == body) & (context[f'votes_{label}'] == 7)].sort_values('instance_id')
            selected = candidates.iloc[np.unique(np.linspace(0, len(candidates) - 1, min(12, len(candidates)), dtype=int))]
            waves = []
            for row in selected.itertuples():
                if sha256_file(row.file_path) != hashes[row.file_path]:
                    raise ValueError('Raw training waveform changed')
                wave = load_measurement(row.file_path)
                x, y = wave.x.to_numpy(), wave.y.to_numpy()
                marker = early_index(x, y)
                relative = x - x[marker]
                centered = y - np.median(y)
                norm = np.max(abs(centered[(relative >= -10) & (relative <= 80)]))
                aligned = np.interp(grid, relative, centered) / norm
                waves.append(aligned)
                ax.plot(grid, aligned, color=color, alpha=.10, lw=.5)
                records.append(dict(instance_id=row.instance_id, specimen_id=body, prediction_consensus=label,
                                    anchor_x=float(x[marker]), normalization=float(norm), raw_sha256=hashes[row.file_path]))
            ax.plot(grid, np.median(waves, axis=0), color=color, label=f'Alle 7 → {label}, n={len(waves)}')
        ax.set(title=body, xlabel='Signal-X relativ zum frühen Anker', ylabel='Medianzentrierte normierte Amplitude', xlim=(3, 80))
        ax.legend(fontsize=8)
    fig.suptitle('Wahre Klasse 2: nach Modellkonsens ausgewählte Beispiele; keine Ground Truth')
    fig.tight_layout(); fig.savefig(output / 'class2_example_waves.png', dpi=170); plt.close(fig)
    pd.DataFrame(records).to_csv(output / 'class2_example_wave_ids.csv', index=False)


def body_bootstrap(output, data, models, draws=2000):
    """Paired body bootstrap of fixed OOF predictions, not repeated training CV."""
    from sklearn.metrics import confusion_matrix
    bodies = sorted(data.specimen_id.unique())
    weights = np.random.default_rng(42).multinomial(len(bodies), np.full(len(bodies), 1 / len(bodies)), size=draws)
    records = []
    for name in models:
        scores = {}
        for arm in ['four_class', 'three_class']:
            frame = pd.read_csv(output / 'predictions' / f'{arm}__{name}.csv')
            frame = frame[frame.label != 2]
            matrices = np.array([confusion_matrix(g.label, g.prediction, labels=[0, 1, 2, 3]) for body in bodies for g in [frame[frame.specimen_id == body]]])
            cm = (weights @ matrices.reshape(len(bodies), -1)).reshape(draws, 4, 4)
            tp = cm[:, KNOWN, KNOWN]
            true_n = cm[:, KNOWN, :].sum(axis=2)
            pred_n = cm[:, :, KNOWN].sum(axis=1)
            scores[arm] = dict(macro_f1=np.mean(2 * tp / np.maximum(true_n + pred_n, 1), axis=1),
                               balanced_accuracy=np.mean(tp / np.maximum(true_n, 1), axis=1),
                               critical_1_rate=cm[:, 1, 3] / np.maximum(cm[:, 1, :].sum(axis=1), 1),
                               false_alarm_rate=(cm[:, 3, 1] + cm[:, 3, 2]) / np.maximum(cm[:, 3, :].sum(axis=1), 1))
        for key in scores['four_class']:
            delta = scores['three_class'][key] - scores['four_class'][key]
            lo, med, hi = np.quantile(delta, [.025, .5, .975])
            records.append(dict(model=name, metric=key, delta_p025=lo, delta_median=med, delta_p975=hi, draws=draws))
    pd.DataFrame(records).to_csv(output / 'paired_body_bootstrap.csv', index=False)


def stability_plot(output):
    delta = pd.read_csv(output / 'paired_deltas.csv')
    body = delta[(delta.level == 'specimen_id') & (delta.metric == 'critical_1_rate')].dropna().pivot(index='group', columns='model', values='delta') * 100
    bootstrap = pd.read_csv(output / 'paired_body_bootstrap.csv')
    bootstrap = bootstrap[bootstrap.metric == 'critical_1_rate'].set_index('model').loc[body.columns, ['delta_median', 'delta_p025', 'delta_p975']] * 100
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw=dict(width_ratios=[1.6, 1]))
    extent = max(abs(body.to_numpy()).max(), 1)
    image = axes[0].imshow(body, cmap='coolwarm', vmin=-extent, vmax=extent, aspect='auto')
    axes[0].set_xticks(range(len(body.columns)), body.columns, rotation=40, ha='right')
    axes[0].set_yticks(range(len(body.index)), body.index)
    for i in range(len(body)):
        for j in range(len(body.columns)):
            axes[0].text(j, i, f'{body.iloc[i, j]:+.1f}', ha='center', va='center', fontsize=8)
    axes[0].set_title('Δ kritische Klasse-1-Rate je Körper (Prozentpunkte)')
    fig.colorbar(image, ax=axes[0], shrink=.65)
    y = np.arange(len(bootstrap))
    mid = bootstrap.delta_median.to_numpy()
    axes[1].errorbar(mid, y, xerr=np.vstack([mid - bootstrap.delta_p025, bootstrap.delta_p975 - mid]), fmt='o', capsize=3)
    axes[1].set_yticks(y, bootstrap.index)
    axes[1].axvline(0, color='gray', linestyle='--')
    axes[1].set(title='Gepaarter Körperbootstrap: 2,5–97,5 %', xlabel='Δ kritisch 1→3: ohne minus mit Klasse 2 (pp)')
    fig.tight_layout(); fig.savefig(output / 'critical_error_stability.png', dpi=160); plt.close(fig)


def review(output):
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text())
    data, X, features, _ = load_inputs(Path(plan['source']))
    data = pd.concat([data, positions(data)], axis=1)
    metrics = pd.read_csv(output / 'known_metrics.csv')
    pooled = metrics[(metrics.level == 'pooled') & (metrics['view'] == 'native')]
    models = pooled.model.drop_duplicates().tolist()
    a = pooled[pooled.arm == 'four_class'].set_index('model').loc[models]
    b = pooled[pooled.arm == 'three_class'].set_index('model').loc[models]
    comparison = pd.concat([a.add_suffix('_four'), b.add_suffix('_three')], axis=1)
    comparison.to_csv(output / 'pooled_comparison.csv')
    body_bootstrap(output, data, models)
    delta = []
    for level in ['fold', 'specimen_id', 'source_dataset']:
        part = metrics[(metrics.level == level) & (metrics['view'] == 'native')]
        four = part[part.arm == 'four_class'].set_index(['model', 'group'])
        three = part[part.arm == 'three_class'].set_index(['model', 'group'])
        for key in ['macro_f1', 'balanced_accuracy', 'critical_1_rate', 'false_alarm_rate', 'false_alarm_1_rate', 'recall_1', 'recall_3']:
            for (model, group), value in (three[key] - four[key]).items():
                delta.append(dict(level=level, model=model, group=group, metric=key, delta=value))
    pd.DataFrame(delta).to_csv(output / 'paired_deltas.csv', index=False)
    stability_plot(output)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, key, title in zip(axes.flat, ['macro_f1', 'balanced_accuracy', 'recall_1', 'critical_1_rate', 'false_alarm_rate', 'false_alarm_1_rate'],
                              ['Macro-F1 (0/1/3)', 'Balanced Accuracy', 'Recall Klasse 1', 'Kritisch 1→3', 'False Alarm 3→1/2', 'False Alarm 3→1']):
        x = np.arange(len(models))
        ax.bar(x - .2, a[key] * 100, .4, label='Training 0/1/2/3')
        ax.bar(x + .2, b[key] * 100, .4, label='Training 0/1/3')
        ax.set(title=title, ylabel='%'); ax.set_xticks(x, models, rotation=40, ha='right')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'known_class_comparison.png', dpi=160); plt.close(fig)

    predictions, summaries = [], []
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, model in zip(axes.flat, models):
        frame = pd.read_csv(output / 'predictions' / f'three_class__{model}.csv')
        if not frame.instance_id.equals(data.instance_id):
            raise ValueError('OOF alignment changed')
        frame = frame[frame.label == 2].copy()
        frame['model'] = model
        frame['q_cavity_1_vs_3'] = frame.p_1 / np.maximum(frame.p_1 + frame.p_3, 1e-15)
        frame['max_probability'] = frame[['p_0', 'p_1', 'p_3']].max(axis=1)
        frame['entropy'] = -(frame[['p_0', 'p_1', 'p_3']].to_numpy() * np.log(np.maximum(frame[['p_0', 'p_1', 'p_3']].to_numpy(), 1e-15))).sum(axis=1)
        predictions.append(frame)
        for level, groups in [('pooled', [('all', frame)]), *[(field, frame.groupby(field)) for field in ['fold', 'specimen_id', 'source_dataset']]]:
            for group, g in groups:
                row = dict(model=model, level=level, group=group, points=len(g),
                           p0_mean=g.p_0.mean(), p1_mean=g.p_1.mean(), p3_mean=g.p_3.mean(),
                           confidence_median=g.max_probability.median(), q_median=g.q_cavity_1_vs_3.median(), entropy_mean=g.entropy.mean())
                for label in KNOWN:
                    selected = g.prediction == label
                    row[f'predicted_{label}_count'] = int(selected.sum())
                    row[f'predicted_{label}_fraction'] = float(selected.mean())
                    row[f'mean_probability_given_prediction_{label}'] = g.loc[selected, f'p_{label}'].mean()
                summaries.append(row)
        for source, g in frame.groupby('source_dataset'):
            ax.hist(g.q_cavity_1_vs_3, bins=np.linspace(0, 1, 21), histtype='step', label=source)
        ax.set(title=model, xlabel='p(1) / [p(1)+p(3)]', ylabel='Klasse-2-Punkte')
    axes.flat[0].legend(fontsize=7); axes.flat[-1].axis('off')
    fig.tight_layout(); fig.savefig(output / 'class2_probabilities.png', dpi=160); plt.close(fig)
    long = pd.concat(predictions, ignore_index=True)
    long.to_csv(output / 'class2_probabilities.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'class2_assignment_summary.csv', index=False)
    consensus = long.groupby('instance_id').agg(mean_p0=('p_0', 'mean'), mean_p1=('p_1', 'mean'), mean_p3=('p_3', 'mean'),
                                               votes_0=('prediction', lambda p: int((p == 0).sum())), votes_1=('prediction', lambda p: int((p == 1).sum())),
                                               votes_3=('prediction', lambda p: int((p == 3).sum())))
    consensus['mean_q'] = consensus.mean_p1 / np.maximum(consensus.mean_p1 + consensus.mean_p3, 1e-15)
    context = data[data.label == 2].merge(consensus, on='instance_id', validate='one_to_one').merge(neighbor_context(data), on='instance_id', validate='one_to_one')
    context.to_csv(output / 'class2_spatial_context.csv', index=False)
    spatial_rows = []
    for body, g in context.groupby('specimen_id'):
        valid = g[['mean_q', 'neighbor_cavity_fraction']].dropna()
        rho = spearmanr(valid.mean_q, valid.neighbor_cavity_fraction).statistic if len(valid) > 2 and valid.nunique().min() > 1 else np.nan
        spatial_rows.append(dict(specimen_id=body, points=len(g), mean_p0=g.mean_p0.mean(), mean_p1=g.mean_p1.mean(), mean_p3=g.mean_p3.mean(),
                                 unanimous_1=int((g.votes_1 == 7).sum()), unanimous_3=int((g.votes_3 == 7).sum()),
                                 neighbor_spearman=rho))
    pd.DataFrame(spatial_rows).to_csv(output / 'class2_body_context.csv', index=False)
    representative_waves(output, context, Path(plan['source']))
    # One panel per specimen preserves spatial context; original labels are never replaced.
    bodies = sorted(context.specimen_id.unique())
    fig, axes = plt.subplots(int(np.ceil(len(bodies) / 4)), 4, figsize=(15, 3.2 * np.ceil(len(bodies) / 4)), squeeze=False)
    for ax, body in zip(axes.flat, bodies):
        background = data[data.specimen_id == body]
        g = context[context.specimen_id == body]
        ax.scatter(background.x, background.y, c='lightgray', s=3)
        point = ax.scatter(g.x, g.y, c=g.mean_q, vmin=0, vmax=1, cmap='coolwarm', s=8)
        ax.set(title=body, xlabel='X', ylabel='Y', aspect='equal')
    for ax in list(axes.flat)[len(bodies):]: ax.axis('off')
    fig.tight_layout(); fig.colorbar(point, ax=list(axes.flat), label='Mittleres q(1 vs 3), unkalibriert', shrink=.4)
    fig.savefig(output / 'class2_spatial_maps.png', dpi=150, bbox_inches='tight'); plt.close(fig)
    mixture_diagnostics(output, data, X)
    mixture = context.merge(pd.read_csv(output / 'class2_mixture_oof.csv'), on=['instance_id', 'fold'], validate='one_to_one')
    mixture.groupby(['fold', 'component', 'source_dataset']).agg(points=('instance_id', 'size'), mean_p1=('mean_p1', 'mean'),
                                                              mean_p3=('mean_p3', 'mean'), neighbor_fraction=('neighbor_cavity_fraction', 'mean')).to_csv(output / 'mixture_source_context.csv')
    # Actual class distributions in each input feature, independent of model-assigned labels.
    feature_rows = []
    for label in [0, 1, 2, 3]:
        for col, values in zip(features, X[data.label == label].T):
            q = np.nanquantile(values, [.1, .25, .5, .75, .9])
            feature_rows.append(dict(label=label, feature=col, p10=q[0], p25=q[1], median=q[2], p75=q[3], p90=q[4]))
    pd.DataFrame(feature_rows).to_csv(output / 'feature_label_quantiles.csv', index=False)
    print('Descriptive review complete; probability scores are not ground truth.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    review(parser.parse_args().output)
