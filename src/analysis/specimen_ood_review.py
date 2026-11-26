"""Cross-fit review scores on the frozen train-only Pareto baselines."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.analysis.nested_release_review import load_scores
from src.analysis.rbf_svm_oof_thresholds import frozen_train
from src.data.split_manifest import sha256_file
from src.experiments.svm_nested_hpo import KEYS, SOURCE
from src.experiments.tabular_nested_hpo import SVM, verified_assignments
from src.features.specimen_inputs import load_early_manual_train, read_json
from src.paths import ANALYSIS_RESULTS

PARETO = ANALYSIS_RESULTS / 'nested_release_pareto_v1'
DETECTORS = ('isolation', 'distance', 'neighbor_risk', 'uncertainty')
OUTPUT = ANALYSIS_RESULTS / 'specimen_ood_review_v2'


def novelty(fit, held, features):
    if set(fit.specimen_id) & set(held.specimen_id):
        raise ValueError('Specimen overlap in detector fit')
    # Bound the influence of densely measured specimens on the reference population.
    rng = np.random.default_rng(42)
    ref = np.concatenate([rng.choice(group.index, min(len(group), 256), replace=False)
                          for _, group in fit.groupby('specimen_id', sort=True)])
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    x = scaler.fit_transform(imputer.fit_transform(features.loc[ref]))
    z = scaler.transform(imputer.transform(features.loc[held.index]))
    forest = IsolationForest(n_estimators=100, max_samples=256, random_state=42, n_jobs=1).fit(x)
    distances, indices = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(x).kneighbors(z)
    labels = fit.loc[ref, 'label'].isin([1, 2]).to_numpy()
    # The last score is supervised local risk, separate from the two novelty scores.
    return pd.DataFrame({'isolation': -forest.score_samples(z), 'distance': distances[:, -1],
                         'neighbor_risk': labels[indices].mean(axis=1)}, index=held.instance_id)


def budget_cutoff(score, release, basis_points):
    """Strict > rejects at most the inner budget (all measurements); ties stay together."""
    available = np.asarray(score)[release]
    allowed = len(release) * basis_points // 10000
    if not len(available) or allowed >= len(available):
        return -np.inf
    index = len(available) - allowed - 1
    return float(np.partition(available, index)[index])


def assess(inner, outer, budget, target, fold, decisions):
    options = [(0, 0, 'none', np.inf)]
    crit = inner.label.isin([1, 2])
    for name in DETECTORS:
        threshold = budget_cutoff(inner[name], inner.release, budget)
        flag = inner.release & (inner[name] > threshold)
        options.append((-int((flag & crit).sum()), int(flag.sum()), name, threshold))
    best = min(options, key=lambda option: option[:2])
    groups = []
    for caught, count, name, threshold in options:
        flag = outer.release & (outer[name] > threshold) if name != 'none' else outer.release & False
        released = outer.release & ~flag
        y = outer.label
        rows = pd.DataFrame(dict(specimen=outer.specimen_id, n=1, n1=y.eq(1).astype(int),
                                 n2=y.eq(2).astype(int), n3=y.eq(3).astype(int),
                                 e1=(released & y.eq(1)).astype(int), e2=(released & y.eq(2)).astype(int),
                                 caught1=(flag & y.eq(1)).astype(int), caught2=(flag & y.eq(2)).astype(int),
                                 extra=flag.astype(int), released=released.astype(int),
                                 reviews=(outer.review | flag).astype(int), alarms=outer.alarm.astype(int)))
        grouped = rows.groupby('specimen').sum().reset_index().assign(
            target_bp=target, fold=fold, budget_bp=budget, detector=name)
        groups.append(grouped)
        if name == best[2]:
            groups.append(grouped.assign(detector='selected'))
        decisions.append(dict(target_bp=target, fold=fold, budget_bp=budget, detector=name,
                              selected=name == best[2], threshold=threshold, inner_n=len(inner),
                              inner_extra=count, inner_caught=-caught))
    return groups


def base_frames(row, train, assignment, cache):
    scores = []
    for m in row.members.split('+'):
        key = (row.features, m)
        if key not in cache:
            cache[key] = load_scores(m, row.features, row.fold, train, assignment)
        scores.append(cache[key])
    frames = []
    for side in (0, 1):
        data = scores[0][side][KEYS].copy()
        released = np.ones(len(data), dtype=bool)
        probs = []
        for model, item in zip(row.members.split('+'), scores):
            f = item[side]
            pd.testing.assert_frame_equal(data[KEYS], f[KEYS])
            probs.append(f.p3.to_numpy())
            released &= f.base.eq(3) & f.p3.ge(item[2]['threshold'])
            if model == row.lead:
                data['base'] = f.base
        data['p3_min'] = np.minimum.reduce(probs)
        data['uncertainty'] = 1 - data.p3_min
        data['release'] = released & data.p3_min.ge(float(row.cutoffs))
        data['review'] = data.base.eq(3) & ~data.release
        data['alarm'] = data.label.eq(3) & data.base.isin([1, 2])
        frames.append(data.set_index('instance_id', verify_integrity=True))
    np.testing.assert_array_equal([frames[0].release.sum(), frames[0].review.sum()], [row.released, row.reviews])
    return frames


def run(output):
    if output.exists():
        raise FileExistsError(output)
    train = frozen_train(SOURCE)
    features = load_early_manual_train(train)
    assignments = verified_assignments(train, SVM / 'inner_assignments.parquet',
                                       read_json(SVM / 'plan.json')['inner_assignment_sha256'])
    choices = pd.read_csv(PARETO / 'choices.csv', dtype={'cutoffs': str})
    choices = choices.query("preference == 'release' and target_bp in [50, 200]")
    groups, decisions, points = [], [], []
    with threadpool_limits(limits=1):
        for fold in range(3):
            assignment = assignments.loc[assignments.outer.eq(fold)]
            fit = train.loc[train.fold.ne(fold)]
            inner_fold = fit.instance_id.map(assignment.set_index('instance_id').fold)
            if inner_fold.isna().any():
                raise ValueError('Incomplete inner assignment')
            inside = pd.concat([novelty(fit.loc[inner_fold.ne(k)], fit.loc[inner_fold.eq(k)], features)
                                for k in range(3)], verify_integrity=True)
            outside = novelty(fit, train.loc[train.fold.eq(fold)], features)
            cache = {}
            for row in choices.loc[choices.fold.eq(fold)].itertuples():
                inner, outer = base_frames(row, train, assignment, cache)
                inner, outer = inner.join(inside, validate='one_to_one'), outer.join(outside, validate='one_to_one')
                if inner.isna().any(axis=None) or outer.isna().any(axis=None):
                    raise ValueError('Incomplete OOD join')
                points.append(outer.assign(target_bp=row.target_bp))
                for budget in (50, 100, 200):
                    groups.extend(assess(inner, outer, budget, row.target_bp, fold, decisions))
    results = pd.concat(groups, ignore_index=True)
    totals = results.groupby(['target_bp', 'budget_bp', 'detector'], as_index=False).sum(numeric_only=True).drop(columns='fold')
    baseline = pd.read_csv(PARETO / 'nested.csv').query("preference == 'release'").set_index('target_bp')
    for row in totals.query("detector == 'none'").itertuples():
        old = baseline.loc[row.target_bp]
        np.testing.assert_array_equal([row.n, row.e1, row.e2, row.reviews, row.alarms, row.released],
                                      old[['n', 'error1', 'error2', 'reviews', 'alarms', 'released']])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / 'result'
        stage.mkdir()
        results.to_csv(stage / 'specimens.csv', index=False)
        totals.to_csv(stage / 'totals.csv', index=False)
        pd.DataFrame(decisions).to_csv(stage / 'selection.csv', index=False)
        pd.concat(points).reset_index().to_parquet(stage / 'train_oof.parquet', index=False)
        stage.rename(output)
    mlflow.set_tracking_uri(f'sqlite:///{output.resolve() / "mlflow.db"}')
    exp = mlflow.MlflowClient().create_experiment('specimen-ood-review', artifact_location=(output.resolve() / 'artifacts').as_uri())
    with mlflow.start_run(experiment_id=exp):
        mlflow.log_params({'choices_sha256': sha256_file(PARETO / 'choices.csv'), 'seed': 42, 'neighbors': 15,
                           'reference_per_specimen': 256, 'detectors': ','.join(DETECTORS),
                           'source_sha256': sha256_file(Path(__file__)), 'svm_plan': sha256_file(SVM / 'plan.json')})
        for row in totals.query("detector == 'selected'").itertuples():
            mlflow.log_metrics({f'{row.target_bp}.{row.budget_bp}.{k}': getattr(row, k)
                                for k in ('e1', 'e2', 'extra', 'released', 'reviews')})
        for path in output.glob('*.csv'):
            mlflow.log_artifact(str(path))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, default=OUTPUT)
    run(cli.parse_args().output)
