"""Compare prespecified review policies and evaluate inner-only operating-point selection."""

import argparse
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.analysis.nested_release_review import FAMILIES, load_scores
from src.analysis.rbf_svm_oof_thresholds import frozen_train
from src.data.split_manifest import sha256_file
from src.experiments.svm_nested_hpo import FEATURES, KEYS, SOURCE
from src.experiments.tabular_nested_hpo import OUTPUT, SVM, verified_assignments
from src.features.specimen_inputs import read_json
from src.paths import ANALYSIS_RESULTS

# Integer basis points keep strict boundaries exact, including 0.5%.
TARGETS = (50, 100, 200)
GRID = (10, 25, 50, 75, 100, 150, 200, 300, 500, 800)
CORE = ('rbf_svm', 'mlp', 'logistic_regression')
MEMBERS = ([(m,) for m in FAMILIES] + list(combinations(CORE, 2)) + [CORE, FAMILIES])
KEY = ['features', 'members', 'lead', 'rule', 'inner_bp']
COUNTS = ['n', 'n1', 'n2', 'n3', 'error1', 'error2', 'alarms', 'reviews', 'released']
OBJECTIVES = ['critical_rate', 'alarm_rate', 'review_rate', 'release_rate']


def risk_cutoffs(scores, labels, folds, bp):
    """Use -1 for ineligible rows; thresholds are inclusive and ties stay together."""
    critical = np.isin(labels, [1, 2])

    def cutoff(mask):
        support = int(mask.sum())
        if not support:
            raise ValueError('Missing critical support in inner constraint')
        allowed = (bp * support - 1) // 10000
        eligible = scores[mask & (scores >= 0)]
        if len(eligible) <= allowed:
            return 0.
        index = len(eligible) - allowed - 1
        return float(np.nextafter(np.partition(eligible, index)[index], np.inf))

    return {'pooled': cutoff(critical),
            'fold': max(cutoff(critical & (folds == f)) for f in np.unique(folds)),
            'class': max(cutoff(labels == c) for c in (1, 2))}


def pareto_mask(frame):
    # All candidates cover the same rows, hence integer counts give exact dominance.
    costs = frame[['error1', 'error2', 'alarms', 'reviews', 'released']].to_numpy()
    costs = np.column_stack((costs[:, 0] + costs[:, 1], costs[:, 2:4], -costs[:, 4]))
    return np.array([not np.any(np.all(costs <= row, axis=1) & np.any(costs < row, axis=1))
                     for row in costs])


def choose(frame, target, preference):
    errors = frame.error1 + frame.error2
    feasible = frame.loc[(10000 * errors < target * (frame.n1 + frame.n2)) & frame.released.gt(0)]
    if feasible.empty:
        raise ValueError(f'No nonempty release rule for {target} basis points')
    order = ['released', 'reviews', 'alarms'] if preference == 'release' else ['reviews', 'alarms', 'released']
    return feasible.assign(errors=errors).sort_values(
        [*order, 'errors', *KEY], ascending=[k != 'released' for k in order] + [True] * (1 + len(KEY)),
    ).iloc[0]


def add_rates(frame):
    frame = frame.copy()
    frame['critical_rate'] = 100 * (frame.error1 + frame.error2) / (frame.n1 + frame.n2).replace(0, np.nan)
    for col, num, denom in [('class1_rate', 'error1', 'n1'), ('class2_rate', 'error2', 'n2'),
                            ('alarm_rate', 'alarms', 'n3'), ('review_rate', 'reviews', 'n'),
                            ('release_rate', 'released', 'n')]:
        frame[col] = 100 * frame[num] / frame[denom].replace(0, np.nan)
    return frame


class Counts:
    """Precompute supports and count class-3 releases per true class and group."""

    def __init__(self, labels, groups, bases):
        self.bins = groups * 4 + labels
        self.size = 4 * (1 + groups.max())
        self.support = self.count(np.ones(len(labels), dtype=bool))
        self.base3 = {m: self.count(b == 3).sum(axis=1) for m, b in bases.items()}
        self.alarms = {m: self.count(np.isin(b, [1, 2]))[:, 3] for m, b in bases.items()}

    def count(self, mask):
        return np.bincount(self.bins[mask], minlength=self.size).reshape(-1, 4)

    def rows(self, released, lead):
        total = released.sum(axis=1)
        return np.column_stack((self.support.sum(axis=1), self.support[:, 1:], released[:, 1:3],
                                self.alarms[lead], self.base3[lead] - total, total))


def assess(combo, fold, train, assignments):
    inside, outside, bases, held_bases = {}, {}, {}, {}
    inner_keys = None
    for m in FAMILIES:
        inner, held, choice = load_scores(m, combo, fold, train, assignments)
        if inner_keys is None:
            inner_keys, held_keys = inner[KEYS], held[KEYS]
        else:
            pd.testing.assert_frame_equal(inner[KEYS], inner_keys)
            pd.testing.assert_frame_equal(held[KEYS], held_keys)
        for frame, scores, base in ((inner, inside, bases), (held, outside, held_bases)):
            base[m] = frame.base.to_numpy()
            scores[m] = np.where((base[m] == 3) & (frame.p3 >= choice['threshold']), frame.p3, -1.)
    labels, folds = inner_keys.label.to_numpy(), inner_keys.fold.to_numpy()
    bodies, names = pd.factorize(held_keys.specimen_id, sort=True)
    ic = Counts(labels, np.zeros(len(labels), dtype=int), bases)
    oc = Counts(held_keys.label.to_numpy(), bodies, held_bases)
    inner_rows, outer_rows = [], []
    singles = {}
    for members in MEMBERS:
        si = np.minimum.reduce([inside[m] for m in members])
        so = np.minimum.reduce([outside[m] for m in members])
        leads = members if len(members) < 6 else ('rbf_svm',)

        def record(rule, bp, thresholds, ri, ro):
            i, o = ic.count(ri), oc.count(ro)
            for lead in leads:
                key = (combo, '+'.join(members), lead, rule, bp)
                inner_rows.append((*key, fold, '|'.join(map(repr, thresholds)), *ic.rows(i, lead)[0]))
                outer_rows.extend((*key, fold, name, *row) for name, row in zip(names, oc.rows(o, lead)))

        record('archived_floor', 0, [0.], si >= 0, so >= 0)
        for bp in GRID:
            cuts = risk_cutoffs(si, labels, folds, bp)
            if len(members) == 1:
                singles[members[0], bp] = cuts
            for rule, threshold in cuts.items():
                record(rule, bp, [threshold], si >= threshold, so >= threshold)
            if len(members) > 1 and bp in TARGETS:
                for rule in ('pooled', 'fold'):
                    cuts_m = [singles[m, bp][rule] for m in members]
                    ri = np.logical_and.reduce([inside[m] >= c for m, c in zip(members, cuts_m)])
                    ro = np.logical_and.reduce([outside[m] >= c for m, c in zip(members, cuts_m)])
                    record('member_' + rule, bp, cuts_m, ri, ro)
    return (pd.DataFrame(inner_rows, columns=[*KEY, 'fold', 'cutoffs', *COUNTS]),
            pd.DataFrame(outer_rows, columns=[*KEY, 'fold', 'specimen', *COUNTS]))


def aggregate(frame, keys):
    return add_rates(frame.groupby(keys, as_index=False)[COUNTS].sum())


def run(output):
    if output.exists():
        raise FileExistsError(output)
    train = frozen_train(SOURCE)
    assignments = verified_assignments(train, SVM / 'inner_assignments.parquet',
                                       read_json(SVM / 'plan.json')['inner_assignment_sha256'])
    inside, outside, winners = [], [], []
    with threadpool_limits(limits=1):
        for fold in range(3):
            inner_parts = []
            for combo in FEATURES:
                inner, outer = assess(combo, fold, train, assignments.loc[assignments.outer.eq(fold)])
                inner_parts.append(inner)
                outside.append(outer)
            inner = pd.concat(inner_parts, ignore_index=True)
            for target in TARGETS:
                for preference in ('release', 'review'):
                    choice = choose(inner, target, preference)
                    winners.append({**choice.to_dict(), 'target_bp': target, 'preference': preference})
            inside.append(inner)
            print(f'Outer fold {fold}: complete', flush=True)
    by_body = pd.concat(outside, ignore_index=True)
    summary = aggregate(by_body, KEY)
    if not summary.n.eq(len(train)).all() or not summary.n1.eq((train.label == 1).sum()).all():
        raise ValueError('Incomplete evaluation')
    summary['pareto'] = pareto_mask(summary)
    choices = pd.DataFrame(winners)
    selectors = by_body.merge(choices[[*KEY, 'fold', 'target_bp', 'preference']], on=[*KEY, 'fold'])
    ranking = aggregate(selectors, ['target_bp', 'preference'])
    if len(ranking) != 6 or not ranking.n.eq(len(train)).all():
        raise ValueError('Incomplete nested selection')
    # Bind local MLflow artifact location to its final directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        stage = Path(temp) / 'report'
        stage.mkdir()
        summary.to_csv(stage / 'policies.csv', index=False)
        summary.loc[summary.pareto].to_csv(stage / 'pareto.csv', index=False)
        by_body.to_parquet(stage / 'specimens.parquet', index=False)
        pd.concat(inside).to_parquet(stage / 'inner.parquet', index=False)
        choices.to_csv(stage / 'choices.csv', index=False)
        ranking.to_csv(stage / 'nested.csv', index=False)
        stage.rename(output)
    mlflow.set_tracking_uri(f'sqlite:///{output.resolve() / "mlflow.db"}')
    client = mlflow.MlflowClient()
    exp = client.create_experiment('nested-release-pareto', artifact_location=(output.resolve() / 'artifacts').as_uri())
    with mlflow.start_run(experiment_id=exp):
        mlflow.log_params({'svm_plan': sha256_file(SVM / 'plan.json'), 'tabular_plan': sha256_file(OUTPUT / 'plan.json'),
                           'seed': 42, 'grid_basis_points': str(GRID), 'source_sha256': sha256_file(Path(__file__))})
        mlflow.log_metric('n_pareto', int(summary.pareto.sum()))
        for row in ranking.itertuples():
            mlflow.log_metrics({f'{row.preference}.{row.target_bp}.{k}': getattr(row, k) for k in OBJECTIVES})
        for path in sorted(output.glob('*.csv')):
            mlflow.log_artifact(str(path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ANALYSIS_RESULTS / 'nested_release_pareto_v1')
    run(parser.parse_args().output)
