"""Paired outer-fold diagnostics for spatial input features."""

import numpy as np
import pandas as pd

from src.analysis.spatial_postprocessing_report import (
    _bootstrap,
    summarize,
)

POINT_ONLY = 'point_only'
PRIOR_ARMS = ('prior_native', 'prior_rescue')
KEYS = ['feature_set', 'arm', 'instance_id']
REQUIRED = [*KEYS, 'specimen_id', 'source_dataset', 'fold', 'label', 'x', 'y', 'prediction']


def _historical_errors(arms, feature_set):
    """Audit only the errors retained by the previous rescue, after outer prediction."""
    if 'prior_rescue' not in arms:
        return []
    prior = arms['prior_rescue']
    truth = prior.label.to_numpy()
    rescued = prior.prediction.to_numpy()
    error_masks = {
        'cavity_release_1_to_3': (truth == 1) & (rescued == 3),
        'uncertain_release_2_to_3': (truth == 2) & (rescued == 3),
        'false_alarm_3_to_1_or_2': (truth == 3) & np.isin(rescued, [1, 2]),
    }
    records = []
    for name, included in error_masks.items():
        if not included.any():
            continue
        specimens = prior.loc[included, ['source_dataset', 'specimen_id']].drop_duplicates()
        for arm, rows in arms.items():
            predictions = rows.prediction.to_numpy()[included]
            records.append({
                'feature_set': feature_set, 'arm': arm, 'historical_error': name,
                'specimens': len(specimens), 'support': int(included.sum()),
                **{f'predicted_{label}': int((predictions == label).sum()) for label in range(4)},
                'correct': int((predictions == truth[included]).sum()),
            })
    return records


def summarize_spatial_features(predictions, *, draws=2000, seed=42):
    """Compare identical outer measurements and reuse fixed spatial diagnostics.

    The matched point-only fit is the primary reference. If supplied, the
    historical arms provide additional paired intervals and error diagnostics.
    None of these summaries is used for feature or SVM selection.
    """
    frame = predictions[REQUIRED]
    if frame.empty or frame.isna().any(axis=None) or frame.duplicated(KEYS).any():
        raise ValueError('Expected unique complete prediction rows for every feature set and arm')
    if frame.arm.isin(['baseline', 'pointwise']).any():
        raise ValueError('Use point_only for the matched control and distinct names for other arms')
    controls = frame.loc[frame.arm.eq(POINT_ONLY)]
    cohort = None
    for _, control in controls.groupby('feature_set', sort=True):
        measurements = (control.drop(columns=['feature_set', 'arm', 'prediction'])
                        .set_index('instance_id').sort_index())
        if cohort is not None and not cohort.equals(measurements):
            raise ValueError('Feature sets must use the same measured points and group folds')
        cohort = measurements
    if controls.groupby(['feature_set', 'source_dataset', 'specimen_id']).fold.nunique().gt(1).any():
        raise ValueError('A specimen must belong to one evaluation fold')
    normalized = frame.merge(
        controls[['feature_set', 'instance_id', 'prediction']].rename(
            columns={'prediction': 'baseline_prediction'},
        ),
        on=['feature_set', 'instance_id'], how='left', validate='many_to_one',
    )
    if normalized.baseline_prediction.isna().any():
        raise ValueError('Every measurement requires a matched point_only decision')
    normalized = normalized.rename(columns={'feature_set': 'baseline', 'arm': 'strategy'})
    normalized['strategy'] = normalized.strategy.replace(POINT_ONLY, 'baseline')
    tables = summarize(normalized, draws=draws, seed=seed)
    del tables['paired_pointwise']
    tables = {name: table.rename(columns={'baseline': 'feature_set', 'strategy': 'arm'})
              for name, table in tables.items()}
    for table in tables.values():
        if 'arm' in table.columns:
            table['arm'] = table.arm.replace('baseline', POINT_ONLY)
    tables['paired_bootstrap']['reference_arm'] = POINT_ONLY

    intervals = []
    historical_errors = []
    # summarize has already verified exact metadata and label pairing for every arm.
    for feature_set, group in frame.groupby('feature_set', sort=True):
        arms = {arm: rows.sort_values('instance_id').reset_index(drop=True)
                for arm, rows in group.groupby('arm', sort=True)}
        historical_errors.extend(_historical_errors(arms, feature_set))
        for reference in PRIOR_ARMS:
            if reference not in arms:
                continue
            for arm, rows in arms.items():
                if arm == reference or arm == POINT_ONLY:
                    continue
                paired = rows.assign(baseline_prediction=arms[reference].prediction)
                intervals.extend(
                    {'feature_set': feature_set, 'arm': arm, 'reference_arm': reference, **item}
                    for item in _bootstrap(paired, draws, seed)
                )
    if intervals:
        tables['paired_bootstrap'] = pd.concat(
            [tables['paired_bootstrap'], pd.DataFrame(intervals)], ignore_index=True,
        )
    if historical_errors:
        tables['prior_rescue_errors'] = pd.DataFrame(historical_errors)
    return tables
