"""Paired feature-set comparison: validation decides, test is descriptive only."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.paths import EXPERIMENTS_ROOT

METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1',
           'critical_errors', 'critical_error_rate', 'false_alarms', 'false_alarm_rate']


def paired_comparison(manual, tsfresh, split, *, feature_names=('manual52', 'tsfresh708')):
    """Align fixed models; never pool validation and test observations."""
    left = manual.loc[manual.split.eq(split)].set_index('model').sort_index()
    right = tsfresh.loc[tsfresh.split.eq(split)].set_index('model').sort_index()
    if left.empty or not left.index.is_unique or not right.index.is_unique or not left.index.equals(right.index):
        raise ValueError('Exactly one row per matching model and split required')
    if 'critical_denominator' in left and 'critical_denominator' in right:
        if not left.critical_denominator.eq(right.critical_denominator).all():
            raise ValueError('Critical error supports differ')
    if 'false_alarm_denominator' in left and 'false_alarm_denominator' in right:
        if not left.false_alarm_denominator.eq(right.false_alarm_denominator).all():
            raise ValueError('False alarm supports differ')
    left_name, right_name = feature_names
    delta = 'manual_minus_tsfresh' if feature_names == ('manual52', 'tsfresh708') else f'{left_name}_minus_{right_name}'
    evaluation_role = {
        'cv': 'training_group_cv',
        'validation': 'primary_validation_comparison',
        'test': 'exploratory_test_comparison_not_for_selection',
    }[split]
    rows = []
    for model in left.index:
        row = {'model': model, 'evaluation_role': evaluation_role}
        for key in METRICS:
            row[f'{left_name}_{key}'] = float(left.loc[model, key])
            row[f'{right_name}_{key}'] = float(right.loc[model, key])
            row[f'delta_{delta}_{key}'] = float(left.loc[model, key] - right.loc[model, key])
        rows.append(row)
    return pd.DataFrame(rows)


def validation_assessment(manual, tsfresh):
    """Assess the pre-existing critical-error priority using validation only.

    Win counts across models are descriptive, not independent replicates or a
    significance test. Mixed critical-error results yield no uniform preference.
    """
    paired = paired_comparison(manual, tsfresh, 'validation')
    differences = paired.delta_manual_minus_tsfresh_critical_error_rate
    preference = ('tsfresh708' if differences.gt(0).all() else
                  'manual52' if differences.lt(0).all() else 'mixed_no_uniform_preference')
    summary = {'basis': 'validation only; no test metrics used', 'models': len(paired),
               'preferred_for_critical_error_priority': preference, 'selection_uses_test': False}
    for metric in ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1']:
        summary[f'manual52_higher_{metric}'] = int(paired[f'delta_manual_minus_tsfresh_{metric}'].gt(0).sum())
    for metric in ['critical_error_rate', 'false_alarm_rate']:
        summary[f'manual52_lower_{metric}'] = int(paired[f'delta_manual_minus_tsfresh_{metric}'].lt(0).sum())
    return paired, summary


def write_comparison(manual_csv, tsfresh_csv, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manual, tsfresh = pd.read_csv(manual_csv), pd.read_csv(tsfresh_csv)
    validation, summary = validation_assessment(manual, tsfresh)
    output_dir.mkdir(parents=True, exist_ok=False)
    validation.to_csv(output_dir / 'validation.csv', index=False)
    (output_dir / 'validation_assessment.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Test results are generated after the validation-only assessment is fixed.
    paired_comparison(manual, tsfresh, 'test').to_csv(output_dir / 'exploratory_test.csv', index=False)
    return summary


def write_fusion_comparison(fusion_run, baseline_comparison, output_dir):
    """Compare the same cohorts and only model pairs that have actual results."""
    fused_cv = pd.read_csv(fusion_run / 'pooled_oof_metrics.csv').assign(split='cv')
    fused_holdout = pd.read_csv(fusion_run / 'holdout_metrics.csv')
    reference_cv = pd.read_csv(baseline_comparison / 'cv_comparison.csv').assign(split='cv')
    if {'critical_1_support', 'critical_2_support'} <= set(reference_cv):
        reference_cv['critical_denominator'] = reference_cv.critical_1_support + reference_cv.critical_2_support
    reference_holdout = pd.read_csv(baseline_comparison / 'holdout_comparison.csv')
    fused = pd.concat([fused_cv, fused_holdout], ignore_index=True).assign(feature_set='fusion')
    reference = pd.concat([reference_cv, reference_holdout], ignore_index=True)
    ranks = pd.concat([fused, reference], ignore_index=True).sort_values(
        ['split', 'critical_errors', 'false_alarms', 'feature_set', 'model'],
    )
    pairs = []
    for (split, feature_set), right in reference.groupby(['split', 'feature_set'], sort=False):
        left = fused.loc[fused.split.eq(split) & fused.model.isin(right.model)]
        paired = paired_comparison(
            left, right.dropna(axis=1, how='all'), split, feature_names=('fusion', 'reference'),
        )
        paired.insert(0, 'split', split)
        paired.insert(1, 'reference_feature_set', feature_set)
        pairs.append(paired)
    output_dir.mkdir(parents=True, exist_ok=False)
    ranks.to_csv(output_dir / 'ranked_results.csv', index=False)
    pd.concat(pairs, ignore_index=True).to_csv(output_dir / 'paired_deltas.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manual', type=Path)
    parser.add_argument('--tsfresh', type=Path)
    parser.add_argument('--fusion-run', type=Path)
    parser.add_argument('--baseline-comparison', type=Path,
                        default=EXPERIMENTS_ROOT / 'svm_group_cv_comparison_v1')
    args = parser.parse_args()
    if args.fusion_run is not None and args.manual is None and args.tsfresh is None:
        write_fusion_comparison(args.fusion_run, args.baseline_comparison, args.output_dir)
    elif args.fusion_run is None and args.manual is not None and args.tsfresh is not None:
        print(json.dumps(write_comparison(args.manual, args.tsfresh, args.output_dir), indent=2))
    else:
        parser.error('Specify --fusion-run or both --manual and --tsfresh')


if __name__ == '__main__':
    main()
