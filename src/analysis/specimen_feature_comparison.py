"""Paired feature-set comparison: validation decides, test is descriptive only."""
import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ['accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1',
           'critical_errors', 'critical_error_rate', 'false_alarms', 'false_alarm_rate']


def paired_comparison(manual, tsfresh, split):
    """Align fixed models; never pool validation and test observations."""
    left = manual.loc[manual.split.eq(split)].set_index('model').sort_index()
    right = tsfresh.loc[tsfresh.split.eq(split)].set_index('model').sort_index()
    if left.empty or not left.index.is_unique or not right.index.is_unique or not left.index.equals(right.index):
        raise ValueError('Exactly one row per matching model and split required')
    if 'critical_denominator' in left and 'critical_denominator' in right:
        if not left.critical_denominator.equals(right.critical_denominator):
            raise ValueError('Critical error supports differ')
    rows = []
    for model in left.index:
        row = {'model': model, 'evaluation_role': ('primary_validation_comparison' if split == 'validation'
                                                  else 'exploratory_test_comparison_not_for_selection')}
        for key in METRICS:
            row[f'manual52_{key}'] = float(left.loc[model, key])
            row[f'tsfresh708_{key}'] = float(right.loc[model, key])
            row[f'delta_manual_minus_tsfresh_{key}'] = float(left.loc[model, key] - right.loc[model, key])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['manual', 'tsfresh', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_comparison(args.manual, args.tsfresh, args.output_dir), indent=2))


if __name__ == '__main__':
    main()
