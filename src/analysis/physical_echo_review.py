"""Read-only audit of the V1 THz echo detector on frozen specimen cohorts."""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.split_manifest import sha256_file
from src.features import thz_physical as v1
from src.features.loading import load_measurement
from src.features.specimen_inputs import (
    TSFRESH,
    load_cohorts,
    read_json,
    require_hash,
    tsfresh_manifest,
    write_json,
)
from src.features.thz_echo_candidates import (
    candidates,
    correlation_curve,
    nonoverlapping_second,
    trace_windows,
)
from src.features.thz_physical_v2 import ADDED, REPLACED
from src.paths import FEATURE_TABLES_ROOT, PROJECT_ROOT

FEATURE_FILE = FEATURE_TABLES_ROOT / 'specimen_thz_physical_v1.parquet'
PRIMARY = 'physical__ncc__primary_signed'
SECONDARY = 'physical__ncc__secondary_to_primary'
SPACING = 'physical__ncc__peak_separation_over_reference'
PER_STRATUM = 3


def inspect_trace(x, y):
    t, y, dx, lo, hi, start, end = trace_windows(x, y)
    ref = y[lo:hi]
    score, rms = correlation_curve(ref, y[start:end])
    peaks = candidates(score, rms, dx)
    second = nonoverlapping_second(peaks, len(ref))
    endpoint = np.abs(score[[0, -1]]) >= v1.MIN_NCC
    eligible = rms > v1.MIN_RMS_FRACTION

    early = correlation_curve(ref, y[hi:end])
    extended = correlation_curve(ref, y[start:min(len(y), end + round(20 / dx))])
    summary = {
        'baseline_count': len(peaks),
        'primary_ncc': score[peaks[0]] if len(peaks) else np.nan,
        'relative_separation': abs(peaks[1] - peaks[0]) / len(ref) if len(peaks) > 1 else np.nan,
        'second_ratio': abs(score[peaks[1]] / score[peaks[0]]) if len(peaks) > 1 else np.nan,
        'best_ncc_rms_eligible': float(np.max(np.abs(score[eligible]))) if eligible.any() else np.nan,
        'rms_eligible_fraction': float(eligible.mean()),
        'rms_above_gate_at_height_endpoint': bool((endpoint & eligible[[0, -1]]).any()),
        'post_start_x': t[start],
        'template_edge_energy_fraction': float(np.sum(np.r_[ref[:10], ref[-10:]]**2) / np.dot(ref, ref)),
        'disjoint_second': second is not None,
        'relaxed_height_count': len(candidates(score, rms, dx, height=.35)),
        'relaxed_prominence_count': len(candidates(score, rms, dx, prominence=.04)),
        'relaxed_rms_count': len(candidates(score, rms, dx, gate=.01)),
        'earlier_count': len(candidates(*early, dx)),
        'extended_count': len(candidates(*extended, dx)),
    }
    return summary, (t, y, lo, hi, start, end, score, rms, peaks)


def missing_summary(frame, keys):
    groups = frame.groupby(keys, sort=True, observed=True)
    return groups.agg(
        n=('instance_id', 'size'),
        no_primary=('primary_missing', 'sum'),
        no_second=('second_missing', 'sum'),
        overlapping_second=('overlapping_second', 'sum'),
        disjoint_second=('disjoint_second', 'sum'),
    ).assign(no_second_rate=lambda d: d.no_second / d.n).reset_index()


def sample_rows(frame):
    """Deterministic within-body, label and detection-status audit sample."""
    parts = []
    for _, group in frame.groupby(['specimen_id', 'label', 'echo_status'], sort=True, observed=True):
        ordered = group.sort_values('instance_id')
        parts.append(ordered.iloc[np.linspace(0, len(group) - 1, min(len(group), PER_STRATUM), dtype=int)])
    return pd.concat(parts).sort_values('instance_id')


def plot_examples(sample, read_trace, output):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    selectors = [
        ('Kein Treffer', sample.baseline_count.eq(0)),
        ('Nur ein Treffer', sample.baseline_count.eq(1)),
        ('Schwelle 0,35 sensibel', sample.baseline_count.lt(2) & sample.relaxed_height_count.ge(2)),
        ('Bis +100 sensibel', sample.baseline_count.lt(2) & sample.extended_count.ge(2)),
        ('Überlappende Fenster', sample.baseline_count.ge(2) & ~sample.disjoint_second),
        ('Getrennte Fenster', sample.disjoint_second),
    ]
    fig, axes = plt.subplots(len(selectors), 2, figsize=(12, 14), layout='constrained')
    chosen = []
    for i, (name, mask) in enumerate(selectors):
        ax, corr = axes[i]
        section = sample.loc[mask]
        if section.empty:
            ax.set(title=name)
            corr.set_axis_off()
            continue
        row = section.iloc[len(section) // 2]
        chosen.append({'case': name, 'instance_id': row.instance_id})
        _, (t, y, lo, hi, start, end, score, rms, peaks) = inspect_trace(*read_trace(row.instance_id))
        ax.plot(t, y, lw=.8)
        ax.axvspan(t[lo], t[hi-1], alpha=.2, color='tab:blue')
        ax.axvspan(t[start], t[end-1], alpha=.12, color='tab:orange')
        ax.set(xlim=(-7, 102), title=f'{name} · {row.instance_id} · Klasse {row.label}',
               ylabel='Norm. Amplitude')
        locations = t[start:start+len(score)] + v1.LEFT
        corr.plot(locations, np.abs(score), color='.70', lw=.7)
        corr.plot(locations, np.where(rms > v1.MIN_RMS_FRACTION, abs(score), np.nan), lw=.8)
        corr.axhline(v1.MIN_NCC, ls='--', lw=.7, color='.35')
        corr.scatter(locations[peaks], np.abs(score[peaks]), color='tab:red', s=14)
        corr.set(xlim=(-7, 102), ylim=(0, 1.02), ylabel='|NCC|')
    for ax in axes[-1]:
        ax.set_xlabel('X relativ zum Frühanker')
    fig.savefig(output / 'signal_examples.png', dpi=140)
    plt.close(fig)
    pd.DataFrame(chosen).to_csv(output / 'signal_examples.csv', index=False)


def injection_controls(frame, read_trace):
    """Known injected returns on flat backgrounds using measured TRAIN templates."""
    train = frame.loc[frame.split.eq('train')]
    templates = train.groupby(['source_dataset', 'specimen_id', 'label', 'echo_status'],
                              sort=True, observed=True).sample(n=1, random_state=20261120)
    scenarios = [
        ('one_exact', (30,), False), ('one_tapered', (30,), True),
        ('two_separated_exact', (30, 49), False), ('two_separated_tapered', (30, 49), True),
        ('two_close', (30, 31.8), False), ('two_overlapping', (30, 35), False),
        ('before_search', (6,), False), ('after_last_alignment', (79,), False),
        ('one_weak', (42,), False),
    ]
    rows = []
    for instance_id in templates.instance_id:
        t, y, dx, lo, hi, start, end = trace_windows(*read_trace(instance_id))
        ref = y[lo:hi] - y[lo:hi].mean()
        for scenario, centers, taper in scenarios:
            signal = np.zeros_like(y)
            pulse = ref * np.hanning(len(ref)) if taper else ref
            positions = [int(np.argmin(abs(t - center))) - round(v1.LEFT / dx) for center in centers]
            gain = .01 if scenario == 'one_weak' else .15
            for position, amplitude in zip(positions, (gain, -.10)):
                signal[position:position+len(ref)] += amplitude * pulse
            score, rms = correlation_curve(ref, signal[start:end])
            peaks = candidates(score, rms, dx)
            second = nonoverlapping_second(peaks, len(ref))
            pair = [peaks[0], second] if second is not None else peaks[:1]
            tolerance = round(.30 / dx)
            localized = all(np.any(np.abs(np.asarray(peaks) + start - p) <= tolerance) for p in positions)
            localized_v2 = all(np.any(np.abs(np.asarray(pair) + start - p) <= tolerance) for p in positions)
            rows.append(dict(
                instance_id=instance_id, scenario=scenario, injected_returns=len(centers),
                multiple_v1=len(peaks) >= 2, disjoint_v2=second is not None,
                all_localized_v1=localized, all_localized_v2=localized_v2,
            ))
    return pd.DataFrame(rows)


def compare_variants(v1_table, v1_manifest, v2_path, output, *, resume):
    v2_manifest = read_json(v2_path.with_suffix('.json'))
    require_hash(v2_path, v2_manifest['parquet_sha256'])
    for path, digest in v2_manifest['input_sha256'].items():
        require_hash(PROJECT_ROOT / path, digest)
    v2 = pd.read_parquet(v2_path)
    shared = [*v1_manifest['metadata_columns'],
              *(name for name in v1_manifest['retained_features'] if name not in REPLACED)]
    pd.testing.assert_frame_equal(v1_table[shared], v2[shared], check_exact=True)
    disjoint = v2[ADDED[0]].notna()
    eligible = v1_table[SECONDARY].notna()
    unchanged_pair = v1_table[SPACING].ge(1)
    np.testing.assert_array_equal(disjoint, v2[ADDED[1]].notna())
    np.testing.assert_array_equal((disjoint & ~eligible).any(), False)
    np.testing.assert_array_equal(v1_table.loc[unchanged_pair, list(REPLACED)].to_numpy(),
                                  v2.loc[unchanged_pair, list(ADDED[:2])].to_numpy())
    assert v2[ADDED[0]].dropna().between(0, 1).all() and v2[ADDED[1]].dropna().ge(1).all()
    assert v2[ADDED[2]].between(0, 1).all() and v2[ADDED[3]].between(0, 1).all()
    comparison = v1_table.assign(v1_second_missing=~eligible, v2_second_missing=~disjoint,
                                 v2_eligible=v2[ADDED[3]], v2_best_ncc=v2[ADDED[2]])
    for key in ('label', 'source_dataset', 'specimen_id', 'split'):
        summary = comparison.groupby(key, as_index=False).agg(
            n=('instance_id', 'size'),
            v1_missing=('v1_second_missing', 'sum'),
            v2_missing=('v2_second_missing', 'sum'),
            v1_overlap=('overlapping_second', 'sum'),
            mean_eligible=('v2_eligible', 'mean'),
            mean_best_ncc=('v2_best_ncc', 'mean'),
        )
        summary['v1_missing_rate'] = summary.v1_missing / summary.n
        summary['v2_missing_rate'] = summary.v2_missing / summary.n
        path = output / f'v1_v2_by_{key}.csv'
        if resume and path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), summary)
        else:
            summary.to_csv(path, index=False, mode='x')
    return v2_manifest['parquet_sha256']


def run(output, *, resume=False, v2_path=None):
    started = time.perf_counter()
    manifest = read_json(FEATURE_FILE.with_suffix('.json'))
    protected = {FEATURE_FILE: manifest['parquet_sha256'],
                 FEATURE_FILE.with_suffix('.diagnostics.csv'): manifest['diagnostics_sha256'],
                 FEATURE_FILE.with_suffix('.json'): None,
                 PROJECT_ROOT / 'src/features/thz_physical.py': manifest['inputs_sha256']['src/features/thz_physical.py']}
    for path, checksum in protected.items():
        if checksum:
            require_hash(path, checksum)
        else:
            protected[path] = sha256_file(path)
    membership, train = load_cohorts()
    frame = pd.read_parquet(FEATURE_FILE)
    pd.testing.assert_frame_equal(frame[membership.columns], membership)
    np.testing.assert_array_equal(frame.cv_fold.dropna(), train.fold)
    assert frame.cv_fold.notna().equals(frame.split.eq('train'))
    frame['primary_missing'] = frame[PRIMARY].isna()
    frame['second_missing'] = frame[SECONDARY].isna()
    frame['echo_status'] = np.select([frame.primary_missing, frame.second_missing], [0, 1], default=2)
    frame['overlapping_second'] = frame[SPACING].lt(1)
    frame['disjoint_second'] = frame[SPACING].ge(1)
    tsfresh_manifest()
    raw = pd.read_csv(TSFRESH / 'raw_inputs.csv')
    if (not raw.instance_id.equals(frame.instance_id)
            or raw.sha256.tolist() != read_json(TSFRESH / 'extraction_inputs.json')['raw_hashes']):
        raise ValueError('Unexpected archive membership or raw hashes')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    for filename, keys in [
        ('by_source', ['source_dataset']), ('by_class', ['label']),
        ('by_source_class', ['source_dataset', 'label']), ('by_specimen', ['specimen_id']),
        ('by_specimen_class', ['specimen_id', 'label']), ('by_split', ['split']),
    ]:
        path = output / f'{filename}.csv'
        expected = missing_summary(frame, keys)
        if resume and path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), expected)
        else:
            expected.to_csv(path, index=False, mode='x')

    inputs = raw.set_index('instance_id')

    def read_trace(instance_id):
        path, digest = inputs.loc[instance_id, ['file_path', 'sha256']]
        require_hash(Path(path), digest)
        wave = load_measurement(path)
        return wave.x.to_numpy(), wave.y.to_numpy()

    columns = ['instance_id', 'specimen_id', 'source_dataset', 'label', 'split']
    reference = sample_rows(frame).reset_index(drop=True)
    sample_path = output / 'sample_diagnostics.csv'
    if resume and sample_path.exists():
        sample = pd.read_csv(sample_path)
    else:
        details = [inspect_trace(*read_trace(instance_id))[0] for instance_id in reference.instance_id]
        sample = pd.concat([reference[columns], pd.DataFrame(details)], axis=1)
        sample.to_csv(sample_path, index=False, mode='x')
    pd.testing.assert_frame_equal(sample[columns], reference[columns])
    for feature, key in [(PRIMARY, 'primary_ncc'), (SECONDARY, 'second_ratio'), (SPACING, 'relative_separation')]:
        np.testing.assert_allclose(sample[key], reference[feature], atol=1e-10, rtol=1e-10)

    plot_examples(sample, read_trace, output)
    controls_path = output / 'injection_controls.csv'
    if not controls_path.exists():
        injection_controls(frame, read_trace).to_csv(controls_path, index=False, mode='x')
    v2_sha256 = compare_variants(frame, manifest, v2_path, output, resume=resume) if v2_path else None
    for path, digest in protected.items():
        require_hash(path, digest)
    write_json(output / 'manifest.json', {
        'sample': 'up to three evenly spaced ordered IDs per specimen, label and V1 detection status',
        'sample_size': len(sample), 'protected_sha256': {str(k.relative_to(PROJECT_ROOT)): v for k, v in protected.items()},
        'raw_input_catalog_sha256': sha256_file(TSFRESH / 'raw_inputs.csv'),
        'audit_source_sha256': sha256_file(Path(__file__)), 'seconds': time.perf_counter() - started,
        'injection_controls_sha256': sha256_file(controls_path),
        'sample_sha256': sha256_file(sample_path),
        'v2_parquet_sha256': v2_sha256,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--v2', type=Path)
    args = parser.parse_args()
    run(args.output, resume=args.resume, v2_path=args.v2)
