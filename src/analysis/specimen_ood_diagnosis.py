"""Describe frozen outer-fold errors, with training-only waveform and feature controls."""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

from src.analysis.rbf_svm_oof_thresholds import frozen_train
from src.analysis.specimen_ood_review import OUTPUT
from src.experiments.svm_nested_hpo import KEYS, SOURCE
from src.features.experimental_anchors import early_index
from src.features.loading import load_measurement
from src.features.specimen_inputs import (
    ANCHOR,
    load_early_manual_train,
    read_json,
    require_hash,
)


def check_annotations(path, expected):
    """Compare only the requested training coordinates with the source annotations."""
    found = {}
    with path.open() as stream:
        for line in stream:
            fields = line.partition('#')[0].split()
            if not fields:
                continue
            position = tuple(map(float, fields[:2]))
            if position in expected:
                if position in found:
                    raise ValueError('Duplicate source annotation')
                found[position] = int(fields[2])
    if found != expected:
        raise ValueError('Missing or inconsistent source annotation')


def diagnose(root=OUTPUT):
    train = frozen_train(SOURCE)
    features = load_early_manual_train(train)
    points = pd.read_parquet(root / 'train_oof.parquet').query('target_bp == 200').sort_values('instance_id', ignore_index=True)
    pd.testing.assert_frame_equal(points[KEYS], train[KEYS])
    files = pd.read_csv(ANCHOR / 'cohort_folds.csv')
    pd.testing.assert_frame_equal(files[KEYS], train[KEYS])
    data = pd.concat([points, features], axis=1).assign(file_path=files.file_path)
    xy = data.instance_id.str.extract(r'_x(-?\d+)_y(-?\d+)$').astype(float)
    body = data.specimen_id
    primary = body.eq('messreihe_2_pk_08')
    near = cKDTree(xy.loc[primary & data.label.eq(1)]).query(xy)[0]
    correct = data.base.eq(data.label) & ~data.review
    groups = {
        'PK08 Fehler 1': primary & data.label.eq(1) & data.release,
        'PK08 lokale 3': primary & data.label.eq(3) & (near <= 3),
        'PK04 korrekte 1': body.eq('messreihe_2_pk_04') & data.label.eq(1) & correct,
        'Reihe 2 PK07/09 korrekte 1': body.isin(['messreihe_2_pk_07', 'messreihe_2_pk_09']) & data.label.eq(1) & correct,
        '3/06 Fehler 1': body.eq('messreihe_3_pk_06') & data.label.eq(1) & data.release,
        '3/06 korrekte 1': body.eq('messreihe_3_pk_06') & data.label.eq(1) & correct,
        '1/03 Fehler 2': body.eq('messreihe_1_pk_03') & data.label.eq(2) & data.release,
        '1/03 korrekte 2': body.eq('messreihe_1_pk_03') & data.label.eq(2) & correct,
        'Reihe 2 Fitklasse 1': data.source_dataset.eq('messreihe_2') & data.fold.ne(2) & data.label.eq(1),
        'Reihe 2 Fitklasse 3': data.source_dataset.eq('messreihe_2') & data.fold.ne(2) & data.label.eq(3),
    }
    checked = groups['PK08 Fehler 1'] | groups['PK08 lokale 3']
    expected = dict(zip(map(tuple, xy.loc[checked].to_numpy()), data.loc[checked, 'label'], strict=True))
    source = Path(data.loc[primary, 'file_path'].iloc[0]).parent.parent / 'hohltestk2_tr_markArea_XYList.txt'
    check_annotations(source, expected)
    hashes = read_json(ANCHOR / 'raw_sha256.json')
    cols = ['y_peak_to_peak', 'second_peak_to_surface_ratio', 'second_peak_snr', 'post_peak_energy',
            'p3_min', 'neighbor_risk', 'distance', 'isolation']
    stats, signal_stats = [], []
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    grid = np.arange(-2., 40.01, .05)
    for name, mask in groups.items():
        rows = data.loc[mask]
        for column in cols:
            q = rows[column].quantile([.25, .5, .75])
            stats.append(dict(group=name, n=len(rows), feature=column, q25=q.iloc[0], median=q.iloc[1], q75=q.iloc[2]))
        if name.startswith('Reihe'):
            continue
        waves = []
        for row in rows.sample(min(32, len(rows)), random_state=42).itertuples():
            path = Path(row.file_path)
            require_hash(path, hashes[str(path)])
            wave = load_measurement(path)
            x, y = wave.x.to_numpy(), wave.y.to_numpy()
            y = y - np.median(y)
            peak = early_index(x, y)
            z = np.interp(x[peak] + grid, x, y) / y[peak]
            late = (grid >= 6) & (grid <= 40)
            signal_stats.append(dict(group=name, instance_id=row.instance_id, anchor=x[peak],
                                     peak_to_peak=np.ptp(y), late_max=np.max(abs(z[late])),
                                     late_rms=np.sqrt(np.mean(z[late]**2))))
            waves.append(z)
        if name in ('PK08 Fehler 1', 'PK08 lokale 3', 'PK04 korrekte 1'):
            lo, med, hi = np.quantile(waves, [.25, .5, .75], axis=0)
            for ax in axes:
                line, = ax.plot(grid, med, label=f'{name} (n={len(waves)})', lw=1.4)
                ax.fill_between(grid, lo, hi, color=line.get_color(), alpha=.13)
    axes[0].set(xlim=(-2, 6), ylabel='Amplitude relativ zum frühen Extremum', title='Frühe Reflexion')
    axes[1].set(xlim=(6, 40), title='Nachlauf')
    for ax in axes:
        ax.set_xlabel('Abstand zum frühen Extremum (X-Einheiten)')
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=7)
    fig.savefig(root / 'signale.png', dpi=160)
    plt.close(fig)
    pd.DataFrame(stats).to_csv(root / 'distributions.csv', index=False)
    pd.DataFrame(signal_stats).to_csv(root / 'waveforms.csv', index=False)


if __name__ == '__main__':
    diagnose()
