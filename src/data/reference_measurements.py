"""Confirmed reference regions, scoped to the three original scan coordinates.

Confirmed by the dataset owner: upper isolated regions are references, not
specimens. This rule deliberately does not infer references from cluster size.
"""
import pandas as pd

REFERENCE_BOUNDS = {
    'messreihe_1': (-28, -8, 102, 120),
    'messreihe_2': (-29, -9, 103, 120),
    'messreihe_3': (-28, -8, 102, 112),
}


def reference_mask(frame):
    """Use original IDs if available, otherwise explicit source coordinates."""
    if 'instance_id' in frame:
        parsed = frame.instance_id.astype(str).str.extract(
            r'^(?P<source_dataset>messreihe_[123])_x(?P<x>-?\d+)_y(?P<y>-?\d+)$')
        parsed[['x', 'y']] = parsed[['x', 'y']].apply(pd.to_numeric)
    elif {'source_dataset', 'x', 'y'} <= set(frame):
        parsed = frame
    else:
        return pd.Series(False, index=frame.index)
    mask = pd.Series(False, index=frame.index)
    for source, (xmin, xmax, ymin, ymax) in REFERENCE_BOUNDS.items():
        mask |= (parsed.source_dataset.eq(source) & parsed.x.between(xmin, xmax)
                 & parsed.y.between(ymin, ymax))
    return mask


def exclude_references(frame):
    """Retain row order, indices, labels and feature values of specimen rows."""
    return frame.loc[~reference_mask(frame)].copy()


def require_no_references(frame):
    count = int(reference_mask(frame).sum())
    if count:
        raise ValueError(f'{count} reference measurements are excluded from training. '
                         'Use exclude_references on the input table before preparing data; '
                         'historical split manifests require a separate new evaluation setup.')
