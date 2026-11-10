"""A cheap fake extractor verifies ID alignment and train-only imputation state."""
import json

import pandas as pd

from src.data.specimen_splits import write_specimen_splits
from src.features import specimen_tsfresh as module


def test_fresh_features_are_aligned_and_use_new_training_only(tmp_path, monkeypatch):
    rows = []
    mapping = []
    for group in range(35):
        for label in range(4):
            iid = f'id_{group:02d}_{label}'
            signal = tmp_path / f'{iid}.txt'
            signal.write_text(str(group * 4 + label))
            rows.append([iid, label, f's{group % 3}', group, label, str(signal)])
            mapping.append([iid, f'pk_{group:02d}'])
    catalog = pd.DataFrame(rows, columns=['instance_id', 'label', 'source_dataset', 'x', 'y', 'file_path'])
    catalog_dir = tmp_path / 'data/measurements'
    catalog_dir.mkdir(parents=True)
    catalog_path = catalog_dir / 'combined_dataset.csv'
    catalog.to_csv(catalog_path, index=False)
    map_path = tmp_path / 'mapping.csv'
    pd.DataFrame(mapping, columns=['instance_id', 'specimen_id']).to_csv(map_path, index=False)
    splits = tmp_path / 'splits'
    data = write_specimen_splits(catalog_path, map_path, splits, candidates=16)
    for name in ['extract_tsfresh_features.py', 'loading.py']:
        target = tmp_path / 'src/features' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('test source')
    class Pool:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def map(self, function, tasks, **kwargs): return map(function, tasks)
    def extract(task):
        return {'y__mean': float(module.Path(task[0]).read_text())}
    def select(features, labels, train_index, **kwargs):
        assert set(train_index) == set(data.index[data.split.eq('train')])
        return features
    monkeypatch.setattr(module, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(module, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(module, 'extract_measurement', extract)
    monkeypatch.setattr(module, 'select_training_features', select)
    output = tmp_path / 'features'
    module.build_specimen_features(splits, output, workers=1)
    result = pd.read_csv(output / 'features.csv')
    assert result.instance_id.tolist() == sorted(catalog.instance_id)
    assert result.y__mean.tolist() == list(range(140))
    imputation = pd.read_csv(output / 'imputation.csv', index_col=0)
    assert imputation.loc['y__mean', 'max'] == result.loc[data.split.eq('train'), 'y__mean'].max()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['training_only'] and manifest['raw_features'] == 1
