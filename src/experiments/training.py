"""Common fit timing and exact RBF kernel reuse for grouped evaluation."""
import time
from itertools import repeat
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from src.experiments.model_registry import create_model


def fit_and_predict(model, X, y, evaluations):
    start = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - start
    predictions, durations = {}, {}
    for name, values in evaluations.items():
        start = time.perf_counter()
        predictions[name] = model.predict(values)
        durations[name] = time.perf_counter() - start
    return predictions, fit_seconds, durations


def shared_rbf_candidates(X_train, X_val, y_train, pool, output, *, grid, seed):
    """Reuse exact RBF matrices within one fold, backing the largest array by disk."""
    preprocessing = create_model('rbf_svm', seed=seed)[:-1]
    train = preprocessing.fit_transform(X_train)
    val = preprocessing.transform(X_val)
    block_size = 512
    for specification in dict.fromkeys(params['model__gamma'] for params in grid):
        indices = [i for i, params in enumerate(grid) if params['model__gamma'] == specification]
        variance = train.var()
        gamma = (1 / (train.shape[1] * variance) if variance else 1.0) if specification == 'scale' else specification
        started = time.perf_counter()
        with TemporaryDirectory(prefix='kernel-', dir=output) as temporary:
            kernel = np.memmap(Path(temporary) / 'train.dat', dtype='float64', mode='w+',
                               shape=(len(train), len(train)))
            for start in range(0, len(train), block_size):
                end = min(start + block_size, len(train))
                kernel[start:end] = rbf_kernel(train[start:end], train, gamma=gamma)
            kernel.flush()
            train_kernel_seconds = time.perf_counter() - started

            def fit(i, train_kernel):
                model = create_model('rbf_svm', seed=seed)[-1]
                model.set_params(kernel='precomputed', cache_size=192,
                                 **{key.removeprefix('model__'): value for key, value in grid[i].items()})
                start = time.perf_counter()
                model.fit(train_kernel, y_train)
                return model, time.perf_counter() - start

            fitted = list(pool.map(fit, indices, repeat(kernel)))
            del kernel
        started = time.perf_counter()
        predictions = np.empty((len(indices), len(val)), dtype=int)
        predict_seconds = np.zeros(len(indices))
        kernel_seconds = 0.0
        for start in range(0, len(val), block_size):
            end = min(start + block_size, len(val))
            block_started = time.perf_counter()
            block = rbf_kernel(val[start:end], train, gamma=gamma)
            kernel_seconds += time.perf_counter() - block_started
            for j, (model, _) in enumerate(fitted):
                predict_started = time.perf_counter()
                predictions[j, start:end] = model.predict(block)
                predict_seconds[j] += time.perf_counter() - predict_started
        print(f'shared gamma {specification}: {train_kernel_seconds:.1f}s train kernel, '
              f'{time.perf_counter() - started:.1f}s validation incl. kernel', flush=True)
        for j, (i, (model, fit_seconds)) in enumerate(zip(indices, fitted)):
            yield i, predictions[j], dict(
                training_seconds=fit_seconds, prediction_seconds=float(predict_seconds[j]),
                train_kernel_seconds=train_kernel_seconds, validation_kernel_seconds=kernel_seconds,
                support_vectors=int(model.n_support_.sum()),
            )
