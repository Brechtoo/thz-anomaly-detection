"""Exact RBF SVC fits sharing one kernel pair at a time within a nested split."""

import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from src.experiments.model_registry import create_model

LABELS = (0, 1, 2, 3)
IN_MEMORY_BYTES = 32 * 1024**2
BLOCK_BYTES = 32 * 1024**2


def _resolved_gamma(specification, train):
    if specification == 'auto':
        return 1 / train.shape[1]
    if specification == 'scale':
        variance = train.var()
        return float(1 / (train.shape[1] * variance)) if variance else 1.0
    value = float(specification)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'Expected a positive finite RBF gamma, got {specification!r}')
    return value


def _write_kernel(target, left, train, gamma):
    start = time.perf_counter()
    block_size = max(1, min(512, BLOCK_BYTES // (8 * len(train))))
    for lo in range(0, len(left), block_size):
        hi = min(lo + block_size, len(left))
        target[lo:hi] = rbf_kernel(left[lo:hi], train, gamma=gamma)
    if isinstance(target, np.memmap):
        target.flush()
    return time.perf_counter() - start


def _fit(candidate, train, evaluation, y, gamma, seed, train_kernel_seconds, evaluation_kernel_seconds):
    model = create_model('rbf_svm', seed=seed)[-1]
    model.set_params(
        C=candidate['C'], gamma=gamma, class_weight=candidate['class_weight'],
        kernel='precomputed', decision_function_shape='ovr', cache_size=192,
    )
    start = time.perf_counter()
    model.fit(train, y)
    training_seconds = time.perf_counter() - start
    if not np.array_equal(model.classes_, LABELS):
        raise ValueError(f'Unexpected SVC classes {model.classes_!r}')
    start = time.perf_counter()
    prediction = model.predict(evaluation)
    scores = model.decision_function(evaluation)
    prediction_seconds = time.perf_counter() - start
    if scores.shape != (len(evaluation), len(LABELS)) or not np.isfinite(scores).all():
        raise ValueError('Unexpected SVC decision scores')
    return candidate, prediction, scores, {
        'training_seconds': training_seconds,
        'prediction_seconds': prediction_seconds,
        'train_kernel_seconds': train_kernel_seconds,
        'evaluation_kernel_seconds': evaluation_kernel_seconds,
        'support_vectors': int(model.n_support_.sum()),
        'resolved_gamma': gamma,
    }


def _try_fit(candidate, *arguments):
    try:
        return _fit(candidate, *arguments)
    except (ValueError, FloatingPointError, RuntimeError) as error:
        return error


def iter_rbf_fits(X_fit, y_fit, X_eval, candidates, workspace, *, seed=42, jobs=1, on_error=None):
    """Yield candidates in gamma groups with four OvR columns ordered by label.

    The runner supplies fold-prepared features. This function uses no evaluation
    labels and applies no preprocessing or calibration. An optional callback
    records per-candidate fit failures and lets the remaining candidates proceed.
    """
    train = np.asarray(X_fit, dtype=np.float64, order='C')
    evaluation = np.asarray(X_eval, dtype=np.float64, order='C')
    y = np.asarray(y_fit)
    if (train.ndim != 2 or evaluation.ndim != 2 or train.shape[1] != evaluation.shape[1]
            or not train.shape[1] or not len(train) or not len(evaluation)
            or y.shape != (len(train),) or not np.array_equal(np.unique(y), LABELS)):
        raise ValueError('Expected compatible nonempty features and all four training labels')
    if not np.isfinite(train).all() or not np.isfinite(evaluation).all():
        raise ValueError('Fold-prepared features must be finite')
    if not isinstance(jobs, int) or jobs < 1:
        raise ValueError('jobs must be a positive integer')

    grouped = {}
    for candidate in candidates:
        gamma = _resolved_gamma(candidate['gamma'], train)
        grouped.setdefault(gamma, []).append(candidate)
    if not grouped:
        return
    train_shape = (len(train), len(train))
    eval_shape = (len(evaluation), len(train))
    bytes_needed = 8 * len(train) * (len(train) + len(evaluation))
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='svm-nested-kernel-', dir=workspace) as scratch:
        if bytes_needed <= IN_MEMORY_BYTES:
            train_kernel, eval_kernel = np.empty(train_shape), np.empty(eval_shape)
        else:
            free = shutil.disk_usage(scratch).free
            if bytes_needed > free:
                raise OSError(f'RBF kernel files need {bytes_needed:,} bytes, with {free:,} available in {scratch}')
            train_kernel = np.memmap(Path(scratch) / 'train.dat', dtype='float64', mode='w+', shape=train_shape)
            eval_kernel = np.memmap(Path(scratch) / 'eval.dat', dtype='float64', mode='w+', shape=eval_shape)
        for gamma, group in grouped.items():
            build_train = _write_kernel(train_kernel, train, train, gamma)
            build_eval = _write_kernel(eval_kernel, evaluation, train, gamma)
            # All workers read the same pair. Refill it only after the group is done.
            arguments = (train_kernel, eval_kernel, y, gamma, seed, build_train, build_eval)
            width = min(jobs, len(group))
            workers = ThreadPoolExecutor(max_workers=width) if width > 1 else nullcontext()
            with workers as pool:
                fit_all = pool.map if pool else map
                outcomes = fit_all(_try_fit, group, *(repeat(argument) for argument in arguments))
                for candidate, outcome in zip(group, outcomes, strict=True):
                    if isinstance(outcome, Exception):
                        if on_error is None:
                            raise outcome
                        on_error(candidate, outcome)
                    else:
                        yield outcome
