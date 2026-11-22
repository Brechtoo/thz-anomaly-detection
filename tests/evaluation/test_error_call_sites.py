"""Guard original-label arguments without importing experiment entry points.

Some entry points create directories at import time. Parse their source so
these checks cannot load models, start training, or write experiment outputs.
"""

import ast
from importlib import import_module
from pathlib import Path

import pytest


@pytest.mark.parametrize("filename,original_labels", [
    ("experiments/run_experiment.py", "y_val"),
    ("transformer/transformer_experiments.py", "y_true"),
    ("experiments/optimize_thresholds.py", "y_val"),
    ("evaluation/optimized_models_eval/evaluate_optimized.py", "y_val"),
    ("evaluation/final_eval/final_evaluation.py", "y_test"),
    ("transformer/final_evaluation_transformer.py", "y_test"),
])
def test_error_call_uses_original_labels(filename, original_labels):
    source = Path(__file__).resolve().parents[2] / "src" / filename
    tree = ast.parse(source.read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "calculate_error_counts"
    ]
    assert len(calls) == 1
    assert isinstance(calls[0].args[0], ast.Name)
    assert calls[0].args[0].id == original_labels


@pytest.mark.parametrize('module_name,attribute,subdirectory', [
    ('src.experiments.optimize_thresholds', 'OPTIMIZATION_DIR', 'optimization'),
    ('src.evaluation.final_eval.final_evaluation', 'OPTIMIZATION_DIR', 'optimization'),
    ('src.evaluation.final_eval.final_evaluation', 'FINAL_EVALUATION_DIR', 'final_evaluation'),
])
def test_legacy_evaluation_paths_use_project_root(monkeypatch, tmp_path, module_name, attribute, subdirectory):
    monkeypatch.chdir(tmp_path)
    module = import_module(module_name)
    project_root = Path(__file__).resolve().parents[2]
    experiment_dir = project_root / 'results' / 'experiments' / 'extended_features'

    assert getattr(module, attribute) == experiment_dir / subdirectory
