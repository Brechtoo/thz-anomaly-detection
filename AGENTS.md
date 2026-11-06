# THz Anomaly Detection – Agent Instructions

## Project Context

Python 3.12 ML project for detecting cavities in THz reflection measurements using supervised and unsupervised methods. Goals: reliable generalization, reproducible experiments, maintainable software, and eventual real-world deployment.

Stack: NumPy, pandas, scikit-learn, MLflow, pytest, Git. Use the existing architecture, dependencies, and `.venv/bin/python`.

## Scientific Requirements

Classification labels:
- `0`: Poor-quality measurement
- `1`: Cavity
- `2`: Uncertain
- `3`: No cavity

Critical errors: true `1/2` predicted as `3`.
False alarms: true `3` predicted as `1/2`.

Prioritize minimizing critical errors.

Preserve established baselines, preprocessing, features, data splits, evaluation definitions, and model parameters unless changes are explicitly approved. Keep new experiments separate.

Prevent data leakage and document relevant seeds, configurations, and dataset versions. Never claim reproducibility or generalization without supporting evidence.

## Development Guidelines

- Prefer the simplest solution that fully satisfies the requirements. Keep code, tests, documentation, and generated artifacts proportional to the task's complexity and risks. Avoid unnecessary complexity without compromising correctness, maintainability, or reproducibility.
- Prefer simple, readable, modular code consistent with the existing project.
- Avoid unnecessary abstractions, dependencies, duplicate code, and excessive documentation or generated artifacts.
- Write deterministic pytest tests for new or modified logic; prefer synthetic test data.
- Use MLflow for relevant experiments, recording parameters, metrics, and configurations.
- Never overwrite original measurement data or expose secrets.
- Do not upload project data or artifacts to external services without authorization.
- Do not commit large generated artifacts or local databases.

## Autonomy and Approval

For routine tasks, independently inspect, implement, test, review, and summarize the changes. Do not request approval for every step.

Request approval before:
- Changing established scientific methodology or model behavior.
- Destructive operations or modification of existing experiment results.
- Major architectural changes or significant new dependencies.
- Any Git commit or push.

## Communication and Learning

This project also serves to develop software engineering and MLOps skills.

- Briefly explain important new concepts and architectural decisions.
- Distinguish verified results from assumptions.
- Summarize changes, tests, and remaining limitations concisely.
- Ask questions only when essential.
- Avoid unnecessary reports, files, and intermediate confirmations.

A task is complete when the requested changes are implemented, relevant tests pass, and no unauthorized changes were introduced. Report tests that could not be executed.