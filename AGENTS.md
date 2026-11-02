# THz Anomaly Detection – Agent Instructions

## 1. Project Overview

This project develops a Python-based machine learning
system for detecting cavities in terahertz (THz)
reflection measurements.

The goals are:
- Develop reliable machine learning methods for cavity detection.
- Compare supervised and unsupervised approaches.
- Build reproducible ML experiments.
- Improve software engineering and MLOps practices.
- Eventually support integration into a real-world measurement system.

## 2. Technology Stack

- Python 3.12
- NumPy, pandas, scikit-learn
- MLflow for experiment tracking
- pytest for automated testing
- Git for version control
- PyCharm as the primary IDE

Use the existing project dependencies and architecture
whenever possible.

Use the Python interpreter located at .venv/bin/python.

## 3. Machine Learning Requirements

The existing supervised classification system uses four labels:
- 0: Poor-quality measurement
- 1: Cavity detected
- 2: Uncertain classification
- 3: No cavity detected

Project-specific error definitions:
- Critical errors: true label 1 or 2 predicted as label 3.
- False alarms: true label 3 predicted as label 1 or 2.

Minimizing critical errors is a major project objective.

Preserve existing scientific methodology unless
explicitly authorized to change it.

In particular:
- Do not modify existing data splits without approval.
- Do not modify preprocessing or feature definitions without approval.
- Do not change evaluation metrics or error definitions without approval.
- Do not change established model parameters without approval.
- Prevent data leakage between training, validation, and test data.
- Preserve reproducibility through fixed random seeds and documented configurations.

New experimental methods may be proposed and implemented
separately without replacing established baselines.

## 4. Software Engineering Guidelines

- Prefer simple, readable, modular, maintainable Python code.
- Follow existing project structure and naming conventions.
- Avoid unnecessary abstractions, classes, dependencies, and architectural complexity.
- Reuse existing functions where appropriate.
- Separate data processing, feature engineering, model training, evaluation, and experiment tracking.
- Use meaningful names and type hints where helpful.
- Avoid duplicated code.
- Write pytest tests for new or modified logic.
- Prefer small synthetic datasets for unit tests.
- Make tests deterministic and independent of external services whenever possible.

## 5. Autonomous Development

For routine, well-defined development tasks:
- Independently inspect the relevant code.
- Create an implementation plan internally.
- Implement the requested changes.
- Run relevant tests.
- Fix problems introduced by the changes.
- Review the resulting code.
- Report the results.

Do not request approval for every intermediate step.

Request explicit approval before:
- Changing scientific methodology or evaluation definitions.
- Changing established model behavior or data splits.
- Performing destructive operations.
- Introducing major architectural changes.
- Adding significant new dependencies.
- Modifying or deleting existing experiment results.

Never commit or push changes without explicit permission.

## 6. Experiment Tracking and Reproducibility

- Use MLflow for relevant machine learning experiments.
- Log important hyperparameters, metrics, and experiment configurations.
- Preserve existing baseline results.
- Keep experimental approaches separate from established baselines.
- Document random seeds and dataset versions where available.
- Do not claim reproducibility unless it has been verified.
- Report failed experiments and unexpected results transparently.

## 7. Data Protection

- Never modify or overwrite original measurement data.
- Do not expose API keys, credentials, or secrets.
- Do not upload measurement data, experiment artifacts, or sensitive project information to external services without explicit authorization.
- Use synthetic data for tests when possible.
- Do not commit large generated datasets, local databases, or temporary artifacts unless explicitly requested.

## 8. Communication and Learning

This repository is also used to learn professional
software engineering, machine learning engineering,
and MLOps.

- Prefer understandable solutions over unnecessary complexity.
- Briefly explain significant architectural decisions.
- Explain newly introduced tools, libraries, or patterns.
- Identify important trade-offs when relevant.
- Distinguish verified results from assumptions.
- Summarize modified files, executed tests, and remaining issues after each task.
- Ask questions only when essential information is missing or a high-impact decision requires approval.

## 9. Definition of Done

A development task is considered complete when:
- The requested functionality has been implemented.
- Relevant automated tests pass.
- Existing functionality has not been unintentionally changed.
- No unauthorized changes to scientific methodology or data have occurred.
- Important limitations or remaining issues are documented.
- A concise summary of changes and test results has been provided.

Do not claim completion when required tests could not be
executed. Explain the limitation instead.
