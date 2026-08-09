from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from src.paths import FEATURE_BASELINE_ROOT, HGB_OUTPUT_ROOT


# ============================================================
# Pfade / Einstellungen
# ============================================================

FEATURE_TABLE_PATH = Path(
    "/results/feature_tables/feature_table_extension.csv"
)

OUTPUT_DIR = Path(
    HGB_OUTPUT_ROOT
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT_PATH = OUTPUT_DIR / "hgb_extension_classification_report_0.8.txt"

TARGET_COL = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# ============================================================
# Schwellenlogik
# ============================================================

P3_THRESHOLD = 0.8
P0_THRESHOLD = 0.60

HOHLRAUM_LABELS = [1, 2]
BAD_MEASUREMENT_LABEL = 0
NO_HOHLRAUM_LABEL = 3

LABELS = [0, 1, 2, 3]

TARGET_NAMES = [
    "0_schlechte_messung",
    "1_hohlraum",
    "2_vielleicht_hohlraum",
    "3_kein_hohlraum",
]


# ============================================================
# Hilfsfunktionen
# ============================================================

def threshold_predict(proba, classes, p3_threshold=0.85, p0_threshold=0.6):
    """
    Eigene Entscheidungslogik:

    1. Nur wenn p(kein Hohlraum) hoch genug ist -> Klasse 3
    2. Sonst wenn p(schlechte Messung) hoch genug ist -> Klasse 0
    3. Sonst lieber Hohlraum/Verdacht -> Klasse 1 oder 2
    """

    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    preds = []

    for row in proba:
        p0 = row[class_to_idx[BAD_MEASUREMENT_LABEL]] if BAD_MEASUREMENT_LABEL in class_to_idx else 0.0
        p3 = row[class_to_idx[NO_HOHLRAUM_LABEL]] if NO_HOHLRAUM_LABEL in class_to_idx else 0.0

        if p3 >= p3_threshold:
            preds.append(NO_HOHLRAUM_LABEL)

        elif p0 >= p0_threshold:
            preds.append(BAD_MEASUREMENT_LABEL)

        else:
            available_hohlraum_labels = [
                label for label in HOHLRAUM_LABELS if label in class_to_idx
            ]

            if available_hohlraum_labels:
                best_label = max(
                    available_hohlraum_labels,
                    key=lambda label: row[class_to_idx[label]]
                )
                preds.append(best_label)
            else:
                preds.append(classes[np.argmax(row)])

    return np.array(preds)


def get_critical_error_stats(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_hohlraum = np.isin(y_true, HOHLRAUM_LABELS)
    pred_no_hohlraum = y_pred == NO_HOHLRAUM_LABEL

    critical_errors = true_hohlraum & pred_no_hohlraum

    n_true_hohlraum = true_hohlraum.sum()
    n_critical = critical_errors.sum()

    critical_rate = n_critical / n_true_hohlraum if n_true_hohlraum > 0 else np.nan

    true_no_hohlraum = y_true == NO_HOHLRAUM_LABEL
    pred_hohlraum = np.isin(y_pred, HOHLRAUM_LABELS)

    false_alarms = true_no_hohlraum & pred_hohlraum

    n_true_no_hohlraum = true_no_hohlraum.sum()
    n_false_alarms = false_alarms.sum()

    false_alarm_rate = (
        n_false_alarms / n_true_no_hohlraum
        if n_true_no_hohlraum > 0
        else np.nan
    )

    return n_critical, critical_rate, n_false_alarms, false_alarm_rate


def create_report_section(y_true, y_pred, title):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        target_names=TARGET_NAMES,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2", "true_3"],
        columns=["pred_0", "pred_1", "pred_2", "pred_3"],
    )

    n_critical, critical_rate, n_false_alarms, false_alarm_rate = (
        get_critical_error_stats(y_true, y_pred)
    )

    section = f"""
{title}

Accuracy:          {accuracy:.4f}
Balanced Accuracy: {balanced_acc:.4f}
Macro F1:          {macro_f1:.4f}
Weighted F1:       {weighted_f1:.4f}

Kritische Fehler:
Echte Hohlraum/Verdacht als Kein Hohlraum: {n_critical}
Kritische FN-Rate: {critical_rate:.4f}

False Alarms:
Echte Kein-Hohlraum als Hohlraum/Verdacht: {n_false_alarms}
False-Alarm-Rate: {false_alarm_rate:.4f}

Classification Report:
{report}

Confusion Matrix:
{cm_df.to_string()}
""".strip()

    return section


# ============================================================
# Daten laden
# ============================================================

df = pd.read_csv(FEATURE_TABLE_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Target-Spalte '{TARGET_COL}' nicht gefunden.")


# ============================================================
# Features / Target vorbereiten
# ============================================================

y = df[TARGET_COL]

cols_to_drop = [
    TARGET_COL,
    "file_path",
    "x",
    "y",
    "source_dataset",
    "instance_id"
]

X = df.drop(columns=[TARGET_COL])
X = X.select_dtypes(include=[np.number])
X = X.drop(columns=["x", "y"], errors="ignore")

X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    raise ValueError("Keine numerischen Features gefunden.")


# ============================================================
# Label-Verteilung
# ============================================================

label_distribution = y.value_counts().sort_index()


# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# Modell trainieren
# ============================================================

model = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.03,
    max_iter=800,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    random_state=RANDOM_STATE,
)

sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

model.fit(X_train, y_train, sample_weight=sample_weight)


# ============================================================
# Vorhersagen
# ============================================================

proba_test = model.predict_proba(X_test)

y_pred_raw = model.classes_[np.argmax(proba_test, axis=1)]

y_pred_threshold = threshold_predict(
    proba=proba_test,
    classes=model.classes_,
    p3_threshold=P3_THRESHOLD,
    p0_threshold=P0_THRESHOLD,
)


# ============================================================
# Report erstellen
# ============================================================

raw_section = create_report_section(
    y_true=y_test,
    y_pred=y_pred_raw,
    title="TEST — HistGradientBoosting raw argmax",
)

threshold_section = create_report_section(
    y_true=y_test,
    y_pred=y_pred_threshold,
    title="TEST — HistGradientBoosting mit Entscheidungsschwellen",
)

output_text = f"""
HistGradientBoosting Classification Report

Messungen: {len(df)}
Features:  {X.shape[1]}

Gelernte Klassen: {list(model.classes_)}
Anzahl Boosting-Iterationen: {model.n_iter_}

P3_THRESHOLD: {P3_THRESHOLD}
P0_THRESHOLD: {P0_THRESHOLD}

Label-Verteilung gesamt:
{label_distribution.to_string()}

{"=" * 80}

{raw_section}

{"=" * 80}

{threshold_section}
""".strip()


# ============================================================
# Report ausgeben und speichern
# ============================================================

with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(output_text)

print()
print(f"Report gespeichert unter:")
print(OUTPUT_REPORT_PATH)