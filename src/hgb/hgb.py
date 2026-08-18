from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
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
# Schwellenlogik
# ============================================================

P3_THRESHOLD = 0.8 # ab welcher sicherheit label 3 gewählt wird
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
# Pfade / Einstellungen
# ============================================================

FEATURE_TABLE_PATH = Path(FEATURE_BASELINE_ROOT)

OUTPUT_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/hgb")

OUTPUT_DIR = OUTPUT_PATH / "hgb_baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT_PATH = OUTPUT_DIR / f"hgb_baseline_report.txt"

JOBLIB_PATH = Path(OUTPUT_DIR / "hgb_baseline.joblib")

TARGET_COL = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.2

EXPERIMENT_NAME = "hgb_baseline"
MARKDOWN_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/docs/overview_hgb.md")

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

    class_to_idx = {label: idx for idx, label in enumerate(classes)} # labels zu spalten zuordnen

    preds = []

    for row in proba:
        p0 = row[class_to_idx[BAD_MEASUREMENT_LABEL]] if BAD_MEASUREMENT_LABEL in class_to_idx else 0.0
        p3 = row[class_to_idx[NO_HOHLRAUM_LABEL]] if NO_HOHLRAUM_LABEL in class_to_idx else 0.0

        if p3 >= p3_threshold:
            preds.append(NO_HOHLRAUM_LABEL)

        elif p0 >= p0_threshold:
            preds.append(BAD_MEASUREMENT_LABEL)

        else: # wenn weder klasse 0 noch klasse 3 unter den bedingungen von oben
            available_hohlraum_labels = [
                label for label in HOHLRAUM_LABELS if label in class_to_idx # nur noch zwischen klasse 1 und 2 entscheiden
            ]

            if available_hohlraum_labels:
                best_label = max(
                    available_hohlraum_labels,
                    key=lambda label: row[class_to_idx[label]] # die klasse mit der höheren wahrscheinlichkeit
                )
                preds.append(best_label)
            else:
                preds.append(classes[np.argmax(row)])

    return np.array(preds)


def get_critical_error_stats(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    true_hohlraum = np.isin(y_true, HOHLRAUM_LABELS) # maske ist überall true wo das echte label 1 oder 2 ist
    pred_no_hohlraum = y_pred == NO_HOHLRAUM_LABEL # überall true wo klasse 3 vorhergesagt wurde

    critical_errors = true_hohlraum & pred_no_hohlraum # kritischer fehler, wenn echtes label 1 oder 2 ist, aber vorhersage 3

    n_true_hohlraum = true_hohlraum.sum() # anzahl tatsächlicher klassen 1 und 2
    n_critical = critical_errors.sum() # anzahl fälschlicherweise als 3 klassifizierter fälle

    critical_rate = (n_critical / n_true_hohlraum) * 100 if n_true_hohlraum > 0 else np.nan # kritische fehlerrate

    true_no_hohlraum = y_true == NO_HOHLRAUM_LABEL # überall true wo das echte label 3 ist
    pred_hohlraum = np.isin(y_pred, HOHLRAUM_LABELS) # überall true wo klasse 1 oder 2 vorhergesagt wurde

    false_alarms = true_no_hohlraum & pred_hohlraum # falscher alarm, wenn echtes label 3, aber vorhersage 1 oder 2

    n_true_no_hohlraum = true_no_hohlraum.sum() # anzahl tatsächlicher klassen 3
    n_false_alarms = false_alarms.sum() # anzahl fälschlicherweise als 1 oder 2 klassifizierter fälle

    false_alarm_rate = (
        (n_false_alarms / n_true_no_hohlraum) * 100
        if n_true_no_hohlraum > 0
        else np.nan
    ) # false-alarm rate

    return n_critical, critical_rate, n_false_alarms, false_alarm_rate


def create_report_section(y_true, y_pred, title):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred) # mittelwert der recall-werte aller klassen
    macro_f1 = f1_score(y_true, y_pred, average="macro") # ungewichteter mittelwert der F1-scores jeder klasse
    weighted_f1 = f1_score(y_true, y_pred, average="weighted") # häufigere klassen erhalten gößeres gewicht

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
1 oder 2 als 3 klassifiziert: {n_critical}
Kritische FN-Rate: {critical_rate:.2f}%

False Alarms:
3 als 1 oder 2 klassifiziert: {n_false_alarms}
False-Alarm-Rate: {false_alarm_rate:.2f}%

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
X = X.select_dtypes(include=[np.number]) # nur numerische werte
X = X.drop(columns=["x", "y"], errors="ignore")

X = X.replace([np.inf, -np.inf], np.nan) # imputation von NaN werten
X = X.dropna(axis=1, how="all") # falls pro zeile einige sinnvollen werte unter NaN vorkommen, diese behalten

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

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)


# ============================================================
# Modell trainieren
# ============================================================
print("training model...")


hgb = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.03,
    max_iter=800,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True,
    validation_fraction=0.15, # effektiv 85% der 80% trainingsdaten verwenden
    n_iter_no_change=30,
    random_state=RANDOM_STATE,
)

sample_weight = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
) # klasse 3 niedrigere gewichte, weil häufigeres vorkommen

hgb.fit(X_train, y_train, sample_weight=sample_weight)


# ============================================================
# Vorhersagen
# ============================================================

proba_test = hgb.predict_proba(X_test) # bei 10 000 testmessungen ein array der form (10 000, 4), eine zeile [0.04, 0.2, 0.16, 0.6]

y_pred_raw = hgb.classes_[np.argmax(proba_test, axis=1)] # sucht in jeder zeile die position des größten wertes

y_pred_threshold = threshold_predict(
    proba=proba_test,
    classes=hgb.classes_,
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

Gelernte Klassen: {list(hgb.classes_)}
Anzahl Boosting-Iterationen: {hgb.n_iter_}

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

# ============================================================
# Ergebnisse für Markdown-Tabelle vorbereiten
# ============================================================

# Raw Argmax
raw_accuracy = accuracy_score(y_test, y_pred_raw)
raw_balanced_acc = balanced_accuracy_score(y_test, y_pred_raw)
raw_macro_f1 = f1_score(y_test, y_pred_raw, average="macro")
raw_weighted_f1 = f1_score(y_test, y_pred_raw, average="weighted")

(
    raw_critical_errors,
    raw_critical_fn_rate,
    raw_false_alarms,
    raw_false_alarm_rate,
) = get_critical_error_stats(y_test, y_pred_raw)


# Thresholds
threshold_accuracy = accuracy_score(y_test, y_pred_threshold)
threshold_balanced_acc = balanced_accuracy_score(y_test, y_pred_threshold)
threshold_macro_f1 = f1_score(y_test, y_pred_threshold, average="macro")
threshold_weighted_f1 = f1_score(
    y_test,
    y_pred_threshold,
    average="weighted"
)

(
    threshold_critical_errors,
    threshold_critical_fn_rate,
    threshold_false_alarms,
    threshold_false_alarm_rate,
) = get_critical_error_stats(y_test, y_pred_threshold)


# ============================================================
# Markdown-Ergebnisübersicht speichern
# ============================================================

joblib.dump(hgb, JOBLIB_PATH)

if not MARKDOWN_PATH.exists():
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# HistGradientBoosting Ergebnisse\n\n"
            "| Experiment | Prediction | P3 | P0 | Iterationen | "
            "Accuracy | Balanced Acc | Macro F1 | Weighted F1 | "
            "Kritische FN | FN-Rate | False Alarms | FA-Rate |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|\n"
        )


with open(MARKDOWN_PATH, "a", encoding="utf-8") as f:

    # Raw Argmax
    f.write(
        f"| {EXPERIMENT_NAME} "
        f"| Raw Argmax "
        f"| - "
        f"| - "
        f"| {hgb.n_iter_} "
        f"| {raw_accuracy:.4f} "
        f"| {raw_balanced_acc:.4f} "
        f"| {raw_macro_f1:.4f} "
        f"| {raw_weighted_f1:.4f} "
        f"| {raw_critical_errors} "
        f"| {raw_critical_fn_rate:.2f}% "
        f"| {raw_false_alarms} "
        f"| {raw_false_alarm_rate:.2f}% |\n"
    )

    # Thresholds
    f.write(
        f"| {EXPERIMENT_NAME} "
        f"| Thresholds "
        f"| {P3_THRESHOLD:.2f} "
        f"| {P0_THRESHOLD:.2f} "
        f"| {hgb.n_iter_} "
        f"| {threshold_accuracy:.4f} "
        f"| {threshold_balanced_acc:.4f} "
        f"| {threshold_macro_f1:.4f} "
        f"| {threshold_weighted_f1:.4f} "
        f"| {threshold_critical_errors} "
        f"| {threshold_critical_fn_rate:.2f}% "
        f"| {threshold_false_alarms} "
        f"| {threshold_false_alarm_rate:.2f}% |\n"
    )