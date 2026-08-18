"""
Training und Evaluation eines RF-Rlassifikators

Zu Beginn festlegen, ob Klassen 1 und 2 zusammengelegt werden sollen, sonst: RF-Baseline


- mergen?, thresholds?, input path, output path
- bereitet Features und Labels vor,
- teilt die Daten in Training und Test,
- trainiert einen Random Forest,
- berechnet Evaluationsmetriken,
- speichert einen Ergebnisbericht.
"""

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

import joblib

from src.paths import FEATURE_BASELINE_ROOT

# ============================================================
# settings
# ============================================================

MERGE_HOHLRAUM_CLASSES = False # wenn label 1 und 2 zusammengelegt werden sollen
USE_THRESHOLDS = True # wenn erst ab gewissen thresholds klassen vorhersagen vorgenommen werden sollen

P3_THRESHOLD = 0.85
P0_THRESHOLD = 0.60

RANDOM_STATE = 42

FEATURE_CSV_PATH = FEATURE_BASELINE_ROOT
OUTPUT_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/random_forest")

OUTPUT_DIR = Path(OUTPUT_PATH / "rf_merge_and_thresholds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT_PATH = (OUTPUT_DIR / f"rf_{MERGE_HOHLRAUM_CLASSES}_{USE_THRESHOLDS}.txt")

MARKDOWN_RF_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/docs/overview_rf.md")

JOBLIB_PATH = Path(OUTPUT_DIR / f"rf_{MERGE_HOHLRAUM_CLASSES}_{USE_THRESHOLDS}.joblib")

TARGET_COL = "label"

# ============================================================
# load data, use sample
# ============================================================

df = pd.read_csv(FEATURE_CSV_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"Spalte '{TARGET_COL}' wurde nicht gefunden.")

# USE_SAMPLE = True -> verwende sample der größe SAMPLE_SIZE
USE_SAMPLE = False
SAMPLE_SIZE = 10_000

if USE_SAMPLE:
    df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df[TARGET_COL], # klassenverteilung bleibt im sample ungefähr gleich
        random_state=42,
    )

# ============================================================
# labels vorbereiten
# ============================================================

# 0 = schlechte Messung
# 1 = Hohlraum oder vielleicht Hohlraum
# 3 = kein Hohlraum

ORIGINAL_TARGET_COL = "label"

if MERGE_HOHLRAUM_CLASSES:
    df["label_conservative"] = df[ORIGINAL_TARGET_COL].replace({
        0: 0,
        1: 1,
        2: 1,
        3: 3,
    })
    TARGET_COL = "label_conservative"
else:
    TARGET_COL = ORIGINAL_TARGET_COL

# ============================================================
# training (features) vorbereiten
# ============================================================

y = df[TARGET_COL]

cols_to_drop = [
    "label",
    "label_conservative",
    "file_path",
    "x",
    "y",
    "source_dataset",
    "instance_id",
]

X = df.drop(columns=cols_to_drop, errors="ignore")

X = X.select_dtypes(include=[np.number]) # nur numerische werte

X = X.replace([np.inf, -np.inf], np.nan) # NaN werte für spätere imputation vorbereiten
X = X.dropna(axis=1, how="all")

if X.shape[1] == 0:
    raise ValueError("Keine numerischen Features gefunden.")

# ============================================================
# Label-verteilung
# ============================================================

label_distribution = y.value_counts().sort_index() # für den report später


# ============================================================
# split (training/test)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

imputer = SimpleImputer(strategy="median")

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ============================================================
# training
# ============================================================

rf = RandomForestClassifier(
    n_estimators=100, # 100 decision trees
    max_depth=None, # max baumtiefe
    min_samples_split=2, # knoten darf aufgeteilt werden, sobald er mindestens zwei messungen enthält
    min_samples_leaf=1, # blatt darf nur eine einzige trainingsmessung enthalten
    class_weight="balanced", # seltene klassen erhalten höheres gewicht
    random_state=42,
    n_jobs=-1,
)
print("training...")
rf.fit(X_train, y_train) # objekt der klasse RandomForestClassifier


# ============================================================
# prediction
# ============================================================

if USE_THRESHOLDS:
    probabilities = rf.predict_proba(X_test)
    classes = rf.classes_

    class_to_idx = {
        label: idx
        for idx, label in enumerate(classes)
    }

    preds = []

    for row in probabilities:

        # Klasse 3 nur bei ausreichend hoher Sicherheit
        if row[class_to_idx[3]] >= P3_THRESHOLD:
            preds.append(3)

        # Klasse 0 nur bei ausreichend hoher Sicherheit
        elif row[class_to_idx[0]] >= P0_THRESHOLD:
            preds.append(0)

        else:
            if MERGE_HOHLRAUM_CLASSES:
                # Klassen 1 und 2 wurden bereits zusammengelegt
                preds.append(1)

            else:
                # Zwischen Hohlraum (1) und Verdacht (2)
                # die wahrscheinlichere Klasse wählen
                if row[class_to_idx[1]] >= row[class_to_idx[2]]:
                    preds.append(1)
                else:
                    preds.append(2)

    y_pred = np.array(preds)

else:
    y_pred = rf.predict(X_test)


# ============================================================
# auswertung
# ============================================================

if MERGE_HOHLRAUM_CLASSES: labels = [0, 1, 3]
else: labels = [0, 1, 2, 3]

accuracy = accuracy_score(y_test, y_pred)
balanced_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")
weighted_f1 = f1_score(y_test, y_pred, average="weighted") # könnte hoch sein, obowhl klasse 2 schlecht erkannt wird

if MERGE_HOHLRAUM_CLASSES:
    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[
            "0_schlechte_messung",
            "1_hohlraum_oder_verdacht",
            "3_kein_hohlraum",
        ],
    )
else:
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=[
            "0_schlechte_messung",
            "1_hohlraum",
            "2_vielleicht_hohlraum",
            "3_kein_hohlraum",
        ],
    )

cm = confusion_matrix(y_test, y_pred, labels=labels)

if MERGE_HOHLRAUM_CLASSES:
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_3"],
        columns=["pred_0", "pred_1", "pred_3"],
    )
else:
    cm_df = pd.DataFrame(
        cm,
        index=["true_0", "true_1", "true_2", "true_3"],
        columns=["pred_0", "pred_1", "pred_2", "pred_3"],
    )


# ============================================================
# criticals und false alarms
# ============================================================

y_test_array = np.array(y_test)
y_pred_array = np.array(y_pred)

y_test_array = np.asarray(y_test)
y_pred_array = np.asarray(y_pred)

if MERGE_HOHLRAUM_CLASSES:

    # echte Klasse 1 -> fälschlich Klasse 3
    critical_errors = np.sum(
        (y_test_array == 1) & (y_pred_array == 3)
    )

    critical_total = np.sum(y_test_array == 1)

    critical_fn_rate = (
        critical_errors / critical_total * 100
        if critical_total > 0
        else 0
    )

    # echte Klasse 3 -> fälschlich Klasse 1
    false_alarms = np.sum(
        (y_test_array == 3) & (y_pred_array == 1)
    )

    false_alarm_total = np.sum(y_test_array == 3)

    false_alarm_rate = (
        false_alarms / false_alarm_total * 100
        if false_alarm_total > 0
        else 0
    )

else:

    # echte Klasse 1 oder 2 -> fälschlich Klasse 3
    critical_errors = np.sum(
        np.isin(y_test_array, [1, 2])
        & (y_pred_array == 3)
    )

    critical_total = np.sum(
        np.isin(y_test_array, [1, 2])
    )

    critical_fn_rate = (
        critical_errors / critical_total * 100
        if critical_total > 0
        else 0
    )

    # echte Klasse 3 -> fälschlich Klasse 1 oder 2
    false_alarms = np.sum(
        (y_test_array == 3)
        & np.isin(y_pred_array, [1, 2])
    )

    false_alarm_total = np.sum(y_test_array == 3)

    false_alarm_rate = (
        false_alarms / false_alarm_total * 100
        if false_alarm_total > 0
        else 0
    )


# ============================================================
# report: evaluationsmetriken, confusion matrix
# ============================================================

if MERGE_HOHLRAUM_CLASSES:
    critical_description = "Klasse 1 (Hohlraum/Verdacht) als 3"
    false_alarm_description = "Klasse 3 als 1 (Hohlraum/Verdacht)"
else:
    critical_description = "Klasse 1 oder 2 als 3"
    false_alarm_description = "Klasse 3 als 1 oder 2"

output_text = f"""
Random Forest Baseline

Messungen: {len(df)}
Features:  {X.shape[1]}

Label-Verteilung gesamt:
{label_distribution.to_string()}

Accuracy:          {accuracy:.4f}
Balanced Accuracy: {balanced_acc:.4f}
Macro F1:          {macro_f1:.4f}
Weighted F1:       {weighted_f1:.4f}

Kritische Fehler:
{critical_description}: {critical_errors}
Kritische FN-Rate: {critical_fn_rate:.2f}%

False Alarms:
{false_alarm_description}: {false_alarms}
False-Alarm-Rate: {false_alarm_rate:.2f}%

Classification Report:
{report}

Confusion Matrix:
{cm_df.to_string()}
""".strip()


# ============================================================
# save and write into .md
# ============================================================

joblib.dump(rf, JOBLIB_PATH)

with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(output_text)

print()
print(f"Report gespeichert unter:\n{OUTPUT_REPORT_PATH}")

if not MARKDOWN_RF_PATH.exists():
    with open(MARKDOWN_RF_PATH, "w", encoding="utf-8") as f:
        f.write(
            "# Random Forest Ergebnisse\n\n"
            "| Approach | Accuracy | Balanced Acc | Macro F1 | "
            "Weighted F1 | Kritische FN | FN-Rate | False Alarms | FA-Rate |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        )

with open(MARKDOWN_RF_PATH, "a", encoding="utf-8") as f:
    f.write(
        f"| EXTENDED: M {MERGE_HOHLRAUM_CLASSES}, T {USE_THRESHOLDS} "
        f"| {accuracy:.4f} "
        f"| {balanced_acc:.4f} "
        f"| {macro_f1:.4f} "
        f"| {weighted_f1:.4f} "
        f"| {critical_errors} "
        f"| {critical_fn_rate:.2f}% "
        f"| {false_alarms} "
        f"| {false_alarm_rate:.2f}% |\n"
    )