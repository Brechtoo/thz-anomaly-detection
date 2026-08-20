from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


def mlp_split_reconstrucion(df, TARGET_COL="label"):
    y = df[TARGET_COL]

    X = df.drop(columns=[TARGET_COL])

    # nur numerische Features
    X = X.select_dtypes(include=[np.number])

    # Koordinaten nicht als Features verwenden
    X = X.drop(columns=["x", "y"], errors="ignore")

    # Inf-Werte behandeln
    X = X.replace([np.inf, -np.inf], np.nan)

    # komplett leere Spalten entfernen
    X = X.dropna(axis=1, how="all")


    # ============================================================
    # train-, test-split
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    return X, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # ============================================================
    # settings
    # ============================================================

    DATA_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/feature_tables/feature_extension_block1.csv")

    OUTPUT_DIR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/nn")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_REPORT_PATH = OUTPUT_DIR / "nn_extended_features_block1.txt"

    JOBLIB_PATH = OUTPUT_DIR / "nn_extended_features_block1.joblib"

    MARKDOWN_MLP_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/docs/overview_mlp.md")

    TARGET_COL = "label"

    # ============================================================
    # load data
    # ============================================================

    df = pd.read_csv(DATA_PATH)

    X, X_train, X_test, y_train, y_test = mlp_split_reconstrucion(df)

    # ============================================================
    # modell pipeline
    # ============================================================

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "neural_network",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    batch_size=256,
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )


    # ============================================================
    # training
    # ============================================================
    print("train...")

    model.fit(X_train, y_train)


    # ============================================================
    # prediction
    # ============================================================

    y_pred = model.predict(X_test)


    # ============================================================
    # evaluation
    # ============================================================

    accuracy = accuracy_score(y_test, y_pred)

    balanced_acc = balanced_accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    weighted_f1 = f1_score(y_test, y_pred, average='weighted')

    critical_mask = y_test.isin([1, 2])
    false_alarm_mask = y_test == 3

    critical_errors = (
        critical_mask
        & (y_pred == 3)
    ).sum()

    false_alarms = (
        false_alarm_mask
        & np.isin(y_pred, [1, 2])
    ).sum()

    critical_total = critical_mask.sum()
    no_cavity_total = false_alarm_mask.sum()

    critical_fn_rate = ((critical_errors / critical_total) * 100)

    false_alarm_rate = (false_alarms / no_cavity_total) * 100

    cm = confusion_matrix(y_test,y_pred,labels=[0, 1, 2, 3])

    cm_df = pd.DataFrame(
            cm,
            index=["true_0", "true_1", "true_2", "true_3"],
            columns=["pred_0", "pred_1", "pred_2", "pred_3"],
        )

    class_report = classification_report(y_test, y_pred, digits=4)

    # ============================================================
    # output
    # ============================================================

    output_text = f"""
    Multilayer Perceptron Neural Network:
    
    Messungen: {len(df)}
    Anzahl Features: {len(X.columns)}
    
    Confusion Matrix:
    {cm_df.to_string()}
    
    Classification Report:
    {class_report}
    
    Accuracy: {accuracy:.4f}
    Balanced Accuracy: {balanced_acc:.4f}
    Macro F1: {macro_f1:.4f}
    Weighted F1: {weighted_f1:.4f}
    
    Kritische Fehler: {critical_errors} / {critical_total} ({critical_fn_rate:.2f})
    False Alarms: {false_alarms} / {no_cavity_total} ({false_alarm_rate:.2f})
    
    """.strip()

    # ============================================================
    # save and write into .md
    # ============================================================

    joblib.dump(model, JOBLIB_PATH)

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)

    print()
    print(f"Report gespeichert unter:\n{OUTPUT_REPORT_PATH}")

    if not MARKDOWN_MLP_PATH.exists():
        with open(MARKDOWN_MLP_PATH, "w", encoding="utf-8") as f:
            f.write(
                "# MLP Ergebnisse\n\n"
                "| Approach | Accuracy | Balanced Acc | Macro F1 | "
                "Weighted F1 | Kritische FN | FN-Rate | False Alarms | FA-Rate |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            )

    with open(MARKDOWN_MLP_PATH, "a", encoding="utf-8") as f:
        f.write(
            f"| mlp_features_extension_block1 "
            f"| {accuracy:.4f} "
            f"| {balanced_acc:.4f} "
            f"| {macro_f1:.4f} "
            f"| {weighted_f1:.4f} "
            f"| {critical_errors} "
            f"| {critical_fn_rate:.2f}% "
            f"| {false_alarms} "
            f"| {false_alarm_rate:.2f}% |\n"
        )

    print(f"markdown gespeichert unter: {MARKDOWN_MLP_PATH}")
