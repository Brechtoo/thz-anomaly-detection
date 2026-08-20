"""
Training und Evaluation multinomialer logistischer Regression

-> Daten laden
-> Features und Zielvariable trennen
-> Daten bereinigen
-> Train/Test-Split
-> Imputation und Standardisierung
-> Training LR -> Prediction
-> Evaluation

"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV

from src.paths import FEATURE_BASELINE_ROOT, FEATURE_EXTENSION_ROOT

def lr_split_reconstrucion(df, TARGET_COL="label", TEST_SIZE=0.2, RANDOM_STATE=42):
    y = df[TARGET_COL]

    X = df.drop(columns=[TARGET_COL])

    X = X.select_dtypes(include=[np.number]) # nur numerische werte
    X = X.drop(columns=["x", "y"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan) # unendliche werte mit NaN ersetzen

    X = X.dropna(axis=1, how="all") # NaN spalten entfernen

    # ============================================================
    # Train-Test-Split
    # ============================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # ============================================================
    # Einstellungen
    # ============================================================

    DATA_PATH = Path(FEATURE_EXTENSION_ROOT)

    OUTPUT_MD_LR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/docs/overview_LR.md")

    OUTPUT_LR_REPORT_PATH= Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/"
                                "results/logistic_regression/LR_extension_block1_report_GRID_CV.txt")

    JOBLIB_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/logistic_regression/lr_extension_block1_report_GRID_CV.joblib")
    RANDOM_STATE = 42

    TARGET_COL = "label"

    TEST_SIZE = 0.2
    RANDOM_STATE = 42


    # ============================================================
    # Daten laden
    # ============================================================

    df = pd.read_csv(DATA_PATH)

    # ============================================================
    # Features und Zielvariable
    # ============================================================

    y = df[TARGET_COL]

    label_distribution = y.value_counts().sort_index()

    X, X_train, X_test, y_train, y_test = lr_split_reconstrucion(
        df,
        TARGET_COL,
        TEST_SIZE,
        RANDOM_STATE,
    )

    # ============================================================
    # Pipeline
    #
    # 1. Fehlende Werte ersetzen
    # 2. Features standardisieren
    # 3. Multinomiale logistische Regression
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
                "logistic_regression",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # ============================================================
    # Hyperparameter für CV
    # ============================================================

    param_grid = {"logistic_regression__C": [0.01, 0.1, 1, 10, 100], "logistic_regression__class_weight": [None, "balanced"]}

    # ============================================================
    # Grid Search mit Cross Validation
    # ============================================================

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # ============================================================
    # results als grid search config
    # ============================================================

    grid_results = pd.DataFrame(grid_search.cv_results_)

    grid_results = grid_results[
        [
            "param_logistic_regression__C",
            "param_logistic_regression__class_weight",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",]].copy()

    grid_results = grid_results.rename(
        columns={
            "param_logistic_regression__C": "C",
            "param_logistic_regression__class_weight": "class_weight",
            "mean_test_score": "cv_macro_f1_mean",
            "std_test_score": "cv_macro_f1_std",
            "rank_test_score": "rank",
        }
    )

    grid_results["class_weight"] = (grid_results["class_weight"].astype(object).where(grid_results["class_weight"].notna(), "None"))

    grid_results = grid_results.sort_values("rank")

    # ============================================================
    # Vorhersagen
    # ============================================================

    y_pred = best_model.predict(X_test)

    # ============================================================
    # Metriken
    # ============================================================

    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    # ============================================================
    # Confusion Matrix
    # ============================================================

    labels = sorted(y.unique())

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    cm_df = pd.DataFrame(cm, index=[f"true_{label}" for label in labels], columns=[f"pred_{label}" for label in labels])

    # ============================================================
    # Classification Report
    # ============================================================

    cr = classification_report(y_test, y_pred, digits=4)

    #-------------------------------------------------------------
    # criticals
    #-------------------------------------------------------------

    critical_mask = (y_test.isin([1, 2]) & (y_pred == 3))

    critical_errors = critical_mask.sum()

    n_hohlraum = y_test.isin([1, 2]).sum()

    critical_error_rate = (critical_errors / n_hohlraum) * 100

    #-------------------------------------------------------------
    # false alarms
    #-------------------------------------------------------------

    false_alarm_mask = ((y_test == 3) & np.isin(y_pred, [1, 2]))

    false_alarms = false_alarm_mask.sum()

    n_label_3 = (y_test == 3).sum()

    false_alarm_rate = (false_alarms / n_label_3) * 100

    # ============================================================
    # save and write into .md
    # ============================================================

    joblib.dump(best_model, JOBLIB_PATH)

    grid_results_text = grid_results.to_string(index=False,formatters={"cv_macro_f1_mean": lambda x: f"{x:.4f}","cv_macro_f1_std": lambda x: f"{x:.4f}",})

    output_text = f"""
    Multinomiale logistische Regression
    
    Beste Parameter:
    {grid_search.best_params_}
    
    Bester CV Macro F1:
    {grid_search.best_score_:.4f}
    
    ============================================================
    Alle Grid-Search-Ergebnisse
    ============================================================
    
    {grid_results_text}
    
    ============================================================
    Evaluation des besten Modells auf dem Testdatensatz
    ============================================================
    
    Messungen: {len(df)}
    Features: {X.shape[1]} # falls hier fehler auftreten: siehe X im zusammenhang mit der funktion lr_split_reconstruction
    
    Label-Verteilung:
    {label_distribution.to_string()}
    
    Accuracy: {accuracy:.4f}
    Balanced Accuracy: {balanced_accuracy:.4f}
    Macro F1: {macro_f1:.4f}
    Weighted F1: {weighted_f1:.4f}
    
    Kritische Fehler:
    Klasse 1 oder 2 als 3 klassifiziert: {critical_errors}
    Kritische FN-Rate: {critical_error_rate:.2f}%
    
    False Alarms:
    Klasse 3 als 1 oder 2 klassifiziert: {false_alarms}
    False Alarm Rate: {false_alarm_rate:.2f}%
    
    Classification Report:
    {cr}
    
    Confusion Matrix:
    {cm_df}
    """.strip()

    with open(OUTPUT_LR_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Report gespeichert unter:\n{OUTPUT_LR_REPORT_PATH}")


    if not OUTPUT_MD_LR.exists():
        with open(OUTPUT_MD_LR, "w", encoding="utf-8") as f:
            f.write(
                "# Logistische Regression Ergebnisse\n\n"
                "| Approach | Accuracy | Balanced Acc | Macro F1 | "
                "Weighted F1 | Kritische FN | FN-Rate | False Alarms | FA-Rate |\n"
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            )

    with open(OUTPUT_MD_LR, "a", encoding="utf-8") as f:
        f.write(
            f"| LR_extension_block1 GRIDSEARCH with CV "
            f"| {accuracy:.4f} "
            f"| {balanced_accuracy:.4f} "
            f"| {macro_f1:.4f} "
            f"| {weighted_f1:.4f} "
            f"| {critical_errors} "
            f"| {critical_error_rate:.2f}% "
            f"| {false_alarms} "
            f"| {false_alarm_rate:.2f}% |\n"
        )

    print(f"Übersicht geschrieben in:\n{OUTPUT_MD_LR}")
