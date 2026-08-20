from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, classification_report

from src.hgb.hgb import hgb_split_reconstrucion, threshold_predict
from src.logistic_regression.multinomial_LR import lr_split_reconstrucion
from src.nn.mlp import mlp_split_reconstrucion
from src.paths import FEATURE_EXTENSION_ROOT
from src.random_forest.rf import rf_split_reconstrucion, merge_treshold_logic

# ============================================================
# settings
# ============================================================

print("setting up ...")

INPUT_PATH = FEATURE_EXTENSION_ROOT

OUTPUT_DIR = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/evaluation_for_1_red_3_green_0_2_yellow")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_REPORT_PATH = OUTPUT_DIR / "report_new_metrices.txt"
OUTPUT_REPORT_PATH.touch()

TARGET_COL = "label"

LR_MODEL_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/logistic_regression/lr_extension_block1_report_GRID_CV.joblib")

RF_BEST_FN_MODEL_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/random_forest/rf_extension_block1/rf_extension_block1_False_True.joblib")

HGB_MODEL_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/hgb/hgb_extension_block1/hgb_extension_block1.joblib")

MLP_MODEL_PATH = Path("/Users/flo/Desktop/Isenhagen/thz-anomaly-supervised/results/nn/nn_extended_features_block1")

# ============================================================
# load data
# ============================================================

print("load data ...")

df = pd.read_csv(INPUT_PATH)

df_lr = df
X_lr, X_train_lr, X_test_lr, y_train_lr, y_test_lr = lr_split_reconstrucion(df, "label")

df_rf = df
X_rf, X_train_rf, X_test_rf, y_train_rf, y_test_rf = rf_split_reconstrucion(df, "label")

df_hgb = df
X_hgb, X_train_hgb, X_test_hgb, y_train_hgb, y_test_hgb = hgb_split_reconstrucion(df, "label")

df_mlp = df
X_mlp, X_train_mlp, X_test_mlp, y_train_mlp, y_test_mlp = mlp_split_reconstrucion(df, "label")

# ============================================================
# load model
# ============================================================

print("load model ...")

lr = joblib.load(LR_MODEL_PATH)

hgb = joblib.load(HGB_MODEL_PATH)

rf = joblib.load(RF_BEST_FN_MODEL_PATH)

mlp = joblib.load(MLP_MODEL_PATH)

# ============================================================
# predicts
# ============================================================

print("prediction ...")

lr_pred = lr.predict(X_test_lr)

hgb_proba = hgb.predict_proba(X_test_hgb)

hgb_raw_pred = hgb.predict(X_test_hgb)

hgb_threshold_pred = threshold_predict(
    proba=hgb_proba,
    classes=hgb.classes_,
    p3_threshold=0.85,
    p0_threshold=0.6,
)

rf_baseline_pred = rf.predict(X_test_rf)

rf_treshold_pred = merge_treshold_logic(rf, X_test_rf, False, True)

mlp_pred = mlp.predict(X_test_mlp)

# ============================================================
# metriken
# ============================================================

print("calculate metrics ...")

def calculate_metrics(model_name, y_true, y_pred):
    """
    recall class 1: anteil echter hohlräume, die als solche erkannt werden
    criticals: anteil echter klasse 1 messungen, die als klasse 3 predicted werden
    uncertainity1: unsicherheitsrate von klasse 1 auf klasse 0 oder 2
    recall klasse 3: siehe oben
    false alarms: echte klasse 3 messungen, die als klasse 1 predicted werden
    uncertainity3: unsicherheitsrate von klasse 3 auf klasse 0 oder 2

    coverage: anteil der klasse 1 und 3 predicts, die eindeutig und richtig sind
    selective accuracy: genauigkeit nur unter diesen predicts


    return: metrics per model
    """

    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1", "true_2","true_3"], columns=["pred_0", "pred_1", "pred_2", "pred_3"])

    report = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        labels=[0, 1, 2, 3],
        target_names=["0 - schlechte Messung", "1 - Hohlraum", "2 - vielleicht Hohlraum", "3 - Kein Hohlraum"])

    critical_errors = cm[1, 3]
    class1_total = np.sum(y_true == 1)
    critical_fn_rate = (critical_errors / class1_total) * 100

    false_alarms = cm[3, 1]
    class3_total = np.sum(y_true == 3)
    false_alarm_rate = (false_alarms / class3_total) * 100

    recall_class1 = (cm[1, 1] / cm[1, :].sum()) * 100
    recall_class3 = (cm[3, 3] / cm[3, :].sum()) * 100

    uncertainity1 = ((cm[1, 0] + cm[1, 2]) / cm[1, :].sum()) * 100
    uncertainity3 = ((cm[3, 0] + cm[3, 2]) / cm[3, :].sum()) * 100

    output_text = f"""
    {model_name}

    Accuracy:          {accuracy:.4f}
    Balanced Accuracy: {balanced_acc:.4f}
    Macro F1:          {macro_f1:.4f}
    Weighted F1:       {weighted_f1:.4f}

    Anteil echter Klasse 1 Messungen, die als Klasse 3 vorhergesagt werden
    Kritische Fehler: {critical_errors}
    Kritische FN-Rate: {critical_fn_rate:.2f}%

    Anteil echter Klasse 3 Messungen, die als Klasse 1 vorhergesagt werden
    False Alarms: {false_alarms}
    False-Alarm-Rate: {false_alarm_rate:.2f}%
    
    Recall Klasse 1: {recall_class1:.2f}%
    Recall Klasse 3: {recall_class3:.2f}%
    
    Unsicherheitsrate Klasse 1 zu 0 oder 2: {uncertainity1:.2f}%
    Unsicherheitsrate Klasse 3 zu 0 oder 2: {uncertainity3:.2f}%
    
    Classification Report:
    {report}

    Confusion Matrix:
    {cm_df.to_string()}
    """.strip()

    return output_text

lr_new_metrics = calculate_metrics("Logistic Regression", y_test_lr, lr_pred)

hgb_raw_new_metrics = calculate_metrics("HistGradientBoosting raw argmax", y_test_hgb, hgb_raw_pred)

hgb_threshold_new_metrics = calculate_metrics("HistGradientBoosting Thresholds", y_test_hgb, hgb_threshold_pred)

rf_baseline_new_metrics = calculate_metrics("Random Forest Baseline", y_test_rf, rf_baseline_pred)

rf_treshold_new_metrics = calculate_metrics("Random Forest Thresholds", y_test_rf, rf_treshold_pred)

mlp_new_metrics = calculate_metrics("MLP", y_test_mlp, mlp_pred)


with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
    f.write(lr_new_metrics + "\n\n")
    f.write(hgb_raw_new_metrics + "\n\n")
    f.write(hgb_threshold_new_metrics + "\n\n")
    f.write(rf_baseline_new_metrics + "\n\n")
    f.write(rf_treshold_new_metrics + "\n\n")
    f.write(mlp_new_metrics + "\n")

print(f"Report gespeichert unter: {OUTPUT_REPORT_PATH}")
