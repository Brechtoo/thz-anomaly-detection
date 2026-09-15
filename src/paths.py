from pathlib import Path
import os


# ============================================================
# project
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"

# ============================================================
# raw data
# ============================================================

RAW_DATA_ROOT = Path(os.environ.get("THZ_RAW_DATA_DIR", PROJECT_ROOT / "raw_data"))

HOHLRAUM_TESTKÖRPER_1_ROOT = RAW_DATA_ROOT / "Hohlraumtestk1_tr" / "scan"
HOHLRAUM_TESTKÖRPER_2_ROOT = RAW_DATA_ROOT / "Hohlraumtestk2_tr" / "scan"
HOHLRAUM_TESTKÖRPER_3_ROOT = RAW_DATA_ROOT / "Hohlraumtestk3_tr" / "scan"

LABELS_1_ROOT = RAW_DATA_ROOT / "Hohlraumtestk1_tr" / "hohltestk1_tr_markArea_XYList2.txt"
LABELS_2_ROOT = RAW_DATA_ROOT / "Hohlraumtestk2_tr" / "hohltestk2_tr_markArea_XYList.txt"
LABELS_3_ROOT = RAW_DATA_ROOT / "Hohlraumtestk3_tr" / "hohltestk3_tr_markArea_XYList.txt"

# ============================================================
# measurements
# ============================================================

MEASUREMENTS_ROOT = RESULTS_ROOT / "measurements"
DATA1_LABELED_ROOT = MEASUREMENTS_ROOT / "data1_labeled.csv"
DATA2_LABELED_ROOT = MEASUREMENTS_ROOT / "data2_labeled.csv"
DATA3_LABELED_ROOT = MEASUREMENTS_ROOT / "data3_labeled.csv"
COMBINED_DATASET_ROOT = MEASUREMENTS_ROOT / "combined_dataset.csv"

# ============================================================
# features
# ============================================================

FEATURE_TABLES_ROOT = RESULTS_ROOT / "feature_tables"
FEATURE_BASELINE_ROOT = FEATURE_TABLES_ROOT / "feature_baseline.csv"
FEATURE_EXTENSION_BLOCK1_ROOT = FEATURE_TABLES_ROOT / "feature_extension_block1.csv"
FEATURE_TABLE_ROOT = FEATURE_TABLES_ROOT / "feature_table.csv"

# ============================================================
# experiments
# ============================================================

EXPERIMENTS_ROOT = RESULTS_ROOT / "experiments"
EXTENDED_EXPERIMENT_DIR = EXPERIMENTS_ROOT / "extended_features"
EXTENDED_MODELS_DIR = EXTENDED_EXPERIMENT_DIR / "models"
OPTIMIZATION_DIR = EXTENDED_EXPERIMENT_DIR / "optimization"

# ============================================================
# models
# ============================================================


LR_MODEL_PATH = EXTENDED_MODELS_DIR / "logistic_regression.joblib"
RF_MODEL_PATH = EXTENDED_MODELS_DIR / "random_forest.joblib"
MLP_MODEL_PATH = EXTENDED_MODELS_DIR / "mlp.joblib"
HGB_MODEL_PATH = OPTIMIZATION_DIR / "hgb_optimized.joblib"

# ============================================================
# analysis
# ============================================================

ANALYSIS_RESULTS = RESULTS_ROOT / "analysis"
PERMUTATION_IMPORTANCE_RESULTS = ANALYSIS_RESULTS / "permutation_importance"
BASELINE_ANALYSIS_RESULTS = ANALYSIS_RESULTS / "feature_baseline_analysis" / "stats_and_boxplots"

# ============================================================
# transformer
# ============================================================

TRANSFORMER_RESULTS = RESULTS_ROOT / "transformer"

UNSUPERVISED_RESULTS = RESULTS_ROOT / "unsupervised"
PCA_RESULTS = UNSUPERVISED_RESULTS / "pca"
ISOLATION_FOREST_RESULTS = UNSUPERVISED_RESULTS / "isolation_forest"
AUTOENCODER_RESULTS = UNSUPERVISED_RESULTS / "autoencoder"

# ============================================================
# external tests
# ============================================================

EXTERNAL_SCAN_ROOT = Path(os.environ.get("THZ_EXTERNAL_SCAN_DIR", PROJECT_ROOT / "external_data" / "HolzElement" / "scan"))
EXTERNAL_TEST_RESULTS = RESULTS_ROOT / "external_test"