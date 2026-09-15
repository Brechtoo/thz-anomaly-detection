from pathlib import Path

import pandas as pd

from src.features.extract_all_features import extract_all_features
from src.paths import COMBINED_DATASET_ROOT, FEATURE_TABLE_ROOT

# ============================================================
# settings
# ============================================================

INPUT_PATH = COMBINED_DATASET_ROOT

OUTPUT_PATH = FEATURE_TABLE_ROOT

MAX_MEASUREMENTS = None

# ============================================================
# feature table creation
# ============================================================

def create_feature_table(df):

    rows = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):

        file_path = row["file_path"]

        features = extract_all_features(file_path)

        # ------------------------------------------------
        # metadata
        # ------------------------------------------------

        features["file_path"] = file_path

        for column in [
            "label",
            "x",
            "y",
            "source_dataset",
            "instance_id"
        ]:
            if column in df.columns:
                features[column] = row[column]

        rows.append(features)

        if i % 1000 == 0:
            print(f"{i}/{len(df)} messungen verarbeitet...")

    feature_df = pd.DataFrame(rows)

    return feature_df


# ============================================================
# main
# ============================================================

def main():

    df = pd.read_csv(INPUT_PATH)

    if MAX_MEASUREMENTS is not None:

        df = df.sample(n=MAX_MEASUREMENTS, random_state=42).reset_index(drop=True)

    print(f"messungen: {len(df)}")

    feature_df = (create_feature_table(df))

    # ========================================================
    # save
    # ========================================================

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("\nfertig.")

    print(f"erfolgreich verarbeitet: {len(feature_df)}")

    print(f"feature table gespeichert unter: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()