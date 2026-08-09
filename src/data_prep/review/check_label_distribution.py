from pathlib import Path
import pandas as pd


# ============================================================
# Pfad zur CSV mit den Labels
# ============================================================

CSV_PATH = Path(
    "/results/measurements/combined_dataset.csv"
)


# ============================================================
# Label-Verteilung ausgeben
# ============================================================

def main():
    df = pd.read_csv(CSV_PATH)

    if "label" not in df.columns:
        raise ValueError(
            f"Spalte 'label' nicht gefunden. Vorhandene Spalten: {list(df.columns)}"
        )

    total = len(df)

    print("\n======================================")
    print("Label-Verteilung")
    print("======================================")
    print(f"Gesamtanzahl Messungen: {total}\n")

    # Labels 0, 1, 2, 3 vollständig anzeigen, auch wenn eines fehlt
    label_counts = df["label"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    label_percentages = label_counts / total * 100

    for label in [0, 1, 2, 3]:
        count = label_counts.loc[label]
        percentage = label_percentages.loc[label]

        print(f"Label {label}: {count} Messungen ({percentage:.2f} %)")

    print("======================================\n")


if __name__ == "__main__":
    main()