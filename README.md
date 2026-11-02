# THz-TDS Anomalieerkennung

Dieses Repository enthält die im Rahmen einer Bachelorarbeit entwickelte Machine-Learning-Pipeline zur automatisierten Erkennung von Hohlräumen in THz-TDS-Reflexionsmessungen.
Das Projekt betrachtet sowohl überwachte als auch unüberwachte Verfahren.


## Überwachte Modelle

* Multinomiale logistische Regression 
* Random Forest
* Histogram-based Gradient Boosting
* Multi-Layer Perceptron
* Transformer
  

## Unüberwachte Verfahren

* Principal Component Analysis (PCA)
* Isolation Forest
* Autoencoder


## Projektstruktur

* `src/analysis/` – Feature-Analyse
* `src/data/` – Laden, Labeln und Split der Daten
* `src/features/` – Feature-Extraktion 
* `src/models/` – Überwachte Modelle
* `src/experiments/` – Training, Experimente und Optimierung
* `src/evaluation/` – Evaluation und Schwellenwertstrategien
* `src/unsupervised/` – Unüberwachte Modelle, inkl. Datenvorbereitung und Experimente
* `src/external_test/` – Tests auf neuen Daten
* `results/` – Feature-Tabellen, trainierte Modelle und Ergebnisse


## Daten

Die für die Machine-Learning-Experimente verarbeiteten und gelabelten Datensätze sowie die erzeugten Feature-Tabellen befinden sich in `results/`.


## Reproduzierbarkeit

Die Ergebnisse der Experimente befinden sich in `results/`.

Die Fehlerbewertung verwendet künftig `four_label_v2`. Bereits vorhandene
Fehlerzählungen und MLflow-Runs stammen aus der alten Rot/Gelb/Grün-Definition
(`rgy_v1`) und bleiben unverändert. Definitionen, Überschneidungen und der
Neubewertungsbedarf sind in [docs/error_metrics.md](docs/error_metrics.md) dokumentiert.

Für die HGB-Baseline sind die verwendete Umgebung in `requirements-lock.txt`
und die geordneten Split-IDs unter `configs/hgb_baseline/splits/` eingefroren.
Wiederherstellung, Provenienz und Grenzen des Feature-Neuaufbaus beschreibt
[docs/hgb_reproducibility.md](docs/hgb_reproducibility.md).

Die [Katalog- und Leakage-Prüfung](docs/catalog_reconstruction_and_leakage.md)
bestätigt die Rekonstruktion aus Originalannotationen und Rohmessungen.
Die bisherigen Splits verteilen jedoch Punkte derselben Messreihen auf
Train/Validation/Test. Die Kennzahlen gelten daher für die punktweise
Within-Scan-Auswertung und belegen keine Generalisierung auf unabhängige neue Scans.

Der [Sessionabschluss vom 02.11.2026](docs/session_closure_2026-11-02.md)
beschreibt den technischen Baseline-/Audit-Tag, die abschließenden Tests und
die Auswahl versionierter Reproduzierbarkeitsnachweise.

