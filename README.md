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

