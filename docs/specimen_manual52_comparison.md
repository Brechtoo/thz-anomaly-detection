# 52 manuelle versus 708 tsfresh-Features: gleicher Prüfkörper-Holdout

Dieser Lauf verwendet die bestehenden `specimen_holdout_v1`-Manifeste unverändert:
34.235 Trainingspunkte / 23 Körper, 7.521 Validierungspunkte / 5 Körper und
10.464 Testpunkte / 7 Körper. Alle drei metallischen Referenzflächen sind ausgeschlossen.
**Validation ist die Grundlage der Feature-Set-Beurteilung. Test ist ausschließlich
explorativ; daraus werden weder Modell noch Feature-Set ausgewählt oder optimiert.**

## Prüfung und Wiederverwendung

`results/feature_tables/feature_table.csv` liegt mit allen 53.192 IDs, unveränderten
Original-Labels und genau 52 manuellen Features vollständig vor. SHA256:
`03d439403489d319feb7299847b159d9bd96e9398f72fd1d19bd64cb84df97ed`.

Die Tabelle wurde vollständig gegen den archivierten unabhängigen Neuaufbau unter
`../thz-archive/reproducibility_checks/2026-11-05/20261102T210010Z_4ec464c0/feature_rebuild/`
geprüft: identische Metadaten und Endlichkeitsmasken; sämtliche Featurewerte erfüllen
die bereits dokumentierte Toleranz `rtol=1e-11, atol=1e-13`. Maximale absolute
Abweichung: 2,80e-11. Zusätzlich wurden die Prüfsummen aller 53.192 Originalmessungen
gegen die archivierten Rohdatenhashes geprüft. Es wurde **nichts neu extrahiert**.

Einige aktuelle Extraktionsdateien unterscheiden sich in der Formatierung vom
archivierten Stand. Der damalige Quelltext wurde ausschließlich in einem temporären
Verzeichnis aus dem archivierten Git-Commit und dessen gespeichertem Patch rekonstruiert.
Seine Hashes stimmen mit dem Archivmanifest überein. Der Vergleich der Syntaxbäume
bestätigt identische Berechnungen; Kommentare/Leerzeilen verändern sie nicht.
Originalquellen, Git-Index und historische Artefakte wurden dabei nicht verändert.
Diese Prüfung bestätigt numerische Reproduzierbarkeit für diesen Datenstand,
keine Bytegleichheit beliebiger zukünftiger Extraktionen.

## Datenaufbereitung und Training

`src.features.specimen_manual` bindet die bestehende Tabelle über `instance_id`
an die eingefrorenen Splits. Eine explizite Schema-Datei
`configs/specimen_holdout_v1/manual_features.json` hält die 52 Merkmalsnamen und
Quelltabellen-Prüfsumme fest. Metadaten werden als Features abgelehnt.

`SimpleImputer(strategy='median', keep_empty_features=True)` wird ausschließlich
auf den Trainings-IDs gefittet. Bei dieser vollständig endlichen Tabelle ersetzt
er keine Werte. Mediane und trainiertes Imputer-Objekt werden trotzdem gespeichert,
um die Vorverarbeitung vollständig zu dokumentieren. Danach wird die gemeinsame
`src.experiments.specimen_baseline`-Pipeline verwendet: unveränderte sieben
Registry-Modelle mit Seed 42, Varianzfilter und bei LR/MLP StandardScaler, jeweils
nur auf Training gefittet. Es gibt keine neue Feature-Selektion, Gewichtung,
Hyperparametersuche, Schwellenwertoptimierung oder Nachtraining auf Validation.

Der gemeinsame Runner akzeptiert die manuelle Variante über ein geprüftes
`manual52`-Manifest; sein bisheriger tsfresh-Modus bleibt erhalten. Er verwendet
dieselben Split-, Metrik-, Gruppenbericht-, Speicher- und MLflow-Komponenten wie
im 708-Feature-Experiment. Originale Klassencodes bleiben immer 0/1/2/3.
Kritische Fehler sind 1/2→3 mit Nenner wahre 1/2; False Alarms sind 3→1/2 mit
Nenner wahre 3. Alle Fehlerraten werden in Prozent angegeben.

Die 708-Feature-Referenz ist ausschließlich der bereits abgeschlossene
Prüfkörperlauf `results/experiments/specimen_holdout_v1/baseline/`, nicht der
historische punktweise 701-Feature-Vergleich. Die alte Referenz wird nur gelesen.

## Reproduktion

Neue Ausgabeziele verwenden; die Funktionen verweigern vorhandene Ziele:

```sh
.venv/bin/python -m src.features.specimen_manual \
  --table results/feature_tables/feature_table.csv \
  --catalog data/measurements/combined_dataset.csv \
  --split-dir configs/specimen_holdout_v1/splits \
  --schema configs/specimen_holdout_v1/manual_features.json \
  --evidence-dir /path/to/archived/feature_rebuild \
  --output-dir /path/to/new-manual-features
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
  .venv/bin/python -m src.experiments.specimen_baseline \
  --split-dir configs/specimen_holdout_v1/splits \
  --feature-dir /path/to/new-manual-features \
  --output-dir /path/to/new-manual-baseline
```

Die neuen lokalen Artefakte liegen unter `results/experiments/specimen_manual52_v1/`.
Der Baseline-Unterordner enthält einen eigenen MLflow-Store (`mlflow.db` und
`mlartifacts/`); bestehende Tracking-Stores werden nicht ergänzt oder verändert.
MLflow und die Laufkonfiguration kennzeichnen den Testvergleich als explorativ.
Klassenberichte, Konfusionsmatrizen, ID-verknüpfte Vorhersagen und Auswertungen
pro Körper/Messreihe werden für beide Holdouts gespeichert.

## Grenzen der Aussage

Die Schlussfolgerung betrifft diesen einen Holdout von fünf Validierungskörpern
innerhalb dreier bekannter Messreihen. Räumliche Gruppen ersetzen keinen Nachweis,
dass zwischen Messreihen keine physischen Körper wiederverwendet wurden.
Im Testkörper aus Reihe 3 fehlen Klasse 1 und fast alle Klasse-2-Fälle.
Der Test war im vorherigen tsfresh-Experiment bereits ausgewertet worden;
der jetzige Feature-Set-Vergleich ist daher ausdrücklich keine neue konfirmatorische
Testprüfung. Für eine abschließende Entscheidung wäre eine weitere unabhängige
Prüfkörperstichprobe erforderlich. Unterschiede der Feature-Sets betreffen auch
ihre Signalaufbereitung und lassen sich nicht allein der Merkmalsanzahl zuschreiben.

## Ergebnisse: primärer Validierungsvergleich

Nur die Validierung begründet die folgende Beurteilung. Die 52 Features
haben bei **keinem** der sieben Modelle weniger kritische Fehler als die
708 Features. Macro-F1 und Weighted-F1 sind dagegen bei fünf Modellen höher,
Balanced Accuracy bei vier und Accuracy bei drei Modellen (ein Gleichstand).
False Alarms sind bei 52 Features nur für LR geringer, bei sechs Modellen höher.
Für das vorab priorisierte Ziel weniger kritischer Freigaben ist deshalb
**tsfresh708 auf diesem Validierungssplit vorzuziehen**. Es gibt keine
universelle Überlegenheit über alle Metriken und Modellfamilien. Die sieben
Vergleiche teilen dieselben fünf Validierungskörper und sind keine sieben
unabhängigen statistischen Replikationen. Es wurden keine Hyperparameter angepasst.

Paare in der Reihenfolge **52 / 708**; F1 ist dimensionslos, Raten sind Prozent:

| Modell | Macro-F1 52 / 708 | Kritische Fehler n 52 / 708 | Kritische Rate % 52 / 708 |
| --- | ---: | ---: | ---: |
| logistic_regression | 0.6436 / 0.7270 | 777 / 529 | 42.74 / 29.10 |
| random_forest | 0.6927 / 0.6410 | 592 / 494 | 32.56 / 27.17 |
| hgb | 0.7153 / 0.6913 | 575 / 542 | 31.63 / 29.81 |
| xgboost | 0.7127 / 0.7108 | 574 / 522 | 31.57 / 28.71 |
| lightgbm | 0.7149 / 0.6987 | 577 / 516 | 31.74 / 28.38 |
| extra_trees | 0.6849 / 0.6240 | 597 / 507 | 32.84 / 27.89 |
| mlp | 0.6664 / 0.7247 | 495 / 395 | 27.23 / 21.73 |

### Alle Metriken der 52 Features: Validation

| Modell | Accuracy % | Balanced Accuracy % | Macro-F1 | Weighted-F1 | Kritisch n (%) | False Alarms n (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 82.40 | 62.90 | 0.6436 | 0.7900 | 777 (42.74) | 59 (1.29) |
| random_forest | 83.09 | 68.06 | 0.6927 | 0.8131 | 592 (32.56) | 265 (5.81) |
| hgb | 84.19 | 69.70 | 0.7153 | 0.8272 | 575 (31.63) | 249 (5.46) |
| xgboost | 83.50 | 69.56 | 0.7127 | 0.8233 | 574 (31.57) | 298 (6.53) |
| lightgbm | 84.58 | 69.77 | 0.7149 | 0.8291 | 577 (31.74) | 215 (4.71) |
| extra_trees | 82.82 | 67.55 | 0.6849 | 0.8085 | 597 (32.84) | 222 (4.87) |
| mlp | 82.22 | 67.56 | 0.6664 | 0.7999 | 495 (27.23) | 317 (6.95) |

Recall der 52 Features je Klasse (%):

| Modell | Klasse 0 | Klasse 1 | Klasse 2 | Klasse 3 |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 72.46 | 77.86 | 3.96 | 97.33 |
| random_forest | 78.25 | 85.62 | 14.92 | 93.47 |
| hgb | 79.21 | 87.68 | 17.81 | 94.10 |
| xgboost | 79.21 | 87.08 | 19.03 | 92.94 |
| lightgbm | 80.18 | 87.42 | 16.74 | 94.72 |
| extra_trees | 78.07 | 85.36 | 13.39 | 93.36 |
| mlp | 78.51 | 90.01 | 10.20 | 91.54 |

### Explorativer Testvergleich, nicht zur Auswahl

| Modell | Accuracy % | Balanced Accuracy % | Macro-F1 | Weighted-F1 | Kritisch n (%) | False Alarms n (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 86.06 | 67.04 | 0.6905 | 0.8412 | 763 (33.55) | 136 (2.01) |
| random_forest | 87.03 | 69.75 | 0.7200 | 0.8536 | 772 (33.95) | 144 (2.13) |
| hgb | 87.84 | 71.28 | 0.7361 | 0.8632 | 714 (31.40) | 140 (2.07) |
| xgboost | 87.88 | 71.62 | 0.7383 | 0.8641 | 665 (29.24) | 151 (2.23) |
| lightgbm | 87.90 | 71.10 | 0.7361 | 0.8632 | 734 (32.28) | 118 (1.75) |
| extra_trees | 86.76 | 69.52 | 0.7186 | 0.8517 | 739 (32.50) | 165 (2.44) |
| mlp | 86.73 | 71.19 | 0.7273 | 0.8554 | 633 (27.84) | 255 (3.77) |

Recall der 52 Features je Klasse (%):

| Modell | Klasse 0 | Klasse 1 | Klasse 2 | Klasse 3 |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 78.63 | 80.30 | 12.04 | 97.20 |
| random_forest | 86.59 | 79.38 | 16.27 | 96.76 |
| hgb | 87.36 | 81.69 | 19.05 | 97.03 |
| xgboost | 86.45 | 83.07 | 20.11 | 96.85 |
| lightgbm | 87.29 | 81.03 | 18.78 | 97.31 |
| extra_trees | 83.52 | 80.76 | 17.20 | 96.58 |
| mlp | 84.99 | 82.67 | 21.83 | 95.26 |

### Einordnung des explorativen Tests

Die Test-Accuracy liegt bei 52 Features für alle sieben Modelle höher.
Die kritische Fehlerrate ist dort für vier Modelle niedriger, für drei höher
als bei 708 Features. Diese abweichende Rangfolge zeigt die Abhängigkeit
von den wenigen gehaltenen Körpern. Sie ändert **nicht** die zuvor ausschließlich
aus Validation abgeleitete Präferenz für tsfresh beim kritischen Fehlerziel.
Der explorative Test wird weder zur Modellauswahl noch zur nachträglichen
Anpassung von Features, Hyperparametern oder Schwellenwerten verwendet.

Die MLP-Konvergenzwarnung bei 52 Features ist dokumentiert: 200 Iterationen
wurden erreicht, ohne dass vollständige Konvergenz gemeldet wurde.
Das bestehende Limit bleibt für einen fairen Vergleich unverändert.

Beispiel HGB auf Validation: Beide Feature-Sets erreichen 84,19 % Accuracy.
Die 52 Features erhöhen Macro-F1 von 0,6913 auf 0,7153, verursachen aber
575 statt 542 kritische Fehler und 249 statt 50 False Alarms. Eine reine
Accuracy- oder F1-Beurteilung würde diesen Unterschied verdecken.

### Artefakte und Prüfung

- `features/verification.json`: Nachweis der Tabellen-, Rohdaten- und Quelltextprüfung.
- `features/imputer.joblib`, `imputation.csv`: ausschließlich aus Training bestimmte Mediane.
- `baseline/results.csv`: vollständige Metriken und Klassen-Recalls beider Holdouts.
- `baseline/evaluations/<model>/<split>/`: alle 4×4-Konfusionsmatrizen,
  Klassenberichte und ID-verknüpften Vorhersagen.
- `baseline/performance_by_specimen_id.csv` und `performance_by_source_dataset.csv`:
  Körper- und Messreihenauswertungen, mit expliziter Rolle der Holdouts.
- `comparison/validation.csv`, `validation_assessment.json`: ausschließlich
  validierungsbasierter Vergleich und maschinenlesbare Beurteilung.
- `comparison/exploratory_test.csv`: separate, ausdrücklich explorative Gegenüberstellung.
- `baseline/models/`, `mlflow.db`, `mlartifacts/`: sieben Modelle und vollständige lokale Runs.
- `baseline/verification.json`: Prüfung gleicher Splits/Parameter, train-only Mediane
  und exakt reproduzierter Vorhersagen nach Laden aller sieben Modelle.

Alle sieben Modelle reproduzieren ihre gespeicherten Vorhersagen auch aus
der ursprünglichen 52-Feature-Tabelle über den gespeicherten Imputer exakt.
Die Hyperparameter einschließlich der bisherigen Pipeline-Schritte stimmen
mit dem 708-Lauf überein. 29.284 geschützte Dateien der ursprünglichen Daten,
Splits und bisherigen Experimente sind unverändert. Das echte Training wurde
nicht ein zweites Mal wiederholt; Modell- und Vorverarbeitungsartefakte wurden
erneut geladen und geprüft. Gleichheit auf anderer Hardware wird nicht behauptet.

Zum reproduzierbaren Vergleich der gespeicherten Ergebnisse:

```sh
.venv/bin/python -m src.analysis.specimen_feature_comparison \
  --manual results/experiments/specimen_manual52_v1/baseline/results.csv \
  --tsfresh results/experiments/specimen_holdout_v1/baseline/results.csv \
  --output-dir /path/to/new-comparison
```

Tests prüfen Vollständigkeit, Prüfsummen, Quelltextgleichheit trotz Formatierung,
Metadatenausschluss, train-only Imputation sowie die Unabhängigkeit der
Validierungsbeurteilung von beliebig veränderten Testmetriken.

**Abschließender Testlauf: 153 Tests bestanden**, sieben bekannte Warnungen.
Zusätzlich sind alle sieben realen MLflow-Runs FINISHED. Die Warnung zur
MLP-Konvergenz ist in dessen Laufartefakten und im Bericht ausdrücklich vermerkt.
