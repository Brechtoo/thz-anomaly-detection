# Prüfkörper-Holdout v1

Diese Variante ist ein separates Experiment. Historische Splits, Feature-Tabellen,
Modelle und Mapping-Dateien bleiben unverändert. Die drei bestätigten metallischen
Referenzflächen werden über ihre IDs ausgeschlossen; die bekannte Koordinatenregel
prüft diesen Ausschluss zusätzlich. Es verbleiben 52.220 Punkte aus 35 räumlichen
Prüfkörpergruppen. Physische Wiederverwendung zwischen Messreihen ist unbekannt;
gleiche oder verschiedene physische Identitäten werden nicht unterstellt.

## Aufteilung vor jedem Modelltraining

`src.data.specimen_splits` führt einen deterministischen Kandidatenvergleich mit
`GroupShuffleSplit` durch. Jeder der 4.096 Kandidaten (Seed 42) enthält 7 Test-,
5 Validierungs- und 23 Trainingsgruppen. Messpunkte eines Körpers bleiben zusammen.
Vor der Zufallsziehung werden Gruppen und IDs kanonisch sortiert. Gleichstände
entscheidet der zuerst erzeugte Kandidat; Eingabezeilenfolge ist irrelevant.

`StratifiedGroupKFold` berücksichtigt gemischte Labels innerhalb von Gruppen,
erzeugt aber gleich große Folds statt direkt 65/15/20. Zwei Stufen würden hier
wenige und grobe Validierungsgruppen ergeben. `GroupShuffleSplit` erlaubt die
gewünschten Gruppenzahlen. Die Kandidatenwahl ergänzt dessen fehlende
Stratifizierung: minimiert wird der mittlere quadrierte Abstand von den Zielanteilen
65/15/20 für Gesamtpunktzahl, jede der vier Klassen und jede Messreihe. Jeder
Split muss alle Klassen und Messreihen enthalten. Ausschließlich Gruppierung,
Labels und Quellen gehen in diesen vorab festgelegten Score ein, keine Features,
Modellmetriken oder Vorhersagen. Dies ist eine balancierte Holdout-Konstruktion,
kein Modell-Tuning; es wurden keine Splits nach Modellleistung ausgewählt.

| Split | Punkte | Anteil | Körper | Klasse 0 | Klasse 1 | Klasse 2 | Klasse 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training | 34.235 | 65,56 % | 23 | 4.553 | 5.297 | 2.303 | 22.082 |
| Validation | 7.521 | 14,40 % | 5 | 1.140 | 1.161 | 657 | 4.563 |
| Test | 10.464 | 20,04 % | 7 | 1.432 | 1.518 | 756 | 6.758 |

| Split | Messreihe 1 | Messreihe 2 | Messreihe 3 |
| --- | ---: | ---: | ---: |
| Training | 13.377 | 13.538 | 7.320 |
| Validation | 3.009 | 3.039 | 1.473 |
| Test | 4.530 | 4.467 | 1.467 |

Die Klassenausstattung ist für eine vierklassige Auswertung ausreichend, aber
nicht exakt stratifiziert: Klasse 2 ist in Validation relativ häufiger als im
Training. Die effektiv unabhängige Stichprobe besteht nur aus 35 Körpern, davon
5 in Validation und 7 im Test. Reihe 3 ist mit jeweils nur einem Körper in beiden
Holdouts vertreten. Hohe Punktzahlen ersetzen keine unabhängigen Körper.

`configs/specimen_holdout_v1/splits/` enthält `train.csv`, `validation.csv`,
`test.csv` im bestehenden v1-Format (`instance_id,label`) sowie `membership.csv`
mit den Körper-IDs, `specimens.csv` mit Körper/Klassen-Häufigkeiten und
`summary.csv`. Manifest und SHA256-Prüfungen sichern die Zuordnung.
`load_membership` prüft zusätzlich Gruppenfreiheit und Sidecar-Hashes;
`restore_splits` wird unverändert wiederverwendet. Der bei der Split-Erstellung
ausgeführte Quellstand ist als `source.py` passend zum Manifesthash archiviert.

## Neue tsfresh-Aufbereitung

Die historische Tabelle mit 701 Features wird nicht als neue Evaluationsgrundlage
verwendet: Ihre Imputation und überwachte Selektion waren auf den alten,
punktweisen Trainingssplit gefittet, der Punkte der neuen Holdout-Körper enthält.
Für den neuen Holdout werden daher alle 777 Merkmale des bisherigen
`efficient`-Presets auf Rohsignalen neu berechnet. Pro Signal ist diese Extraktion
unabhängig von anderen Signalen. Imputationsgrenzen, Mediane und Selektion werden
anschließend ausschließlich auf dem neuen Training bestimmt. Das bestehende
`select_training_features` mit seinen Parametern wird wiederverwendet.
Das neue Ergebnis muss deshalb nicht wieder genau 701 Spalten enthalten.

`src.features.specimen_tsfresh` speichert Rohmerkmale in 512er-Batches unter
`results/experiments/specimen_holdout_v1/features/`. Die Fortsetzung überprüft
Katalog, Split-Manifest, Extraktionsquellen und Rohdatenhashes. Sämtliche
Rohdatenhashes, Imputationswerte, ausgewählte Features und Batch-Hashes werden
archiviert. Die selektierte Tabelle enthält ausschließlich ID, Label und explizite
`y__...`-Features. Gruppen, Koordinaten, Quellen und Rollen sind keine ML-Features.

## Baseline und Evaluation

`src.experiments.specimen_baseline` verwendet unverändert `build_models(seed=42)`:
Logistic Regression, Random Forest, HGB, XGBoost, LightGBM, Extra Trees und MLP.
Parameter, einschließlich der Modellstandards und Vorverarbeitung, werden aus der
bestehenden Registry übernommen und gespeichert. Varianzfilter und gegebenenfalls
Scaler fitten nur auf Training. Es gibt keine Hyperparametersuche, Schwellenwert-
Optimierung, Auswahl nach Testleistung oder Nachtraining auf Validation.

Die interne Zufallsvalidierung von HGB für Early Stopping bleibt als Bestandteil
der unveränderten Konfiguration erhalten. Sie ist innerhalb der Trainingskörper
punktweise und nicht gruppenunabhängig. Die äußeren Validation-/Testkörper bleiben
auch gegenüber diesem Schritt vollständig ausgeschlossen. Eine spätere Optimierung
müsste zusätzlich gruppenbewusste innere Modellselektion berücksichtigen.

Für jedes feste Modell werden Validation und Test separat ausgewertet: Accuracy,
Balanced Accuracy, F1 macro/weighted, vollständige 4×4-Konfusionsmatrix, Recall und
Support je Klasse, bestehende vierklassige Fehlerdefinitionen sowie Auswertung
pro Körper und Messreihe. Relative kritische Fehler beziehen sich auf wahre
Klassen 1/2, False Alarms auf wahre Klasse 3 und werden in Prozent angegeben.
Bei fehlendem Nenner liefert die bestehende Definition 0; ein expliziter Nenner
in den Tabellen verhindert eine Interpretation als nachgewiesene Sicherheit.
Gruppen-Macro-F1 verwendet stets alle vier Labels; Balanced Accuracy mittelt
über die tatsächlich vorhandenen wahren Klassen.

Alle Modelle werden zusätzlich lokal gespeichert. Ein neuer lokaler MLflow-Store
unter dem neuen Baseline-Ausgabeordner schützt die historische MLflow-Datenbank.
Jeder Run dokumentiert Parameter, beide Holdouts, Code-/Umgebungsstand, Warnungen,
Splits, Imputation, Datenhashes, Vorhersagen und Modell. Modelle werden mit einem
Thread gerechnet; Seeds und Versionen sind dokumentiert. Gleichheit auf anderer
Hardware oder mit anderen Bibliotheken wird nicht behauptet.

## Reproduktion

Für jeden neuen Aufbau ein neues Ausgabeziel verwenden:

```sh
.venv/bin/python -m src.data.specimen_splits \
  --catalog data/measurements/combined_dataset.csv \
  --mapping data/metadata_reference_excluded/specimen_mapping.csv \
  --output-dir /path/to/new-splits
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
  .venv/bin/python -m src.features.specimen_tsfresh \
  --split-dir configs/specimen_holdout_v1/splits \
  --output-dir results/experiments/specimen_holdout_v1/features --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42 \
  .venv/bin/python -m src.experiments.specimen_baseline \
  --split-dir configs/specimen_holdout_v1/splits \
  --feature-dir results/experiments/specimen_holdout_v1/features \
  --output-dir results/experiments/specimen_holdout_v1/baseline
```

Die Feature-Extraktion darf ihren eigenen vollständig geprüften Batch-Cache
fortsetzen; vorhandene finale Feature-Tabellen und Baseline-Ausgabeordner werden
nicht überschrieben. Neue Gruppensplits sind nicht mit den alten Manifesten
kombinierbar. Fehler eines Modells werden lokal protokolliert, ohne bereits
berechnete Metriken anderer Modelle zu verwerfen.

## Historische HGB-Referenzen

Zum Vergleich werden gespeicherte Vorhersagen unverändert mit `four_label_v2`
neu ausgewertet; ältere CSV-Fehlerzahlen sind teilweise überholt:

| Historischer Validierungslauf | Accuracy | Balanced Accuracy | Macro-F1 | Kritisch (1/2→3) | False Alarms (3→1/2) |
| --- | ---: | ---: | ---: | ---: | ---: |
| HGB, 52 Features | 88,84 % | 88,76 % | 0,8396 | 136 / 7,27 % | 582 / 10,64 % |
| HGB, 701 tsfresh-Features | 93,10 % | 84,64 % | 0,8721 | 305 / 16,30 % | 91 / 1,66 % |

Die 52-Feature-Referenz verwendet eine andere HGB-Konfiguration und balancierte
Sample-Gewichte; die neue Variante übernimmt die sieben Standardkonfigurationen
aus der tsfresh-Registry. Beide alten Referenzen verwenden den punktweisen Split
mit Referenzmessungen und 8.511 Validierungspunkten. Ein Unterschied gegenüber
dem neuen Ergebnis lässt sich daher weder allein dem Gruppensplit noch allein
dem Modell zuschreiben. Der 701-Feature-HGB ist hinsichtlich Modellparametern
die nähere Referenz, bleibt aber hinsichtlich Gruppen, Selektion und Umfang
methodisch verschieden. Historische Testzahlen mit umcodierten Labels oder
optimierten Schwellenwerten werden nicht als gleichartige Vierklassen-Baseline
behandelt.

### Grenze der Verteilung innerhalb einzelner Messreihen

Die Gesamtklassen sind überall vorhanden, aber die Kombination Messreihe × Klasse
ist nicht vollständig abgedeckt: Der Testkörper aus Reihe 3 enthält 150 Punkte
Klasse 0, **keine Klasse 1**, nur **5 Punkte Klasse 2** und 1.312 Klasse 3.
Der Recall für Klasse 1 in Reihe 3 ist damit nicht empirisch bestimmbar; der
technische Tabellenwert 0 hat Support 0. Reihe 2 hat in Validation nur 11 Punkte
Klasse 1. Die Körper- und Quellenberichte müssen mit ihren Supports gelesen
werden. Der Split wurde nach Feststellung dieser Grenze nicht auf Modellleistung
angepasst. Die Gesamtprüfung ersetzt keine ausreichend große externe Serie mit
Hohlraumkörpern aus jeder Aufnahmesituation.

Automatisierte Überprüfung des neuen und bestehenden Codes: **141 Tests bestanden**
(7 Warnungen: fehlende Vorhersageklassen im synthetischen Metriktest sowie bekannte
Bibliotheks-/Konvergenzwarnungen). Tests decken vollständige/eindeutige Zuordnung,
Referenzausschluss, Gruppenfreiheit, Zeilenfolgeunabhängigkeit, v1-Kompatibilität,
Überschreibschutz, Manipulationserkennung und train-only Feature-Aufbereitung ab.

## Ergebnisse des abgeschlossenen Baseline-Experiments

Nach train-only Selektion verbleiben **708 von 777 Features**. Alle sieben
Modelle wurden mit den bisherigen Hyperparametern trainiert; es gab keine
Trainings- oder Konvergenzwarnungen in den realen Läufen. Alle sieben
MLflow-Runs sind FINISHED. Vorhersagen nach erneutem Laden sämtlicher
lokaler Modelle stimmen exakt mit den gespeicherten Vorhersagen überein.
Die Parameter wurden zusätzlich gegen die unveränderte Registry geprüft.

Accuracy und Balanced Accuracy sind Prozentwerte, F1 liegt zwischen 0 und 1.
Fehlerraten sind Prozentwerte mit den oben definierten klassenbezogenen Nennern.

### validation

| Modell | Accuracy % | Balanced Accuracy % | Macro-F1 | Weighted-F1 | Kritisch n (%) | False Alarms n (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 85.24 | 70.86 | 0.7270 | 0.8363 | 529 (29.10) | 108 (2.37) |
| random_forest | 80.79 | 65.18 | 0.6410 | 0.7834 | 494 (27.17) | 63 (1.38) |
| hgb | 84.19 | 68.35 | 0.6913 | 0.8183 | 542 (29.81) | 50 (1.10) |
| xgboost | 83.98 | 69.86 | 0.7108 | 0.8228 | 522 (28.71) | 109 (2.39) |
| lightgbm | 83.49 | 69.03 | 0.6987 | 0.8167 | 516 (28.38) | 87 (1.91) |
| extra_trees | 80.23 | 64.38 | 0.6240 | 0.7749 | 507 (27.89) | 63 (1.38) |
| mlp | 83.23 | 72.45 | 0.7247 | 0.8233 | 395 (21.73) | 281 (6.16) |

Recall je Klasse (%):

| Modell | Klasse 0 | Klasse 1 | Klasse 2 | Klasse 3 |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 78.42 | 88.03 | 21.61 | 95.40 |
| random_forest | 74.30 | 86.56 | 8.52 | 91.34 |
| hgb | 78.95 | 85.79 | 13.39 | 95.29 |
| xgboost | 80.09 | 86.05 | 19.63 | 93.69 |
| lightgbm | 79.21 | 85.96 | 17.50 | 93.43 |
| extra_trees | 75.09 | 86.56 | 5.18 | 90.71 |
| mlp | 80.70 | 89.23 | 29.83 | 90.03 |

### test

| Modell | Accuracy % | Balanced Accuracy % | Macro-F1 | Weighted-F1 | Kritisch n (%) | False Alarms n (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 85.60 | 72.64 | 0.7369 | 0.8529 | 507 (22.30) | 215 (3.18) |
| random_forest | 85.66 | 65.60 | 0.6847 | 0.8352 | 839 (36.90) | 46 (0.68) |
| hgb | 85.66 | 67.61 | 0.7025 | 0.8414 | 738 (32.45) | 77 (1.14) |
| xgboost | 85.53 | 67.20 | 0.6979 | 0.8395 | 739 (32.50) | 77 (1.14) |
| lightgbm | 85.87 | 68.51 | 0.7144 | 0.8454 | 717 (31.53) | 87 (1.29) |
| extra_trees | 85.33 | 64.98 | 0.6756 | 0.8300 | 868 (38.17) | 39 (0.58) |
| mlp | 83.68 | 71.94 | 0.7197 | 0.8354 | 522 (22.96) | 353 (5.22) |

Recall je Klasse (%):

| Modell | Klasse 0 | Klasse 1 | Klasse 2 | Klasse 3 |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 84.22 | 76.09 | 36.77 | 93.49 |
| random_forest | 77.72 | 73.65 | 12.83 | 98.18 |
| hgb | 80.52 | 73.45 | 19.58 | 96.88 |
| xgboost | 79.54 | 74.11 | 18.25 | 96.89 |
| lightgbm | 78.98 | 75.30 | 23.02 | 96.73 |
| extra_trees | 77.65 | 73.32 | 10.98 | 97.97 |
| mlp | 84.08 | 75.69 | 37.43 | 90.56 |

### HGB-Konfusionsmatrizen

Zeilen: wahre Klassen 0/1/2/3; Spalten: vorhergesagte Klassen 0/1/2/3.

validation:

```text
        pred_0  pred_1  pred_2  pred_3
true_0     900      60      10     170
true_1      26     996      11     128
true_2     124      31      88     414
true_3     165      41       9    4348
```

test:

```text
        pred_0  pred_1  pred_2  pred_3
true_0    1153      46      40     193
true_1      81    1115      73     249
true_2      39      80     148     489
true_3     134      23      54    6547
```

### Einordnung

Der neue HGB erreicht in Validation 84,19 % Accuracy gegenüber 93,10 %
beim historischen tsfresh-HGB (-8,91 Prozentpunkte). Balanced Accuracy
fällt von 84,64 % auf 68,35 %. Die kritische Fehlerrate steigt von
16,30 % auf 29,81 %. Der neue HGB-Test erreicht 85,66 % Accuracy, aber
**738 kritische Fehler / 32,45 %** bei nur **19,58 % Recall für Klasse 2**.
Ein hoher Anteil korrekt erkannter Klasse-3-Punkte verdeckt diese Schwäche.

Die kritischen Testfehlerraten aller sieben Modelle reichen von 22,30 %
bis 38,17 %. Kein Modell wurde anhand des Tests ausgewählt oder nachoptimiert.
Die Unterschiede zur Historie sind wegen neuer Körperzusammensetzung,
Referenzausschluss und neuer Selektion kein isolierter kausaler Effekt des Splits.

Die Körperabhängigkeit ist ausgeprägt: Beim HGB-Test beträgt die Accuracy
je Körper 74,77–96,86 %. Für `messreihe_2_pk_14` werden 212 von 217
kritischen Fällen als Klasse 3 freigegeben (97,70 %). Auf dem Testkörper
aus Reihe 3 werden alle fünf Klasse-2-Punkte als Klasse 3 vorhergesagt;
die 96,86 % Gesamtaccuracy dieser Reihe ist daher kein Nachweis guter
Hohlraumerkennung. Im Validierungskörper aus Reihe 3 sind alle 69 Fälle
der Klassen 1/2 kritische Fehler. Kleine und fehlende Supports beachten.

Die train-only tsfresh-Selektion verhindert Zugriff auf Holdout-Labels.
Ihre statistischen Tests behandeln jedoch Messpunkte als Beobachtungen;
räumliche Abhängigkeit im Training kann die p-Werte optimistisch machen.
Dies ist keine gruppenunabhängige Signifikanzprüfung der Features.

Die Ergebnisse gelten für diesen einen Split räumlicher Körpergruppen
innerhalb bekannter Messreihen. Physische Wiederverwendung zwischen Reihen
bleibt ungeklärt. Nur sieben Testkörper und die unvollständige Klassen-
abdeckung einzelner Reihen begrenzen Aussagen über unbekannte physische
Körper und neue Aufnahmebedingungen. Es wurden keine punktweisen
Konfidenzintervalle als Ersatz für unabhängige Körper berechnet.

### Artefakte und Abschlussprüfung

Unter `results/experiments/specimen_holdout_v1/` liegen:

- `features/`: Rohfeature-Batches, Rohdatenhashes, train-only Imputation,
  selektierte Tabelle, Feature-Manifest und ausgeführter Extraktionsquellstand.
- `baseline/results.csv`: vollständige Metriken für beide Holdouts und alle Modelle.
- `baseline/performance_by_specimen_id.csv`: 84 Körper-/Modell-/Split-Zeilen.
- `baseline/performance_by_source_dataset.csv`: 42 Messreihen-/Modell-/Split-Zeilen.
- `baseline/evaluations/<model>/<split>/`: ID-verknüpfte Vorhersagen,
  4×4-Konfusionsmatrizen, Klassenberichte und einzelne Gruppenberichte.
- `baseline/models/`, `baseline/mlflow.db`, `baseline/mlartifacts/`: Modelle
  und sieben vollständige lokale MLflow-Runs einschließlich Metadaten.
- `baseline/verification.json`: bestätigte Parameteridentität, exakte
  Vorhersagen nach Laden aller sieben Modelle und sieben FINISHED-Runs.
- `historical_hgb_recomputed.csv`: neu berechnete historische Referenzmetriken.

Die Prüfsummen von **213 geschützten Ausgangsartefakten** sind unverändert.
Die echte Split-Zuordnung wurde zusätzlich nach zufälliger Umsortierung
neu berechnet und ist identisch (Kandidat 3168 von 4096, Seed 42). Alle
102 Rohfeature-Batches haben dieselben 777 Spalten und insgesamt 52.220 Zeilen.
141 pytest-Tests und der zusätzliche synthetische sieben-Modell-Integrationslauf
sind bestanden. Eine vollständige Wiederholung des realen Trainings wurde
nicht durchgeführt; die tatsächlichen Modellartefakte wurden erneut geladen
und reproduzieren beide gespeicherten Holdout-Vorhersagen exakt.

Die ergänzenden Prüf-/Berichtsskripte sind unter `baseline/metadata/` archiviert.
Sie werden aus dem Projektverzeichnis mit `PYTHONPATH=.` und dem bestehenden
`.venv/bin/python` ausgeführt. Die generierten Experimentdaten sind per
`.gitignore` ausgeschlossen; es wurde kein Commit oder Push ausgeführt.
