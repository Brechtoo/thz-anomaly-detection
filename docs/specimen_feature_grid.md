# HeRAI-Featurevergleich auf festen Prüfkörpern

## Umfang und Vergleichsbedingungen

Der Plan umfasst 23 Feature-Konfigurationen, acht tabellarische
Vierklassenmodelle und drei eingefrorene Gruppen-Folds auf 34.235
Trainingspunkten aus 23 Prüfkörpern. Jede nichtleere Kombination
aus manuell, tsfresh, TSFEL und höchstens einer der beiden
physikalischen Versionen ist vorgesehen. Das ergibt 184
Modell-Konfigurations-Paare und 552 geplante Fold-Ergebnisse.
Der manuelle Block verwendet die 52 Merkmale mit frühem Anker
aus dem bisherigen Anker-/Klasse-2-Vergleich. TSFEL umfasst 72
Spalten, die physikalischen Alternativen 37 bzw. 39.

Die historische Tabelle mit **701 tsfresh-Features** ist für
dieses Protokoll ungeeignet. Ihre Imputation und überwachte
Selektion nutzten den historischen punktweisen Trainingssplit,
der Punkte heutiger Holdout-Körper enthält. Die archivierte
708er-Tabelle wurde nur auf allen 23 Trainingskörpern angepasst
und wäre in der **inneren Gruppen-CV** ebenfalls vorselektiert.
Für jede äußere CV-Trainingspartition werden deshalb aus den
**777 geprüften signalweisen Rohmerkmalen** die bisherige
Imputation und Selektion neu gelernt. Sie ergeben vor der
Prüfung exakter Konstanten und Duplikate 700, 704 und 684
Merkmale. Der 708er-Stand dient ausschließlich für die
Endfits auf allen 23 Trainingskörpern.
[Historische tsfresh-Provenienz](specimen_holdout.md)

Die Feature-Loader prüfen SHA-256, Messungs-IDs, Labels,
Körper, Splits und Folds. Die nicht-tsfresh-Blöcke verwenden
foldinterne Median-Imputation. Konstante und exakt redundante
Spalten werden anhand der Fit-Zeilen bestimmt. Identische
Spalten werden in der Reihenfolge manuell, tsfresh, TSFEL,
physikalisch repräsentiert. Varianzfilter und Skalierung
der Registry-Pipelines fitten auf denselben Fit-Zeilen.
Klasse 2 bleibt enthalten. Keine Probe wird wegen fehlender
Echo-Merkmale entfernt.

Die acht Registry-Modelle verwenden Seed 42 und ihre
bisherigen festen Konfigurationen. Die RBF-SVM verwendet
den zuvor festgelegten Betriebspunkt `C=0.3`, `gamma=0.001`
und balancierte Gewichte. Für ihre Wahrscheinlichkeiten
wird die native LibSVM-Schätzung aktiviert. Deren interne
Kalibration ist punktweise und nutzt ausschließlich Zeilen
des jeweiligen äußeren Trainings-Folds. Daraus folgt eine
körpergetrennte OOF-Auswertung, ohne Nachweis einer
körpergetrennt kalibrierten Wahrscheinlichkeit. Die
harten Modellvorhersagen stammen für alle Modelle aus
`predict`. Die Testentscheidung basiert auf diesen
Vorhersagen, auch wenn bei LibSVM deren Label vom Argmax
der Wahrscheinlichkeiten abweicht. Die Ausgabespalten
`p_0` bis `p_3` folgen den explizit überprüften Modellklassen.
Es gibt keine Schwellenwertsuche.

## Wiederverwendung und Holdouts

Für die sieben Nicht-SVM-Modelle des manuellen Einzelsets
wurden vorhandene native Vierklassen-OOF-Ausgaben aus
`class2_group_cv_v1` gefunden. Die 21 Fold-Ergebnisse werden
wiederverwendet, sofern archivierte Eingabe-Hashes,
Gruppenzuordnung, Feature-Reihenfolge, Imputationsmediane
und Registry-Parameter übereinstimmen. Die alten
Vierklassenmetriken werden aus den archivierten Vorhersagen
neu berechnet. Die historische kombinierte Fit-/
Vorhersagezeit ist als solche in der Fold-Provenienz
gekennzeichnet. Frühere SVM- und Fusions-OOF-Artefakte
enthielten nur harte Vorhersagen und sind daher keine
Eingabe für die neuen Wahrscheinlichkeitsberichte.

Die gepoolte Rangfolge priorisiert zuerst kritische Fehler
(wahre 1/2 → 3, zusätzlich nach wahrer Klasse getrennt)
und bei Gleichstand Fehlalarme (wahre 3 → 1/2).
F1, Balanced Accuracy, Konfusionsmatrix und Log-Loss
ergänzen den Vergleich. Der finale Validierungsplan
wird aus den fünf besten vollständig ausgewerteten OOF-Paaren
erstellt. `shortlist.json` hält die fünf Kandidaten fest.
Die spätere Testauswertung setzt eine endgültig festgelegte
Konfiguration voraus und wird separat in `test_selection.json`
dokumentiert. Der Test nimmt nicht am Gesamtranking teil.

## Ausführung und Status

Der konkrete Lauf liegt unter
`results/experiments/specimen_feature_grid_v1/`.
Je Fold werden unveränderliche Checkpoints für Vorhersagen,
Wahrscheinlichkeiten, Konfusionsmatrix, Laufzeit,
Vorverarbeitung, Warnungen und MLflow-Run-ID geschrieben.
Die tsfresh-Fold-Caches sind nach Eingabe, Kohorte und
Umgebung prüfsummengesichert. Es laufen höchstens zwei
Fits parallel, jeweils mit einem Thread. Die SVM verwendet
höchstens einen Job samt temporärem Kernel-Array.
Ein Fehler wird pro Fit protokolliert und blockiert die
übrigen Fits nicht. Erfolgreiche Checkpoints werden übersprungen.

Der lokale Supervisor war für periodische Berichte und einen
einmaligen Wiederanlauf zuständig. Die Konfigurationen,
Kohortenzuordnung und Einzelruns gehören zur zentralen Projekt-Datenbank unter
`thz-specimen-feature-grid`. Große Datendateien und
generierte Ergebnisse verbleiben im lokalen Laufordner.

Aktuelle Zwischenergebnisse und den späteren Abschluss
enthält [der laufend aktualisierte Bericht](../results/experiments/specimen_feature_grid_v1/report/status.md).
Nur Paare mit drei vollständigen, zur Kohorte passenden
Fold-Checkpoints werden gerankt. `paired_deltas.csv`
enthält direkte Ergänzungen eines Blocks bei gleichem
Modell sowie V1→V2-Substitutionen. Die Status-Tabelle
unterscheidet ausstehende, fehlgeschlagene und ungültige
Checkpoints.

### Wiederanlauf der Validation am 21. November 2026

Alle 552 CV-Checkpoints und die erste Validation (`manual+tsfel`)
waren vor dem Ende des ursprünglichen Workers gespeichert. Der
letzte Supervisor-Bericht entstand vor diesem Validation-Checkpoint.
Der automatische Wiederanlauf scheiterte an einem Indexvergleich.
Der Parquet-Checkpoint enthält einen `RangeIndex`, die gefilterte
Kohorte behält die Indizes der Gesamttabelle. Die Ursache des
ursprünglichen Worker-Endes ist aus `worker.log` nicht ersichtlich.
Der MLflow-Parent blieb nach dem Prozessende auf `RUNNING`.

`resume_validation.py` im Laufordner prüft die Kohorte mit
normalisiertem Index, verifiziert den eingefrorenen Plan sowie die
OOF-Shortlist und verwendet fertige Checkpoints weiter. Es nutzt
die bestehende Fit- und Logging-Logik unter demselben MLflow-Parent
`ce9ba6175b764560992561401ea9539b`. Das Skript wertet ausschließlich
Validation aus. Der allgemeine Runner würde anschließend auch den
Test auswerten.

Der erste Wiederanlauf setzte die Tracking-URI, wählte aber das
Grid-Experiment nicht explizit. MLflow legte den `manual`-Child im
Default-Experiment an. Dessen Artefaktpfad verweist auf den alten
Projektordner. Der fehlgeschlagene Child bleibt als `FAILED`
nachvollziehbar. Der korrigierte Wiederanlauf prüft das Experiment
und dessen lokalen Artefaktpfad vor dem Fit. Er sichert neue
Vorhersagen vor dem MLflow-Logging, damit bei einem Loggingfehler
kein weiterer Fit erforderlich ist.

Der Wiederanlauf endete um 09:13 Uhr MEZ. Im Grid-Experiment
stehen alle 558 Runs auf `FINISHED` (552 CV-Folds, fünf
Validation-Runs und der Parent). Die fünf Vorhersagedateien für
jeweils 7.521 Messungen und ihre MLflow-Artefakte wurden geprüft.
Manuell plus TSFEL führt die Validation-Rangfolge mit 146
kritischen Fehlern (8,03 %) und 846 Fehlalarmen (18,54 %) an.
Es gibt keine endgültige Test-Konfiguration und keine
Testauswertung. Der Parent dokumentiert den Stand mit
`grid.test_status=held_for_final_configuration`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=42
PYTHONPATH=. .venv/bin/python results/experiments/specimen_feature_grid_v1/resume_validation.py
.venv/bin/python -m src.analysis.feature_grid_report \
  --output-dir results/experiments/specimen_feature_grid_v1
```

Die Originalsplits, früheren Ergebnisse und Feature-Tabellen
werden nur gelesen.
