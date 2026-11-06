# tsfresh-Modellvergleich

Zusätzliche Modellpakete installieren und anschließend starten:

```sh
.venv/bin/python -m pip install -r requirements-comparison.txt
.venv/bin/python -m src.experiments.run_experiment
```

`MODEL_ENABLED` in `src/experiments/model_registry.py` schaltet die Modelle zentral.
RBF-SVM ist standardmäßig aus. Programmatisch kann eine explizite Auswahl übergeben
werden: `run_experiment(enabled={"logistic_regression": True, "hgb": True})`.
Alle aktivierten Modelle werden vor Trainingsbeginn instanziiert; fehlende Pakete
führen dadurch vor dem ersten Training zum Abbruch.

`configs/tsfresh_comparison.json` enthält die separaten SHA-256-Prüfsummen der
Feature-Tabelle, Extraktionsmetadaten und des bestehenden Split-Manifests sowie
Feature-Reihenfolge und Seed. Neue Tabellen benötigen eine bewusst aktualisierte
Vergleichskonfiguration. Die eingefrorenen HGB-Manifeste bleiben unverändert.
Der bisherige HGB-Ablauf ist mit `--baseline` weiterhin ausführbar.

Jede Pipeline entfernt auf Trainingsdaten konstante Features. LR, MLP und SVM
verwenden zusätzlich StandardScaler, ebenfalls nur auf Trainingsdaten angepasst.
Baummodelle benötigen keine Skalierung. Die bestehende tsfresh-Imputation und
statistische Selektion werden nicht wiederholt. Die gespeicherte Pipeline enthält
sämtliche Schritte für spätere Vorhersagen.

Verglichen wird ausschließlich auf der Validierung; der Testsplit wird nur zur
Prüfung der gespeicherten Zuordnung rekonstruiert. Es gibt kein Tuning und keine
automatische Modellauswahl. Kritische Fehler sollten bei der manuellen Beurteilung
Vorrang vor allgemeinen Qualitätsmetriken haben.

Unter `results/experiments/tsfresh_comparison_*` entstehen nach jedem Modell ein
Modellartefakt, Evaluationsdateien, Einzelergebnis und die aktualisierte
`validation_results.csv`. Jeder Modelllauf wird zusätzlich im lokalen MLflow-Store
als eigenes Run des Experiments `thz-tsfresh-comparison` erfasst.
Quellcode, Git-Zustand, Paketversionen, Thread-Konfiguration, Feature-Metadaten,
Parameter und Split-Dateien werden archiviert. Ein Seed und ein Thread reduzieren
Zufallsquellen; identische Ergebnisse über andere Bibliotheksversionen oder
Hardware hinweg sind damit nicht garantiert. Konvergenzwarnungen stehen im
MLflow-Artefakt `metadata/<modell>_preprocessing.json` und in den lokalen Metadaten.

`prediction_seconds` misst einen `predict`-Aufruf auf dem gesamten
Validierungssplit einschließlich der Pipeline-Vorverarbeitung. Die Dauer wird
in MLflow, im Einzelergebnis und in der Vergleichstabelle gespeichert.
Die Spalten `training_status`, `evaluation_status`, `local_save_status` und
`mlflow_status` dokumentieren die Phasen getrennt (`NOT_STARTED`, `FINISHED`,
`FAILED`). Der Gesamtstatus ist `PARTIAL`, wenn die Evaluation erfolgreich war,
aber eine Speicherung scheitert; bei Trainings-/Evaluationsfehlern `FAILED`.
Berechnete Metriken und Laufzeiten bleiben in beiden CSV-Ergebnistabellen erhalten.
Fehlertyp und Meldung stehen in `<phase>_error_type` und `<phase>_error`,
der vollständige Traceback in `<modell>_<phase>_error.txt`. Bei vorhandenem
MLflow-Run wird zusätzlich ein Fehlerartefakt hochgeladen; scheitert auch das,
bleibt der lokale Bericht erhalten. Der Vergleich fährt mit dem nächsten Modell fort.

Die MLflow-Modellspeicherung nutzt `name="model"` und explizit skops.
`src/experiments/model_serialization.py` enthält feste Freigaben pro Modellfamilie
für die internen Typen unserer selbst trainierten Pipelines. Unbekannte Typen
werden nicht automatisch übernommen. Diese Freigaben gelten nicht allgemein für
Modelldateien fremder Herkunft. MLflow archiviert die verwendete Freigabeliste im
Modellartefakt für das anschließende Laden mit `mlflow.sklearn.load_model`.
Die Parameter `artifact_path` beim Hochladen gewöhnlicher Dateien/Verzeichnisse
bleiben gültig; nur bei `log_model` wurde er durch `name` ersetzt.
Gemeinsame Eingangsprüfungen und die Prüfung fehlender Modellpakete bleiben
vor Trainingsbeginn verbindlich. Ein manueller Abbruch wird weitergereicht.

Die separate Vergleichs-Abhängigkeitsdatei fixiert MLflow und skops auf die
mit den Speicher-/Ladetests geprüften Versionen. Sie ersetzt nicht die eingefrorene
HGB-Lockdatei. Die MLflow-Ergebnis-CSV wird erst nach Aktualisierung des finalen
Status hochgeladen; Tests vergleichen sie mit dem lokalen Ergebnis.
