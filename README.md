# HeRAI – THz-Hohlraumerkennung

HeRAI untersucht Hohlräume in THz-TDS-Reflexionsmessungen mit überwachten und
unüberwachten ML-Verfahren. Schwerpunkt sind reproduzierbare Experimente und
eine realistische Bewertung unbekannter Prüfkörper.

Labels: **0** schlechte Messqualität, **1** Hohlraum, **2** unsicher, **3** kein
Hohlraum. Kritisch sind **1/2→3**, False Alarms sind **3→1/2**. Kritische Fehler
haben bei der Beurteilung Vorrang vor mittleren Gütemaßen.

## Aktueller Forschungsstand

- Die drei metallischen Referenzflächen sind aus neuen Experimenten ausgeschlossen:
  **52.220 Punkte, 35 räumliche Prüfkörpergruppen** bleiben übrig.
- `specimen_holdout_v1` friert 23 Trainings-, fünf Validierungs- und sieben Testkörper
  ein. Die Vergleiche mit 52 manuellen und 708 selektierten tsfresh-Features sind
  abgeschlossen. Historische tsfresh-Vergleiche verwenden dagegen 701 Features.
- Anker- und Klasse-2-Ablationen verwenden ausschließlich die 23 Trainingskörper
  mit drei unveränderten Group-CV-Folds. Alte Validation-/Testkörper werden dort
  nicht erneut ausgewertet. Frühe Ankerung ist experimentell, physikalisch noch
  nicht bestätigt; Entfernen von Klasse 2 erhöht kritische Klasse-1-Fehler.
- Räumliche Gruppen beweisen keine physische Unabhängigkeit zwischen Messreihen.
  Die Ergebnisse sind Forschungsbefunde, keine Freigabe für einen Einsatz.

Der [Experimentkatalog](docs/experiments.md) verbindet sämtliche bisherigen
Experimentfamilien mit Berichten, Eingaben, Ergebnissen und lokalen Archiven.
[Kompakte Ergebnisnachweise](docs/evidence/README.md) sind im Repository vorgesehen;
vollständige Modelle, Rohdaten, Vorhersagen und MLflow-Stores bleiben lokal.

## Aufbau

| Ordner | Zweck |
| --- | --- |
| `src/data/`, `src/features/` | Kataloge, Metadaten, Splits und Feature-Extraktion |
| `src/experiments/`, `src/evaluation/` | Sieben klassische Modelle, Experimente und Metriken |
| `src/analysis/` | Fehler-, Signal-, Feature- und Klasse-2-Analysen |
| `src/models/`, `src/transformer/`, `src/unsupervised/` | Modellimplementierungen und ältere Verfahren |
| `src/generalization/` | Historische Auswertung externer Messungen |
| `configs/` | Eingefrorene Splits, Körper-Folds und Feature-/Umgebungsschemata |
| `data/metadata_reference_excluded/` | Aktuelle Prüfkörperzuordnung; `data/metadata/` ist historisch |
| `data/measurements/`, `results/feature_tables/` | Lokale Kataloge bzw. generierte Feature-Tabellen; ignoriert |
| `results/`, `mlruns/` | Historische Referenzen und vollständige lokale Experimente |
| `docs/`, `scripts/reproducibility/` | Berichte, Auditnachweise und Prüfskripte |
| `archive/` | Ignorierte lokale Sicherungen während der Projektpflege |

Die sieben klassischen Modelle sind Logistic Regression, Random Forest, HGB,
XGBoost, LightGBM, Extra Trees und MLP. RBF-SVM ist in der Registry deaktiviert.
Transformer, PCA, Isolation Forest und Autoencoder bleiben als historische
Projektlinien erhalten. Keine bestehenden Ergebnispfade werden umbenannt.

## Umgebung und Prüfungen

Python **3.12**, bestehende `.venv` verwenden. Eine neue Entwicklungsumgebung kann
mit `requirements-dev.txt` und `requirements-comparison.txt` eingerichtet werden:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt -r requirements-comparison.txt
.venv/bin/python -m ruff check .
MPLBACKEND=Agg .venv/bin/python -m pytest -q -ra
```

Die Installation ist für eine neue Umgebung gedacht; eine bestehende wissenschaftliche
Umgebung nicht unbemerkt aktualisieren. `requirements-lock.txt` bleibt der unveränderte
historische Snapshot. Konkrete Experimente archivieren zusätzliche Paketversionen.

GitHub Actions prüft Ruff und pytest mit Python 3.12 auf Ubuntu, ohne lokale THz-Daten
oder externe Tracking-Dienste. Ruff prüft `F`, `I`, `E9`. Die Tests verwenden überwiegend
synthetische Daten; sie reproduzieren keine vollständigen wissenschaftlichen Trainings.

## Daten und Reproduzierbarkeit

Ein frischer Checkout enthält **keine Rohmessungen, vollständigen Feature-Caches oder
neuen Modelle**. Rohdaten über `THZ_RAW_DATA_DIR` bereitstellen; externe Scans über
`THZ_EXTERNAL_SCAN_DIR`. Aktive Kataloge liegen unter `data/measurements/`. Eingefrorene
Kataloge/Run-Metadaten enthalten zum Teil absolute historische Pfade. Sie werden nicht
automatisch umgeschrieben, weil ihre Prüfsummen zu den Reproduzierbarkeitsnachweisen gehören.

Neue Läufe benötigen neue Ausgabeziele. Reproduktion und Voraussetzungen stehen in
den jeweiligen Berichten. Keine Trainingsaufrufe ausführen, nur um einen Checkout zu prüfen.
Die Referenzsperre lehnt alte Tabellen mit Metallreferenzen absichtlich ab.

- [Prüfkörper und Referenzausschluss](docs/specimen_mapping.md)
- [Gruppen-Holdout und tsfresh-Baseline](docs/specimen_holdout.md)
- [52 versus 708 Features](docs/specimen_manual52_comparison.md)
- [Alternative Anker](docs/experimental_anchor_group_cv.md)
- [Einfluss von Klasse 2](docs/class2_training_group_cv.md)
- [Fehlerdefinitionen und historische Grenzen](docs/error_metrics.md)
- [Historische HGB-Reproduktion und Archiv](docs/reproducibility_checks.md)
- [Commitvorbereitung und offene Risiken](docs/git_preparation.md)

Bereits versionierte historische Ergebnisse bleiben erhalten. Neu generierte große
Artefakte gehören nicht in Git; kompakte Tabellen und Grafiken werden explizit mit
Quell- und Exportprüfsummen ausgewählt. Ein Push allein sichert daher keine lokalen
wissenschaftlichen Daten oder Tracking-Stores.
