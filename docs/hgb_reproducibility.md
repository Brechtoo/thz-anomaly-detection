# Reproduzierbare HGB-Baseline: Betrieb und Geltungsbereich

**Technisch reproduzierbarer Within-Scan-Baseline- und Audit-Stand.
Keine gruppenunabhängige Generalisierung nachgewiesen.**

Alle drei Messreihen liegen in Train, Validation und Test. Die Kennzahlen
beschreiben die punktweise Aufgabe innerhalb bekannter Aufnahmen, nicht die
Leistung auf unabhängigen neuen Scans/Prüfkörpern. Die bestehenden Splits werden
für historische Vergleichbarkeit erhalten, nicht als leakage-frei freigegeben.

## Nachweise

- [Kompakter maschinenlesbarer Audit-Nachweis](audit/2026-11-02/summary.json):
  Referenz-/Wiederholungsläufe, Hashes, Kennzahlen und wissenschaftliche Grenzen.
- [Installations- und Feature-Prüfung](hgb_reproducibility_verification.md):
  frische Umgebung und vollständiger Feature-Neuaufbau.
- [Numerische Bewertung](feature_numerical_reproducibility.md): Statistik je
  Feature, Toleranz und unveränderte HGB-Validierungsvorhersagen.
- [Katalog und Leakage](catalog_reconstruction_and_leakage.md): byteidentische
  Rekonstruktion, Gruppenüberlappung und unverändert belassene Splits.

Referenz-Run: `a5b8cf67d02d4214bb8e36ea946e839a`.
Erfolgreiche Trainingswiederholung: `674a859ff5b94331843ce811abda18a0`.
Anschließende Prüfungen liefen ohne weiteres Modelltraining.

## Umgebung

`requirements-lock.txt` fixiert alle 101 verwendeten Distributionen für
**Python 3.12.5 auf macOS arm64**. Die frische Installation und `pip check` wurden
erfolgreich geprüft. Für diese Baseline den Lock statt der allgemeinen,
ungepinnten `requirements.txt`/`requirements-dev.txt` verwenden:

```sh
/path/to/python3.12.5 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q
```

Der Lock enthält Versionspins, keine Plattformmarker oder Wheel-Hashes.
Die bei der erfolgreichen Installation verwendeten Distributionshashes sind
zusätzlich unter `docs/audit/2026-11-02/installed_distributions_sha256.json`
archiviert. Andere Plattformen, Binärpakete und langfristige Paketverfügbarkeit
bleiben Grenzen; ein kontrolliertes Wheel-/VM-Archiv wäre zusätzliche Absicherung.

## Eingabedaten und gespeicherte Splits

Die Originaldaten werden separat bereitgestellt, nicht aus Git heruntergeladen.
`configs/hgb_baseline/splits/` versioniert ausschließlich geordnete IDs und
Original-Labels samt Manifest. Die Mitgliedschaften entsprechen dem geprüften
Lauf: Train 34.042, Validation 8.511, Test 10.639; keine neuen Zufallssplits.

`restore_splits(X, y, instance_ids, manifest_dir)` stellt die Zuordnung auch nach
Umsortierung der Eingabe wieder her. Checksummen, eindeutige IDs, disjunkte und
vollständige Abdeckung, Labelgleichheit und Indexausrichtung werden geprüft.

Der HGB-Einstieg verlangt zusätzlich den exakten Referenz-Feature-Dateihash und
Spaltenreihenfolge. Eine numerisch äquivalente Neu-Extraktion ersetzt die
historische Tabelle nicht automatisch. Die Toleranzfreigabe ist getrennt von
strikter Datenversionierung.

## Künftige HGB-Runs

`src.experiments.run_experiment` prüft vor Training die gesperrte Umgebung und
Daten-/Split-Identität. Jeder Aufruf reserviert einen neuen Ausgabeordner mit
UTC-Zeit und UUID-Suffix. Ein geerbtes `MLFLOW_RUN_ID` oder ein bereits aktiver
Run führt zum Abbruch, damit historische Runs nicht versehentlich fortgesetzt
werden. Verwendet wird der lokale SQLite-Store und das Experiment `thz-baselines`.

Metadaten umfassen Commit-ID, Dirty-Status einschließlich unversionierter
Dateien, getrackten Patch, tatsächlichen Python-Quelltext unter `src/`, Lock,
Split-Manifeste und Plattform-/Threadinfos. Tags enthalten Daten-, Lock- und
Manifesthash, Split-Herkunft und `error_metrics_version=four_label_v2`.
Die Einstiegstests ersetzen Training und MLflow durch Fakes.

## Neuaufbau ohne Überschreiben

Explizite Befehle für spätere Neuaufbauten, jeweils mit neuem Ausgabeziel:

```sh
.venv/bin/python -m src.data.labeling.rebuild_catalog \
  --raw-root /path/to/labeled-data --output-dir /path/to/new-catalog
.venv/bin/python -m src.features.rebuild_feature_table \
  --input /path/to/new-catalog/combined_dataset.csv \
  --output-dir /path/to/new-features
```

Der Katalog-Einstieg übernimmt nur vorhandene Originalannotationen, bewahrt
Rundungs-/ID-Regeln und lehnt mehrdeutige Zuordnungen ab. Die Feature-Extraktion
bewahrt die Reihenfolge und protokolliert Rohdaten-, Quellen- und Ausgabehashes.
Bei Pfadverlagerung sind `--old-raw-root` und `--raw-root` gemeinsam nutzbar.
Die API-Anpassung `trapz` → `trapezoid` bewahrt die Trapezregel.

## Git und lokale Evidenz

Versioniert werden Code, Lock, Split-IDs, Dokumentation und kompakte,
pfadbereinigte Audit-Nachweise unter `docs/audit/`. Vollständige lokale Logs,
Messungskataloge, Feature-Tabellen, Modelle und MLflow-Dateien bleiben lokal
und werden durch `.gitignore` von neuen Commits ausgeschlossen. Bereits früher
versionierte historische Ergebnisse werden nicht entfernt oder verändert.

Wissenschaftlich offen bleiben eine unabhängige gruppenbasierte Evaluation,
autoritative Aufnahme-/Prüfkörper-Gruppen und die vollständige historische
Feature-Extraktionsumgebung. Gruppenalternativen sind dokumentiert; eine
Änderung der bestehenden Splits erfordert eine gesonderte Freigabe.
