# Git-Vorbereitung: Prüfübersicht

Historischer Vorbereitungsstand: 10.11.2026, vor der Kommitterstellung.
**Zum Zeitpunkt dieser Vorbereitung waren keine Commits, Pushes, Merges,
Staging-Aktionen oder Branch-Wechsel ausgeführt.** Der Nutzer hat anschließend
die untenstehende Reihenfolge ausdrücklich zur Umsetzung und zum Push nach
erfolgreicher Prüfung freigegeben. Es werden keine bestehenden Modelle trainiert.

Bei der Umsetzung wird jeder vorgemerkte Index separat auf Artefakte,
Zugangsdatenmuster, Imports und Dokumentationslinks geprüft und ohne lokale Daten
exportiert. Ruff und die jeweils gesamte vorhandene Testsuite laufen auf diesem
isolierten Stand vor dem Commit. Zwei vorausgreifende Dokumentationslinks werden
erst im abschließenden Dokumentationscommit ergänzt. Die Ausführungsprotokolle
liegen lokal unter `archive/commit_execution_2026-11-10/`; Git-Log und GitHub Actions
dokumentieren die tatsächlich veröffentlichten Commits und deren CI-Ergebnis.

## Vorgenommene Aufräumarbeiten

- Gesamten sichtbaren Projektbestand und die ignorierten Experiment-/Tracking-
  Verzeichnisse inventarisiert. Unklare oder historische Dateien bleiben erhalten;
  aktive Module, Referenzmodelle und bestehende Ergebnispfade wurden nicht verschoben.
- Der vorherige uncommittete Quell-/Dokumentstand ist unter
  `archive/maintenance_2026-11-10/before/` mit Prüfsummen gesichert. Git-Vorzustand,
  vollständige wissenschaftliche Bestandshashes und lokale Prüfprotokolle liegen daneben.
- `.gitignore` deckt beide Katalogpfade, den abgeleiteten referenzfreien Katalog,
  lokale Archive, Tracking-Stores und neue Modellserialisierungen ab. Bereits in
  Git vorhandene historische Modelle/CSV-Dateien werden dadurch nicht entfernt.
- Die zentrale Katalogkonfiguration verwendet jetzt das tatsächlich bestehende
  `data/measurements/`; historische Manifestpfade wurden nicht umgeschrieben.
- Neue Quellsnapshots schließen `mlruns`, `mlartifacts`, Datenbanken und Caches aus.
  Das verhindert weitere rekursive Provenienzkopien aus `src/experiments/mlruns/`.
  Die vorhandenen Kopien bleiben wegen historischer Artefaktreferenzen bestehen.
- 31 Ruff-Importbefunde behoben; keine Featureformeln oder Modellhyperparameter geändert.
- 21 regenerierbare Cache-Verzeichnisse und `.DS_Store` entfernt: insgesamt
  180 Dateien, 865.079 Bytes. Die Finder-Datei liegt zusätzlich im lokalen Pflegearchiv.
- README und [Experimentkatalog](experiments.md) aktualisiert. Die Berichte zeigen
  ihre Grafiken jetzt aus [57 ausgewählten Ergebnisnachweisen](evidence/README.md)
  statt ausschließlich aus ignorierten lokalen Runs. Quell-/Exporthashes belegen
  Kopien und die beiden ausdrücklich ausgewiesenen CSV-Spaltenprojektionen.
- Die vorhandenen drei CV-Folds sind zusätzlich als 23 Körper-/Fold-Zeilen unter
  `configs/anchor_group_cv_v1/` dokumentiert. Keine Split-Zuordnung wurde neu gewählt.

## Vorgeschlagene Commit-Reihenfolge

Imports werden jeweils zusammen mit der fachlich zugehörigen Datei eingecheckt.
Die Reihenfolge berücksichtigt bestehende Modulabhängigkeiten; insbesondere muss
der gemeinsame Baseline-Runner seine manuelle Feature-Schemahilfe bereits enthalten.
Neue große Run-Verzeichnisse, Messkataloge und lokale Archive gehören in keinen Commit.

| Nr. | Commit-Message | Dateien / thematische Einheit |
| --- | --- | --- |
| 1 | `feat(data): map specimens and exclude metallic references` | `src/data/{reference_measurements,specimen_mapping,dataset}.py`; Referenzsperren in tsfresh/Transformer/Unsupervised; `src/paths.py`; beide Mapping-Stände unter `data/metadata*/` **ohne** `specimen_dataset.csv` und zunächst ohne `feature_overlap.json`; zugehörige Mapping-/Referenztests; `docs/specimen_mapping.md`; `.gitignore` |
| 2 | `feat(experiments): add reproducible specimen holdout baselines` | `configs/specimen_holdout_v1/`; `src/data/specimen_splits.py`; `src/features/{specimen_tsfresh,specimen_manual}.py`; `src/experiments/specimen_baseline.py`; Snapshot-Hilfe in `provenance.py` und deren Tests; Split-/Feature-/Baseline-Tests; `docs/specimen_holdout.md`; `docs/evidence/specimen_holdout_v1/` |
| 3 | `feat(analysis): compare manual and tsfresh feature sets` | `src/analysis/{feature_overlap,specimen_feature_comparison}.py`; zugehörige Tests; `data/metadata_reference_excluded/feature_overlap.json`; `docs/{feature_overlap,specimen_manual52_comparison}.md`; `docs/evidence/specimen_manual52_v1/` |
| 4 | `feat(analysis): review specimen errors and surface anchors` | `src/analysis/{specimen_error_analysis,surface_anchor_review}.py`; beide Testdateien und Berichte; Evidenzordner `herai_error_analysis_20261110/`, `surface_anchor_review_20261110/` |
| 5 | `feat(experiments): evaluate alternative anchors with group CV` | `src/features/experimental_*.py`; `src/experiments/anchor_group_cv.py`; Ankertests; `configs/anchor_group_cv_v1/`; `tests/data/test_frozen_cv_metadata.py`; `docs/experimental_anchor_group_cv.md`; zugehörige Evidenz |
| 6 | `feat(experiments): assess uncertain class two with paired group CV` | `src/experiments/class2_group_cv.py`; `src/analysis/{class2_training_review,verify_class2_group_cv}.py`; Klasse-2-Tests; `docs/class2_training_group_cv.md`; zugehörige Evidenz |
| 7 | `chore(repo): document experiment inventory and curate review evidence` | README; dieser Bericht, `docs/experiments.md` und `docs/audit/2026-11-10/git_preparation.json`; `docs/evidence/{README.md,manifest.json}` und beide historischen tsfresh-Ergebnisauszüge; `configs/review_evidence.json`; Exportskript und dessen Tests; Snapshot-Anbindung des alten `model_comparison.py`; aktuelle Hinweise in `docs/tsfresh_comparison.md` und `scripts/reproducibility/README.md` |

Diese Tabelle ist keine Aufforderung zu `git add .`: vor jedem späteren Commit
die konkreten Pfade und den staged Diff prüfen. Insbesondere können großzügige
Verzeichnisangaben lokale Kataloge einschließen, wenn Ignore-Regeln umgangen werden.
Einzelne thematische Zwischenstände wurden nicht als separate Git-Commits erzeugt
oder getestet; geprüft wird der vollständige vorbereitete Arbeitsstand.

## Qualitätsprüfung

- Ruff mit der unveränderten Konfiguration `F`, `I`, `E9`.
- Gesamte pytest-Suite einschließlich synthetischer Pipeline-/Serialisierungstests.
- Zusätzliche Prüfumgebung als Kopie des potenziellen Commitinhalts, ohne ignorierte
  Rohdaten, Feature-Caches, neue Modelle oder lokale Run-Verzeichnisse. Gleiche
  konfigurierte `.venv`, keine neue Installation und kein Training realer Modelle.
- Syntaxprüfung aller Python-Dateien des Commitinhalts, statische Prüfung lokaler
  Importziele und reale Imports der 23 neuen/geänderten Pipeline-Module.
- `pip check`, Link-/Evidenzprüfung, Größe neuer Dateien, gezielte Suche nach
  Zugangsdatenmustern sowie `git diff --check`. Die Mustersuche ist keine vollständige
  Garantie, dass jede denkbare vertrauliche Information erkannt wurde.
- Hashvergleich wissenschaftlicher Bestände und Kontrolle, dass HEAD, Branch und
  vorgemerkter Diff unverändert sind. Details des abgeschlossenen Prüfstands stehen unten.

### Abgeschlossener Prüfstand

- **182 Tests bestanden, sieben bekannte Warnungen** in der datenarmen Prüfkopie
  des vollständigen Commitinhalts. Im Arbeitsverzeichnis zuvor 181 Tests und danach
  der ergänzte reine Evidenztest separat bestanden. Keine realen Modelltrainings;
  kleine synthetische Fits bleiben Bestandteil der bestehenden Tests.
- Ruff, `pip check` und `git diff --check` ohne Befund. **99 Python-Dateien**
  kompiliert; 232 lokale Importreferenzen geprüft; keine fehlenden lokalen Module.
  **23 Pipeline-Module** in der datenarmen Kopie erfolgreich importiert.
- **56 Dokumentationslinks** dort gültig; alle 57 Evidenzdateien stimmen mit ihren
  Exporthashes überein, ebenso deren lokale Originale mit den Quellhashes.
- **75.946 wissenschaftliche Bestandsdateien byteidentisch**. Historische
  Tracking-Kopien zusätzlich gegen vorherige gespeicherte Hashes geprüft.
- Keine neu aufzunehmenden Modell-, Datenbank- oder Cache-Dateien und keine neuen
  Dateien über 10 MB. Die gezielte Zugangsdatenmustersuche hatte keine Treffer;
  bestehende historische Messpfade sind ausdrücklich von dieser Aussage zu unterscheiden.
- HEAD, Branch und vorgemerkter Diff unverändert. Es wurde nichts gestaged.

[Maschinenlesbarer Prüfbericht](audit/2026-11-10/git_preparation.json).
Vollständige lokale Prüfprotokolle, Inventar und Bestandshashes liegen im Pflegearchiv.

## Vor einem späteren Push beachten

1. **Lokales Backup bleibt erforderlich.** Große Originalruns (zusammen mehrere GB),
   Rohdaten und Tracking-Stores sind absichtlich nicht Bestandteil des Commitinhalts.
   Die neue lokale Sicherung ist ebenfalls kein externes Backup.
2. **Absolute historische Pfade bleiben erhalten.** Bereits versionierte Transformer-
   und Isolation-Forest-CSV-Dateien enthalten lokale Dateipfade. Sie wurden zum Schutz
   der Historie nicht bereinigt. Vor öffentlicher Veröffentlichung die Sichtbarkeit
   dieser bereits bestehenden Inhalte ausdrücklich prüfen. Neue Auszüge enthalten
   keine solchen Messpfade.
3. **Kein plattformübergreifender Reproduktionsnachweis.** Tests verwenden die vorhandene
   macOS-Umgebung; die eigentliche Ubuntu-CI und eine frische Paketinstallation werden
   erst später separat ausgeführt. `pip check` ersetzt keinen wissenschaftlichen Lock.
4. **Strikte Quellhashes sind beabsichtigt.** Importbereinigung und Snapshot-Hilfe ändern
   aktuelle Quellhashes. Historische Cache-/Auditprüfungen können deshalb mit aktuellem
   Code absichtlich abbrechen. Bestehende Manifeste wurden nicht angepasst; für exakte
   historische Prüfung den archivierten ausgeführten Quellstand verwenden. Neue
   wissenschaftliche Läufe benötigen neue Ausgabeziele und ihre eigene Provenienz.
5. **Legacy-Skripte sind teilweise ausführbare Programme beim Import.** Ältere Plot-/
   Analyse- und Trainingsmodule werden nicht pauschal importiert oder gestartet,
   weil dadurch historische Ausgaben überschrieben oder Trainings ausgelöst werden
   könnten. Sie bleiben dokumentiert; die neuen Pipeline-Module werden separat geprüft.

Die bekannten wissenschaftlichen Grenzen bleiben bestehen: unklare physische
Identität zwischen Reihen, wenige unabhängige Körper, schwache Klasse-2-Erkennung,
problematische Messreihe 3 und explorative Auswahl der frühen Ankerung.
