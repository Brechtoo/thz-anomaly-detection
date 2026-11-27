# Datenstände und Wiederholung

## Welche Eingaben gehören zusammen?

| Versuch | Eingaben und Nachweis |
| --- | --- |
| Punktweise HGB-Baseline | 53.192 Messungen einschließlich Metallreferenzen. Geordnete IDs unter `configs/hgb_baseline/splits/`, exakte Feature-Datei und 101 Paketversionen im historischen Lock. Für die ursprüngliche Umgebung gilt der Tag `baseline-audit-2026-11-02`. |
| Prüfkörper-Holdout | `data/metadata_reference_excluded/` enthält die Zuordnung der 52.220 Punkte. `configs/specimen_holdout_v1/` fixiert 23/5/7 Körper, Feature-Schema und Umgebung. Die älteren Metadaten unter `data/metadata/` enthalten auch die Referenzflächen. |
| Gruppen-CV | `configs/anchor_group_cv_v1/` ordnet die 23 Trainingskörper drei Folds zu. Messungs-ID, Reihenfolge, Klasse und Prüfkörper müssen zu den archivierten Vorhersagen passen. Der gemeinsame Seed der Gruppenexperimente ist 42. |
| Verschachtelte Suche | Die lokalen Ordner `results/experiments/specimen_{feature_grid,svm_nested_hpo,tabular_nested_hpo}_v1/` enthalten Pläne, Paketversionen, Checkpoints und Auswahlentscheidungen. Die innere Gruppenzuordnung stammt aus dem SVM-Lauf. Die Kalibrationskorrektur für Float32-XGBoost-Scores steht in `calibration_precision_revision.json`. |
| Freigaberegeln | `results/analysis/nested_release_review_v1/`, `nested_release_pareto_v1/` und `specimen_ood_review_v2/` enthalten die Auswertung der Trainingskörper. Vollständige Artefakte und MLflow-Stores liegen lokal. |

Der historische Audit rekonstruierte den Katalog zweimal byteidentisch. Die 52 Merkmale stimmten bei `rtol=1e-11, atol=1e-13` überein. Vier Werte verfehlten das ursprüngliche strengere Kriterium. Das gespeicherte HGB lieferte auf beiden Tabellen dieselben Wahrscheinlichkeiten und Entscheidungen für alle 8.511 Validierungspunkte. Referenzlauf `a5b8cf67d02d4214bb8e36ea946e839a`, Trainingswiederholung `674a859ff5b94331843ce811abda18a0`. Das belegt die damalige Umgebung und den geprüften Datenstand. [Audit mit Prüfsummen](audit/2026-11-02/summary.json) · [Prüfskripte](../scripts/reproducibility/README.md)

## Einen Lauf nachvollziehen

Die vier historischen Auditverzeichnisse liegen unter `../thz-archive/reproducibility_checks/2026-11-05/`. Die zugehörigen SHA-256 stehen in `../thz-archive/archive_sha256_2026-11-05.json`. Manche Pläne referenzieren absolute Originalpfade. Ein frischer Checkout benötigt zusätzlich die darin genannten Rohdaten, Tabellen oder Checkpoints. [Räumliches Archiv](spatial_archive.md)

Für die neueste Fehleranalyse genügen die Trainingsartefakte der Grid-, SVM-, tabellarischen und Pareto-Läufe sowie die frühen manuellen Merkmale. Der Aufruf liest die gespeicherten Modellscores, passt die Zusatzdetektoren auf Trainingskörpern an und schreibt Ergebnisse und Tracking in ein neues Ziel.

```sh
.venv/bin/python -m src.analysis.specimen_ood_review --output /tmp/herai-ood-repeat
```

Die Studienprotokolle mit den damaligen Befehlen und Laufpfaden sind lokal im Git-Archiv `archive/exploration-2026-11-26` (Commit `6958f37`) erhalten. Alle Berichtsnamen und einen einzelnen Bericht anzeigen:

```sh
git ls-tree --name-only 6958f37:docs
git show 6958f37:docs/specimen_feature_grid.md
```

Das historische Grid schrieb die Kohorte ohne pandas-Index und beanstandete sie deshalb beim Wiederanlauf. Für neue Läufe vergleicht der Runner die gespeicherten Spalten samt Messungsreihenfolge. Die Originalpläne und die historische OOF-Schwellenprüfung verlangen den damaligen Quellhash. Dafür den Archivstand verwenden. Der reguläre Grid-Runner öffnet ohne `--cv-only` anschließend die Holdouts.

## Prüfung bei der Übergabe am 27.11.2026

Mit der Projekt-venv (Python 3.12.5) ergab der Referenzfit der logistischen Regression auf manuellen Merkmalen in Außenfold 0, Innenfold 0 denselben Vorverarbeitungszustand wie der archivierte Checkpoint. Nach erneutem Fit auf 14.885 Trainingspunkten stimmten alle 7.375 Entscheidungen und sämtliche Log-Scores exakt überein. Die erneute OOD-Zusatzprüfung reproduzierte `specimens.csv`, `selection.csv`, `totals.csv` und `train_oof.parquet` byteidentisch. Die Prüfung umfasst diesen Fit und die Zusatzprüfung auf den Trainingskörpern. Die gesamte Parametersuche wurde nicht wiederholt.
