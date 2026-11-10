# Experimentkatalog und Bestandserhaltung

Stand: 10.11.2026. Aktuelle Software und historische wissenschaftliche Quellen
werden getrennt beurteilt: Die ausgeführten Quellen bleiben in den Originalruns,
auch wenn Imports im Arbeitscode für CI bereinigt wurden. Kein bestehender Run
wurde im Rahmen der Git-Vorbereitung verändert oder erneut trainiert.

## Abgeschlossene neue Untersuchungen

| Untersuchung | Bericht / kompakter Nachweis | Vollständiger lokaler Bestand | Bewertungsgrundlage |
| --- | --- | --- | --- |
| Räumliche Gruppen, Metallreferenzen | [Zuordnung](specimen_mapping.md), `data/metadata_reference_excluded/` | `data/metadata/` bleibt historischer Vorzustand | Koordinaten; 35 Körper + 3 Referenzen, keine Labeländerung |
| 52 vs. 701 Featureformeln | [Überlappung](feature_overlap.md) | `data/metadata_reference_excluded/feature_overlap.json` | Formelanalyse, keine neue Modellbewertung |
| Prüfkörper-Holdout, 708 tsfresh-Features | [Bericht](specimen_holdout.md), [Metriken](evidence/specimen_holdout_v1/baseline/results.csv) | `results/experiments/specimen_holdout_v1/` | 23/5/7 Körper; neue train-only Selektion |
| Manuelle 52 vs. 708 Features | [Vergleich](specimen_manual52_comparison.md), [Validation](evidence/specimen_manual52_v1/validation_comparison.csv) | `results/experiments/specimen_manual52_v1/` | Identische Holdouts; Test ausdrücklich explorativ |
| Fehleranalyse der sieben manuellen Modelle | [Bericht](herai_error_analysis_20261110.md), [Fehlertabelle](evidence/herai_error_analysis_20261110/specimen_errors.csv) | `results/analysis/herai_error_analysis_20261110/` | Gespeicherte Validation-/Testvorhersagen; keine Fits |
| Signal-/Oberflächenankerdiagnostik | [Bericht](surface_anchor_review_20261110.md) | `results/analysis/surface_anchor_review_20261110/` | Rohsignale/Annotationen; keine Features oder Modelle geändert |
| Vier Anker-/Fenstervarianten | [Bericht](experimental_anchor_group_cv.md), [OOF](evidence/anchor_group_cv_v1/cv/pooled_oof_metrics.csv) | `results/experiments/anchor_group_cv_v1/` | 84 Fits, ausschließlich 23 Trainingskörper; frühe Ankerung stärkster interner Kandidat |
| Training mit/ohne Klasse 2 | [Bericht](class2_training_group_cv.md), [Zielmetriken](evidence/class2_group_cv_v1/known_metrics.csv) | `results/experiments/class2_group_cv_v1/` | 42 Fits, gleiche Trainingskörper/Folds; kritische 1→3-Fehler steigen ohne 2 |

Die drei Körper-Folds sind zusätzlich pfadfrei unter
`configs/anchor_group_cv_v1/specimen_folds.csv` gespiegelt (23 Zeilen mit ID/Fold).
Zusammen mit dem eingefrorenen Train-Membership ergibt das exakt die bisherigen
34.235 OOF-Punkte. Das neue Sidecar ist eine Kopie der vorhandenen Gruppenzuordnung,
keine neue Zufallsziehung und keine Änderung eines Split-Manifests.

## Historische Referenzen – auf ihren bisherigen Pfaden erhalten

| Familie | Einstieg / Ergebnisse | Status und Grenzen |
| --- | --- | --- |
| Baseline-/erweiterte manuelle Features | `src/experiments/`, `results/experiments/baseline_features/`, `extended_features/` | Historische punktweise Splits, Optimierung und Schwellenwerte; kein unabhängiger Körpernachweis |
| Eingefrorene HGB-Baseline und Audit | [Betriebsleitfaden](hgb_reproducibility.md), [Audit/Archiv](reproducibility_checks.md), `configs/hgb_baseline/splits/`, `hgb_baseline_reproducibility/`, `hgb_reproduction_20261102T204153Z_155d895a/` | Referenzmodelle, Artefakte und Manifesthashes unverändert; Rohdatenarchiv zusätzlich außerhalb des Repositorys |
| 701-tsfresh-Modellvergleich | [Methodik](tsfresh_comparison.md), [vollständiger Folgelauf](evidence/tsfresh_comparison_20261106T160800Z_5b61f342/validation_results.csv) | Erster Lauf `20261106T151623Z_5b9eb97a` hatte Speicherfehler; Folgelauf `20261106T160800Z_5b61f342` mit sieben FINISHED-Modellen bleibt maßgebliche historische Tabelle |
| Transformer-Gewichtungsvergleich | `src/transformer/`, `results/transformer/transformer_weight_comparison.csv` und zugehörige `.pt`/CSV | Historische Linie, keine nachträgliche gruppenbasierte Evaluation behauptet |
| PCA, Isolation Forest, Autoencoder | `src/unsupervised/`, `results/unsupervised/` | Historische Modell-/Scoreberichte; nicht mit den neuen sieben überwachten Modellen gleichsetzen |
| Externe HolzElement-Auswertung | `src/generalization/`, `results/generalization/` | Historische Summary-/Kartenbestände für überwachte, Transformer- und unüberwachte Verfahren; unabhängige Referenzlabels nicht aus diesen Karten ableitbar |
| Deskriptive Features / Permutationsanalyse | `src/analysis/`, `results/analysis/feature_baseline_analysis/`, `permutation_importance/` | Historische Diagnostik; keine neue Featureselektion oder Rangfolge daraus abgeleitet |

Einige ältere Linien besitzen nur Code und Ergebnisdateien, keinen vollständigen
maschinenlesbaren Run-/Umgebungsnachweis. Dieser Bestand wurde katalogisiert,
nicht rückwirkend als vollständig reproduzierbar erklärt. Frühere `rgy_v1`-Zahlen
bleiben erhalten; ihre Abweichung zu [four_label_v2](error_metrics.md) bleibt sichtbar.

## Aufbewahrung und aktive Pfade

- **Aktiver Katalog:** `data/measurements/combined_dataset.csv`. Die zentrale
  Pfadkonfiguration ist darauf vereinheitlicht. `results/measurements/` ist ein
  historischer Pfad in alten Berichten/Manifesten und bleibt ebenfalls ignoriert.
- **Aktuelle Gruppen:** `data/metadata_reference_excluded/specimen_mapping.csv`.
  Der dortige `specimen_dataset.csv` ist ein lokal regenerierbarer Katalog mit
  absoluten Rohpfaden und bleibt außerhalb von Git. Mapping-/Manifestdateien bleiben erhalten.
- **Tracking/Modelle:** `mlflow.db`, `mlruns/`, `src/experiments/mlruns/` und die
  run-eigenen Datenbanken/Artefaktordner werden nicht verschoben. Alte MLflow-
  Artefakt-URIs können an diese Pfade gebunden sein. Neue Quellsnapshots schließen
  Tracking-Stores und Caches aus, um weitere rekursive Kopien zu verhindern.
- **Lokales Pflegearchiv:** `archive/maintenance_2026-11-10/` enthält den zuvor
  uncommitteten Quell-/Dokumentstand, Git-Vorzustand und Bestandsschutz-Prüfsummen.
  Es wird nicht veröffentlicht und ersetzt kein externes Backup.
- **Vorhandenes externes Archiv:** `../thz-archive/`, siehe [Archivzuordnung](reproducibility_checks.md).
  Es wurde nicht verschoben oder verändert; historische Originalpfade bleiben historisch.

Das Projekt behält aktive Einstiegspfade bewusst bei. Gleichnamige Skripte oder
Quellen in historischen Tracking-Snapshots sind Provenienz, keine zu löschenden
Duplikate. Unklare Altbestände bleiben erhalten. Die [Git-Prüfvorlage](git_preparation.md)
beschreibt konkrete Änderungen, Commit-Reihenfolge und verbleibende Grenzen.
