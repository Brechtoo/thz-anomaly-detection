# Kompakte Experimentnachweise

Diese Auswahl enthält **57** kleine Ergebnisdateien (5.476.282 Bytes) für die
Begutachtung im Repository. Vollständige Originale bleiben unter den in
`manifest.json` dokumentierten lokalen `results/`-Pfaden erhalten. Es wurden keine
Ergebnisse neu berechnet oder Modelle trainiert.

`manifest.json` nennt Quellpfad, SHA256 des Originals, SHA256 des Exports, Größe
und gegebenenfalls ausgewählte CSV-Spalten. Die beiden historischen tsfresh-
Vergleichstabellen sind ausdrücklich **Spaltenprojektionen** ohne Fehlermeldungstexte;
die übrigen 55 Dateien sind byteidentische Kopien. Es gibt keine neue Bewertung
oder Aktualisierung historischer MLflow-Runs. Artefaktstatus/-prüfungen beziehen
sich auf den jeweils ursprünglichen Experimentstand.

| Auswahl | Inhalt |
| --- | --- |
| `tsfresh_comparison_20261106T151623Z_5b9eb97a/` | Erster, teilweise an Speicherung gescheiterter Vergleich; kein vollständiger Baselineersatz |
| `tsfresh_comparison_20261106T160800Z_5b61f342/` | Erfolgreicher historischer Vergleich mit 701 Features |
| `specimen_holdout_v1/` | 708-Feature-Holdout, Körper-/Quellenmetriken und damalige Verifikation |
| `specimen_manual52_v1/` | 52-Feature-Holdout, Validierungsvergleich und ausdrücklich explorativer Testvergleich |
| `herai_error_analysis_20261110/` | Fehlerkonzentration, Modellüberlappung und Grafiken |
| `surface_anchor_review_20261110/` | Signal-/Annotationsdiagnostik und Abbildungen |
| `anchor_group_cv_v1/` | 84-Fit-Ankervergleich: Pool-/Fold-/Körpermetriken, Diagnose und Grafik |
| `class2_group_cv_v1/` | 42-Fit-Klasse-2-Vergleich: Zielmetriken, Zuordnungen, Bootstrap, Dichtediagnose und Grafiken |

Die Datenaufteilungen und Generalisierungsziele unterscheiden sich. Zahlen dürfen
nicht ohne den jeweiligen [Bericht](../experiments.md) gegeneinander gerankt werden.
Tabellen verwenden je nach ursprünglichem Bericht Raten in Prozent oder als Anteil.

Die Auswahl ist in `configs/review_evidence.json` explizit eingefroren. Ein erneuter
Export benötigt vorhandene lokale Originale und **ein neues Ausgabeziel**:

```sh
.venv/bin/python -m scripts.maintenance.export_evidence \
  --specification configs/review_evidence.json --output-dir /path/to/new-evidence
```

Eine Kopie dieser Auswahl ersetzt weder die lokalen Modelle/Tracking-Stores noch
Rohdaten oder Feature-Caches. Vor einem späteren Verlust des Rechners sind diese
Bestände separat zu sichern. Keine Dateien daraus wurden hier entfernt.
