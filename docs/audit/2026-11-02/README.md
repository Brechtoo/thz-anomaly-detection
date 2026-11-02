# Technischer Baseline- und Audit-Nachweis

**Keine gruppenunabhängige Generalisierung nachgewiesen.**

Dieser Ordner enthält kompakte, portable Ergebnisse der lokalen Prüfungen.
Er enthält keine Rohsignale, pro Messung aufgelisteten Dateipfade, Modelle,
virtuellen Umgebungen oder MLflow-Datenbanken.

- `summary.json`: Kennzahlen, Referenz-/Neuaufbauhashes, Validierungsvergleich,
  Katalogrekonstruktion und Gruppenbefunde mit ausdrücklichen Grenzen.
- `feature_differences.csv`: Anzahl sowie maximale absolute/relative Differenz
  je Feature. Relativer Nenner ist der Betrag des Referenzwerts.
- `groups_source_dataset.csv`: bestehende Split-Anteile je Messreihe.
- `source_split_label_counts.csv`: aggregierte Klassenhäufigkeiten.
- `installed_distributions_sha256.json`: Paketnamen, Versionen und Download-
  SHA256 aus dem erfolgreichen frischen Installationsbericht. Ergänzung zum
  Versionslock, keine Behauptung plattformübergreifender Reproduzierbarkeit.

Der vollständige lokale Nachweis bleibt in den in `summary.json` angegebenen
ignorierten Ausgabeordnern erhalten. Die Split-Mitgliedschaften liegen separat
unter `configs/hgb_baseline/splits/`. Sie enthalten IDs und Labels, keine Signale
oder Features. Historische Runs werden durch diesen Nachweis nicht geändert.

Die Detailberichte sind über [den Betriebsleitfaden](../../hgb_reproducibility.md)
verlinkt. Die Primärquelle der Ergebnisse sind die bestehenden Prüfungsdateien;
für diese Zusammenfassung wurde kein Experiment erneut ausgeführt.
