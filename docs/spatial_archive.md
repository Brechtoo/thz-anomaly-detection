# Archiv der räumlichen Untersuchung

Der Branch `archive/spatial-investigation-2026-11-24` hält den
Quellstand der Radius- und Achsen-Ablationen, der Nested-Suche und des
räumlichen Postprocessings fest. Die
[Ergebnisse der Eingabefeatures](spatial_svm_features.md) und des
[Postprocessings](spatial_postprocessing.md) identifizieren die einzelnen
Läufe und ihren Auswertungsumfang.

Das lokale, Git-ignorierte Paket
`archive/spatial_investigation_2026-11-24/artifacts.tar.gz`
enthält 1.574 Dateien. Dazu gehören Vorhersagen, Fit-Checkpoints,
Laufpläne, Logs, Quellenkopien, der kombinierte Bericht, lokale
MLflow-Stores und die in den Laufplänen referenzierten Eingabeartefakte.
Die Wiederherstellungsprüfung umfasst 719 archivierte Eingabe- und
Referenzprüfsummen. Der Paket-SHA-256 lautet `dd5a6bdf3be95aca3c30a83c025cac6a80987655cd94a8947d91fc7647190cbc`.
Abgebrochene temporäre Kernelmatrizen sind reproduzierbare Arbeitsdateien.
Die Originalartefakte und die Sicherung befinden sich lokal.

Zur Wiederherstellung den Archiv-Branch auschecken und aus dem
Projektstammverzeichnis ausführen.

```sh
bundle=archive/spatial_investigation_2026-11-24
shasum -a 256 -c "$bundle/bundle-sha256.txt"
tar -xkzf "$bundle/artifacts.tar.gz"
shasum -a 256 -c "$bundle/input-sha256.txt"
```

Das Tar-Kommando schützt bereits vorhandene Dateien. Die Laufpläne
verwenden die ursprünglichen absoluten Pfade. Für die unveränderte
Wiederaufnahme der archivierten Läufe ist der ursprüngliche Projektpfad
erforderlich. Die verwendeten Paketversionen stehen in den Laufplänen.
