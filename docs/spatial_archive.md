# Räumliche Versuche wiederherstellen

Der Quellstand liegt auf `archive/spatial-investigation-2026-11-24`. Das lokale Paket enthält 1.574 Dateien mit Eingaben, Vorhersagen, Checkpoints, Berichten und Tracking. Für eine Wiederherstellung im passenden Checkout:

```sh
bundle=archive/spatial_investigation_2026-11-24
shasum -a 256 -c "$bundle/bundle-sha256.txt"
tar -xkzf "$bundle/artifacts.tar.gz"
shasum -a 256 -c "$bundle/input-sha256.txt"
```

Der Paket-SHA-256 ist `dd5a6bdf3be95aca3c30a83c025cac6a80987655cd94a8947d91fc7647190cbc`. Die 719 Eingabe- und Referenzprüfsummen gehören zum ursprünglichen Projektpfad. Die Pläne enthalten absolute Pfade.
