# Ergebnisnachweise

Die 59 Dateien dokumentieren den historischen tsfresh-Vergleich, Prüfkörper-Holdouts, Fehler- und Ankeranalysen sowie die Klasse-2-Ablation, Freigabeschwellen und Zusatzprüfung. Die beiden historischen tsfresh-Tabellen enthalten ausgewählte Spalten. Die übrigen 57 Dateien sind byteidentische Kopien. Quellen, Spalten und SHA-256 stehen im [Manifest](manifest.json).

Ein erneuter Export benötigt die lokalen Originale und ein neues Ziel:

```sh
.venv/bin/python -m scripts.maintenance.export_evidence \
  --specification configs/review_evidence.json --output-dir /tmp/herai-evidence
```
