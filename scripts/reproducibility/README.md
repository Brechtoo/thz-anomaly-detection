# Historische HGB-Prüfungen

Alle vier Skripte erwarten explizite Eingaben und ein neues `--output-dir`. Mit `.venv/bin/python scripts/reproducibility/<skript> --help` die Argumente anzeigen.

| Skript | Prüfung |
| --- | --- |
| `feature_probe.py` | 64 gleichmäßig verteilte Katalogzeilen gegen die Feature-Referenz |
| `verify_features.py` | Vollständiger Neuaufbau und Vergleich mit ursprünglichem Kriterium 1e-12 |
| `verify_model_consistency.py` | Gespeichertes HGB auf Referenz und Neuaufbau, mit rtol=1e-11 und atol=1e-13 |
| `audit_catalog_leakage.py` | Katalog aus Originalannotationen, Split-Überlappung und Duplikate |

Die HGB-Inferenzprüfung benötigt den historischen Datenstand einschließlich Metallreferenzen. Das heutige Prüfskript dafür in einem separaten Checkout von `baseline-audit-2026-11-02` unter demselben relativen Pfad ausführen, Eingaben absolut angeben und die historische `mlflow.db` für die Hashprüfung bereitstellen. Für die Bewertung die erzeugten JSON-/CSV-Vergleiche lesen. [Audit und Archivpfade](../../docs/reproducibility_checks.md)
