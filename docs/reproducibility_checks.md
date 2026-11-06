# Reproduzierbarkeitsprüfungen vom 02.11.2026

Zentraler Überblick über die historischen Prüfungen. Die Aufräumaktion vom
05.11.2026 hat keine Experimente wiederholt, Modelle trainiert oder wissenschaftlichen
Prüfkriterien, Daten, Splits und Baselines verändert.

| Prüfung | Wesentliches Ergebnis |
| --- | --- |
| Frische Installation | Python 3.12.5, macOS arm64; alle 101 Distributionen entsprechen dem Lock; `pip check` ohne Befund. |
| Tests | 74 Tests in der frischen Umgebung, später 92 nach Katalogergänzungen und abschließend 94 bestanden; jeweils acht bekannte sklearn-Einzelklassenwarnungen. Historische, unterschiedliche Code-Stände. |
| Katalogrekonstruktion | Alle 53.192 Messungen aus Originalannotationen zugeordnet; zwei Neuaufbauten byteidentisch zur Referenz; keine neuen Labels. |
| Feature-Neuaufbau | Schema, IDs, Labels und Metadaten identisch; 52 Features numerisch reproduziert mit `rtol=1e-11`, `atol=1e-13`, aber nicht byteidentisch. 1.381.899 Zellen unterscheiden sich; maximal absolut ca. `2.80e-11`. Vier Ausreißer des ursprünglichen Kriteriums `rtol=atol=1e-12` bleiben dokumentiert. |
| Gespeichertes HGB | Alle 8.511 Validierungsentscheidungen und Wahrscheinlichkeiten auf Referenz und Neuaufbau exakt gleich; Accuracy 0,88838; Sicherheitsmetriken unverändert. |
| Gruppen und Duplikate | Alle drei Messreihen in Train/Validation/Test. 97,6 % der Validierungs- und 98,2 % der Testpunkte haben einen Trainingsnachbarn derselben Quelle im Abstand ≤1 Koordinateneinheit. Keine exakten Rohdatei-/Feature-Duplikate gefunden. |
| Bestandsschutz | Im historischen Sessionabschluss 137 geschützte Dateien sowie Annotationen, Split-Manifeste und Paket-Records unverändert geprüft. |

## Grenzen

Die Ergebnisse stützen eine technische, punktweise Within-Scan-Baseline.
Generalisierung auf unabhängige neue Scans oder Prüfkörper ist nicht belegt;
die Größe eines möglichen Optimismus durch Gruppenüberlappung wurde nicht ermittelt.
Autoritative Aufnahmegruppen und die historische Feature-Extraktionsumgebung
sind unvollständig dokumentiert. Nahezu gleiche, skalierte oder zeitverschobene
Signale wurden nicht erschöpfend geprüft. Plattformübergreifende Bytegleichheit
und identisches erneutes Training auf der neu extrahierten Tabelle sind nicht belegt.
Die Feature-Toleranz gilt für den geprüften Datenstand; der strikte Hashschutz
der Baseline bleibt unverändert.

## Detailnachweise

- [Maschinenlesbarer historischer Nachweis](audit/2026-11-02/summary.json): Kennzahlen, Hashes, Run-IDs, Gruppenbefunde und Sessionabschluss.
- [Differenzen je Feature](audit/2026-11-02/feature_differences.csv).
- [Klassenhäufigkeiten je Quelle und Split](audit/2026-11-02/source_split_label_counts.csv).
- [Installierte Distributionen mit Download-Hashes](audit/2026-11-02/installed_distributions_sha256.json).
- [Prüfsummen dieser Detailnachweise](audit/2026-11-02/checksums.json).
- Methodik: [ursprüngliche Prüfung](hgb_reproducibility_verification.md), [numerische Bewertung und Modellvergleich](feature_numerical_reproducibility.md), [Katalog und Leakage](catalog_reconstruction_and_leakage.md).
- [Prüfskripte und Aufrufhinweise](../scripts/reproducibility/README.md); [Betriebsleitfaden](hgb_reproducibility.md).

## Vollständiges lokales Archiv

Archiv außerhalb des Repositorys, relativ zur Projektwurzel:

- `../thz-archive/reproducibility_checks/2026-11-05/`: vollständige bisherige vier Prüfverzeichnisse, einschließlich generierter Tabellen, Logs, Manifeste und ursprünglicher Skripte.
- `../thz-archive/docs-audit/2026-11-05/`: unveränderte Sicherung aller sieben bisherigen Audit-Dateien.
- `../thz-archive/archive_sha256_2026-11-05.json`: SHA256 aller 60 archivierten Dateien, mit ursprünglichen Projektpfaden als Schlüsseln.

Vor dem Entfernen der Projektkopien wurden Dateilisten und sämtliche SHA256
zwischen Quelle und Archiv verglichen. Die vier bisherigen Verzeichnisnamen
bleiben im Archiv erhalten. Alte Pfadangaben in historischen Logs, Manifesten
und `summary.json` beschreiben den damaligen Standort; sie sind keine aktiven
Projektabhängigkeiten. `results/reproducibility_checks/<Unterpfad>` wird jetzt auf
`../thz-archive/reproducibility_checks/2026-11-05/<Unterpfad>` abgebildet.
Das in `summary.json` ebenfalls genannte `results/experiments/hgb_reproduction_*`
wurde nicht verschoben.

Die redundante Tabelle `groups_source_dataset.csv` liegt vollständig im Archiv;
ihre Zahlen stehen weiterhin in `summary.json` unter `leakage.source_split_counts`.
Das Archiv enthält lokale Messpfade und generierte Daten und wird nicht veröffentlicht.
