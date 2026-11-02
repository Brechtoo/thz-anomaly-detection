# Sessionabschluss: technischer Baseline- und Audit-Stand

Tag: `baseline-audit-2026-11-02`, Branch: `feature/baseline-reproducibility`.

**Keine gruppenunabhängige Generalisierung nachgewiesen.** Dieser Stand sichert
die technische Reproduzierbarkeit der bisherigen punktweisen Within-Scan-
Baseline und dokumentiert die wissenschaftlichen Grenzen.

## Prüfung und Aufräumen

- Gesamten Arbeitsstand geprüft, einschließlich neuer Dateien und der zuvor
  vorgemerkten Agentenregeln. Keine unbeabsichtigten Modell-, Split- oder
  Featuredefinitionsänderungen. `trapz` → `trapezoid` bewahrt die Trapezregel.
- Schutz vor Fortsetzung historischer MLflow-Runs ergänzt: vorhandenes
  `MLFLOW_RUN_ID` oder aktiver Run wird vor jeder Ausgabe/Trainingsaktion abgelehnt.
- `.gitignore` schützt Roh-/Externdaten, Umgebungen, neue generierte Audit-
  Verzeichnisse, MLflow-Dateien und lokale `.env`-Konfigurationen.
- Vollständige lokale Nachweise bleiben erhalten. Nur redundante eigene
  temporäre Skriptkopien und der isolierte Installations-Venv wurden nach
  Sicherung der Metadaten entfernt. Historische Daten/Modelle/Runs unverändert.
- Versionskontrolle enthält Code, Lock, geordnete Split-IDs/Labels und kompakte
  Audit-Tabellen, keine neuen Rohsignale, Feature-Tabellen oder Modelle.
  Bereits zuvor versionierte historische Ergebnisse wurden nicht entfernt.
- Redundante Betriebsdokumentation zusammengeführt; Detailberichte bleiben
  über den [Betriebsleitfaden](hgb_reproducibility.md) auffindbar. Kompakte,
  pfadbereinigte Nachweise liegen unter [docs/audit/2026-11-02](audit/2026-11-02/README.md).

## Abschließende Verifikation

`.venv/bin/python -m pytest -q --tb=short`: **94 bestanden**, acht bekannte
sklearn-Einzelklassenwarnungen, Dauer 5,39 Sekunden. Einstiegstests verwenden
Fake-Training und Fake-MLflow; für den Sessionabschluss gab es kein Modelltraining.
137 historische Datei-Hashes einschließlich MLflow-Datenbank, Originalannotationen,
Split-Manifeste und Paket-Records der bestehenden `.venv` blieben unverändert.
Die ausgewählte Veröffentlichung wurde auf große Artefakte und typische
Zugangsdatenmuster geprüft; dies ersetzt keine universelle Geheimniserkennung.

## Logische Veröffentlichungseinheiten

1. Agentenregeln und Schutz lokaler/generierter Dateien.
2. Gesperrte HGB-Umgebung, gespeicherte Splits, Provenienz und Schutz neuer Runs.
3. Überschreibsichere Katalog-/Feature-Rekonstruktion und Tests.
4. Leakage-Diagnostik, wissenschaftliche Einordnung und kompakte Nachweise.

Der zuvor erstellte Commit für `four_label_v2` gehört ebenfalls zum veröffentlichten
Branch. Der annotierte Tag kennzeichnet ausdrücklich eine technische Baseline
mit Audit, keine Freigabe als leakage-freie Generalisierungsstudie.

## Verbleibende wissenschaftliche Einschränkungen

Alle drei Messreihen liegen in Train/Validation/Test; rund 97,6 % der Validation
und 98,2 % der Testpunkte besitzen unmittelbare Trainingsnachbarn derselben Reihe.
Eine separate gruppenunabhängige Evaluation auf neuen Aufnahmen/Prüfkörpern
bleibt offen. Dafür müssen tatsächliche Gruppen und das Generalisierungsziel
festgelegt werden. Historische Splits werden nicht stillschweigend geändert.
Weitere Grenzen: Feature-Reproduktion ist numerisch statt byteidentisch;
Toleranz und Quellen sind dokumentiert. Andere Plattformen und die historische
Extraktionsumgebung sind nicht vollständig abgesichert. Rohdaten und vollständige
lokale Artefakte müssen separat aufbewahrt werden; sie sind nicht Teil dieses Tags.
