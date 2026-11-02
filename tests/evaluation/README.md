# Metriktests

Aus dem Projektverzeichnis:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q --tb=short
```

Die Metrikdefinition und Kennzeichnung historischer Ergebnisse sind in
[docs/error_metrics.md](../../docs/error_metrics.md) beschrieben.

Die Tests importieren nur das Metrikmodul. Sie verwenden synthetische Labels,
keine Modelle, Messdaten oder MLflow-Verbindungen. Erwartungswerte sind von Hand
aus Klassenübergängen und mathematischen Metrikdefinitionen berechnet.

Abgedeckt sind allgemeine Klassifikationsmetriken und Bericht, alle 16 Labelpaare,
die fünf Fehlergruppen mit Nennern und Gesamtquote, bestehende gelbe Übergänge,
perfekte Vorhersagen, fehlende Klassen, leere Fehlervektoren, ungültige Vektorformen
sowie Listen, NumPy-Arrays und pandas-Serien. Ein Test zeigt ausdrücklich, warum
`0 → 2` für Fehlerzählungen ungeeignet ist, und prüft unveränderte Eingaben.

Sechs zusätzliche Quelltextprüfungen kontrollieren die Argumente aller Aufrufstellen
mittels AST. Sie importieren die Einstiegsmodule nicht, da einige schon beim
Import Verzeichnisse erzeugen. Diese Prüfungen ergänzen die Verhaltenstests;
sie sind keine vollständigen Experimenttests.

Bei Einzelklassen sind sklearn-Warnungen zur Größe der Konfusionsmatrix zu erwarten.
Leere Eingaben werden nur für die Fehlerzählung als unterstützt vorausgesetzt.
Keine Tests werden mit `xfail` oder `skip` ausgeblendet.
