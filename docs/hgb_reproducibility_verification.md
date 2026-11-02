# Abschlussprüfung der HGB-Reproduzierbarkeit, 02.11.2026

**Spätere Katalog-/Leakage-Prüfung:** Die Rekonstruktion ab Originalannotationen
wurde inzwischen byteidentisch bestätigt. Die vorhandenen Splits teilen jedoch
alle drei Messreihen zwischen Train/Validation/Test; Generalisierung auf
unabhängige neue Scans ist nicht belegt. Die folgenden historischen Prüfergebnisse
bleiben gültig. Aktuelle Einordnung im
[Katalog- und Leakage-Bericht](catalog_reconstruction_and_leakage.md).

**Nachtrag:** Der anschließende Vergleich mit dem gespeicherten HGB bestätigt
identische Validierungsentscheidungen und Wahrscheinlichkeiten. Der Stand wird
unter einer begründeten gemischten Toleranz als numerisch reproduzierbar
akzeptiert; die hier dokumentierte fehlende Bytegleichheit und die vier
Ausreißer des ursprünglichen Prüfkriteriums bleiben bestehen. Siehe
[numerische Abschlussbewertung](feature_numerical_reproducibility.md).

Die HGB-Validierung auf der unveränderten Referenz-Feature-Tabelle war zuvor
exakt reproduziert worden. Diese Prüfung untersucht zusätzlich eine frische
Installation und den Neuaufbau der Features ab kombiniertem Eingabekatalog.
Es wurde kein Modell trainiert und kein MLflow-Run angelegt oder verändert.

## Installation und Tests

Eine neue venv wurde mit Python 3.12.5 unter
einem separaten temporären Verzeichnis erzeugt, ohne System-Site-Packages und
ohne Pakete der bestehenden Projekt-`.venv` einzubinden. Die Installation mittels
`pip install -r requirements-lock.txt` war erfolgreich. Alle 101 installierten
Distributionen stimmen exakt mit dem Lock und der Referenzumgebung überein.
`pip check` meldet keine defekten Abhängigkeiten.

Die gesamte Testsuite wurde in dieser frischen Umgebung ausgeführt:
**74 bestanden, 8 bekannte sklearn-Einzelklassenwarnungen**, Dauer 19,54 Sekunden.
Die temporäre Umgebung wurde beim Sessionabschluss nach Sicherung ihrer
Metadaten entfernt. Installationslog, pip-Installationsbericht mit Download-Hashes
und Umgebungsmetadaten bleiben im lokalen Prüfungsordner erhalten.

## Sicherheitsprüfung vor der Extraktion

- 53.192 lokal vorhandene Messdateien; zusammen 11.066.701.976 Bytes.
- Größte Messdatei: 208.052 Bytes; Referenz-CSV: 63.737.550 Bytes.
- Rund 69 GB freier Speicher vor der vollständigen Prüfung; großzügiges
  Ausgabebudget 1 GiB und konservatives Speicherbudget 2 GiB.
- Probe mit 64 über den Katalog verteilten Messungen: etwa 0,31 Sekunden,
  daraus rund 261 Sekunden reine Extraktion geschätzt. Zusätzliche I/O-Zeit
  für Vor-/Nachher-Hashes und das Schreiben wurde berücksichtigt.
- Ausgabe ausschließlich in einem neuen Unterordner; vorhandene Ziele werden
  vom Rebuild-Werkzeug abgelehnt. Rohdaten werden nur gelesen.

Tatsächlich benötigten vollständige Extraktion und Eingabe-Hashprüfungen
**330,13 Sekunden**. Der Spitzen-RSS des Audit-Prozesses einschließlich
Tabellenvergleich betrug **458.244.096 Bytes** (rund 437 MiB).
Der Plattenplatz blieb ausreichend. Alle 53.192 Messungen wurden erfolgreich
verarbeitet; die Rohdaten-Hashes vor/nach der Verarbeitung stimmten überein.

## Vergleich der Feature-Tabellen

| Merkmal | Ergebnis |
| --- | --- |
| Zeilen | 53.192, identisch |
| Spalten | 58, davon 52 numerische Features, identisches Schema |
| Spalten-Datentypen | identisch |
| ID- und Zeilenreihenfolge | identisch |
| Metadaten einschließlich Original-Labels und Pfaden | identisch |
| NaN- und Endlichkeitsmasken | identisch |
| Exakte numerische Werte | nicht identisch |
| Vorab gewählte Toleranz `rtol=atol=1e-12` | 4 Zellen außerhalb |
| Bytehash | nicht identisch |

Von 2.765.984 numerischen Zellen unterscheiden sich 1.381.899 in der exakten
Gleitkommadarstellung. Die größte absolute Differenz ist
`2.7966962079517543e-11`. Alle Werte liegen innerhalb der **zusätzlich
diagnostisch** geprüften Toleranz `rtol=atol=1e-11`. Diese nachträgliche
Diagnose ersetzt ausdrücklich nicht das ursprüngliche strengere Prüfkriterium.

Die vier Ausreißer des ursprünglichen Kriteriums betreffen
`second_peak_min_width` (ein Fall) und `second_peak_max_width` (drei Fälle),
mit absoluten Differenzen von etwa `1.95e-12` bis `3.20e-12`. Der globale
Maximalfehler ist wegen des relativen Toleranzanteils nicht einer dieser vier
Ausreißer. Identitäten und Werte stehen in `deviation_analysis.json`.

Referenz-SHA256:

```text
03d439403489d319feb7299847b159d9bd96e9398f72fd1d19bd64cb84df97ed
```

Neu aufgebauter SHA256:

```text
a9ad6f850c1586761b97e08ecb02677fb2c84bc753f868b2b55832703e149987
```

Die neue Tabelle ersetzt die Referenz nicht. Der Hashschutz im HGB-Einstieg
lehnt sie weiterhin als abweichenden Datenstand ab. In dieser Etappe wurden
Modelltraining und Vorhersagen nicht untersucht; der anschließende
[Vorhersagevergleich](feature_numerical_reproducibility.md) bestätigt identische Inferenz.

## Untersuchung von trapz/trapezoid und Rundung

Während der vollständigen Extraktion wurden alle **106.384 Integrationsaufrufe**
mit der nachgebildeten früheren eindimensionalen Trapezformel verglichen:

```python
(np.diff(x) * (y[1:] + y[:-1]) / 2.0).sum()
```

Es gab **keine bitweisen Abweichungen** gegenüber `np.trapezoid` auf denselben
Arrays. Die tatsächlich für die Feature-Tabelle verwendeten Werte stammen
unverändert aus `np.trapezoid`. Das Audit hat keinen anderen Rechenweg in die
Ausgabe übernommen und keine Produktionsfunktion verändert.

Damit erklärt der reine API-Namenswechsel die beobachteten Unterschiede nicht.
Abweichungen betreffen zudem nicht nur Flächen, sondern beispielsweise auch
Extremwerte, Streuungen, Peak-Breiten und Korrelationen. Sie passen zu sehr kleinen
Rundungsunterschieden beim historischen Einlesen, Zwischenserialisieren oder
Rechnen mit damals verwendeten Bibliotheksversionen. Eine eindeutige Ursache
ist mangels vollständiger Provenienz der damaligen Feature-Erzeugung nicht
belegbar; das heutige Training-Lock belegt nicht die damalige Extraktionsumgebung.

Die Git-Historie zeigt einen älteren mehrstufigen Weg mit einer gespeicherten
Baseline-CSV als Eingang der Feature-Erweiterung. Im aktuellen Bestand stimmen
die 17 Baseline-Spalten dieser Zwischen-CSV mit der Referenz exakt überein.
Auch zwischen vorhandener Erweiterungs-CSV und finaler Tabelle bestehen jedoch
kleine numerische Differenzen. Diese Zwischenstände wurden nur gelesen.

Zusätzliches Einlesen beider Tabellen mit `float_precision='round_trip'`
änderte in pandas 3.1.0 keine Werte; der Parameter ist dort als veraltet markiert.
Dies erklärt die Abweichungen also nicht durch die gewählte Vergleichsoption,
rekonstruiert aber auch nicht das historische CSV-Parsing. Für die vier
Toleranzausreißer wurden die unverarbeiteten dezimalen CSV-Zeichenfolgen gesichert.

## Nachweise und verbleibende Grenzen

Alle neuen Prüfergebnisse liegen unter:

```text
results/reproducibility_checks/20261102T210010Z_4ec464c0/
```

Darin liegen `installation.log`, `install-report.json`, `fresh_environment.json`,
`pip-check.log`, `pytest.log`, Probe und Ressourcenabschätzung,
`feature_rebuild/` mit neuer Tabelle und Rohdatenmanifest, Vergleichsdateien
pro Feature, die vier Toleranzausreißer sowie der ausgeführte Audit-Quelltext.

Abgesichert ist die erneute Installation auf dieser Plattform sowie die
ID-basierte Split-Wiederherstellung und die bereits zuvor exakt bestätigte
HGB-Baseline auf der eingefrorenen Referenztabelle. Der Feature-Neuaufbau ist
vollständig ausführbar und sehr nahe an der Referenz, aber **nicht exakt
numerisch oder byteidentisch reproduziert**.

Offen bleiben plattformübergreifende/binary-identische Umgebungen und die
historische Feature-Extraktionsumgebung. Die nachfolgende Katalogprüfung hat
die Rekonstruktion ab Annotationen inzwischen bestätigt, zugleich aber
Gruppenüberlappung der Splits nachgewiesen. Der aktuelle Geltungsbereich und
die Veröffentlichungspolitik stehen im [Betriebsleitfaden](hgb_reproducibility.md).
Historische Modelle, Ergebnisdateien und MLflow-Runs bleiben unangetastet.
