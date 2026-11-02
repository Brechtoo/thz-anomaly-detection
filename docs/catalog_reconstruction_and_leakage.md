# Datenpipeline und Leakage-Prüfung, 02.11.2026

## Ergebnis und wissenschaftliche Einordnung

**Die lokale Datenpipeline ist reproduzierbar:** Aus den vorhandenen Rohdateien
und unveränderten Originalannotationstabellen entstand der kombinierte Katalog
zweimal byteidentisch zur Referenz. Die Feature-Pipeline war zuvor numerisch
reproduziert worden; das gespeicherte HGB lieferte identische Validierungs-
Vorhersagen und Wahrscheinlichkeiten.

**Die vorhandenen Splits sind jedoch nicht gruppenunabhängig.** Alle drei
Messreihen/Scanbezeichnungen verteilen sich über Train, Validation und Test.
Damit besteht Gruppenüberlappung beziehungsweise ein erhebliches Leakage-Risiko
für Aussagen über unabhängige neue Scans, Prüfkörper oder Aufnahmesituationen.
Die reproduzierte Accuracy ist keine Bestätigung solcher Generalisierung.
Für das engere Ziel einer punktweisen Vorhersage innerhalb bereits bekannter
Scans ist dies eine andere, ausdrücklich zu benennende Evaluationsaufgabe.

Die bestehenden Splits, Modelle und Ergebnisse wurden nicht verändert. Es
wurden weder Training noch Optimierung oder neue MLflow-Runs ausgeführt.

## Rekonstruktion ab Rohmessungen und Annotationen

Die drei unveränderten Annotationseingänge sind:

| Messreihe | Scanverzeichnis | Annotation | Zeilen |
| --- | --- | --- | ---: |
| messreihe_1 | Hohlraumtestk1_tr/scan | hohltestk1_tr_markArea_XYList2.txt | 21.304 |
| messreihe_2 | Hohlraumtestk2_tr/scan | hohltestk2_tr_markArea_XYList.txt | 21.422 |
| messreihe_3 | Hohlraumtestk3_tr/scan | hohltestk3_tr_markArea_XYList.txt | 10.466 |

Alle 53.192 Messdateien lassen sich bijektiv über die historischen gerundeten
x/y-Koordinaten zuordnen. Es fehlen keine Annotationen oder Messungen; keine
Annotation wird mehrfach verwendet. Labels werden ausschließlich aus diesen
Tabellen übernommen, niemals erschlossen, ergänzt oder neu vergeben.
Die in den Rohdatenordnern zusätzlich liegenden Luftreferenzen/ZIP-Archive und
Arraydarstellungen der Labels sind nicht Teil der historischen Scan-Kataloge.
Es wurde derselbe Eingangsumfang wie in der Referenz rekonstruiert.

Technische Reparaturen:

- `label_data.py` bietet eine reine Aufbau-Funktion. Die bisherige Python-
  Rundungsregel und Labelzuordnung bleiben unverändert. Fehlende Labels,
  doppelte Annotationskoordinaten oder auf gleiche x/y-Koordinaten gerundete
  Mehrfachmessungen werden als Mehrdeutigkeit abgelehnt statt stillschweigend
  überschrieben. Die realen Daten enthalten keine solchen Fälle.
- `data_combining.py` liest/schreibt beim Import keine Daten mehr. Das fehlerhafte
  Verzeichnis-Ausgabeziel wurde durch den CSV-Pfad ersetzt. Quellreihenfolge,
  Zeilenreihenfolge und das ID-Format `messreihe_N_xX_yY` bleiben erhalten.
- `src.data.labeling.rebuild_catalog` verlangt einen neuen Ausgabeordner und
  speichert Original-Annotationshashes, Quellenhashes und Kataloghash. Bestehende
  Ziele werden abgelehnt. Auch die alten Schreibfunktionen überschreiben keine
  vorhandenen Dateien mehr.

Für eine spätere Rekonstruktion, mit neuem Ausgabeziel:

```sh
.venv/bin/python -m src.data.labeling.rebuild_catalog \
  --raw-root /path/to/labeled-data \
  --output-dir /path/to/new-catalog
```

Sortierung: Messreihe 1, 2, 3; innerhalb jeder Reihe aufsteigend x/y. Eine
zusätzliche deterministische Dateiaufzählung und die Ablehnung von Koordinaten-
Mehrdeutigkeiten beseitigen eine Abhängigkeit von der Dateisystem-Reihenfolge.
Tests prüfen dies auch mit umgekehrter Aufzählung der Eingabedateien.

### Vergleich mit der Referenz

Beide Neuaufbauten enthalten dieselben acht Spalten und alle 53.192 Zeilen.
IDs, Labels, Dateipfade, Dateinamen, x/y/z, Quellenzuordnung und Reihenfolge
stimmen exakt überein. Referenz und beide Ausgaben haben denselben SHA256:

```text
507534e5045533be772c9b263ad8d0d4791fd9e6c4a0b9ce1dc9c0b13cd55c48
```

Die drei Annotationsdateien blieben unverändert; ihre SHA256-Werte sind in
`reconstruction.json` und den neuen Katalogmanifesten gespeichert. Zusätzlich
wurden alle 53.192 Rohdateien erneut gehasht und mit dem vollständigen vorherigen
Feature-Neuaufbau verglichen: keine Veränderungen.

Grenzen: Bytegleichheit der Kataloge gilt für den geprüften lokalen Pfadstand
und die eingefrorene Umgebung. Bei Verlagerung der Rohdaten ändern sich absolute
Dateipfade und damit CSV-Hashes, auch wenn IDs/Labels/Zuordnungen gleich bleiben.
Die Originaldaten sind lokal vorhanden, aber nicht als portables Archiv im
Repository enthalten. Die fachliche Richtigkeit der Originalannotation wurde
nicht neu beurteilt; es wurden ausdrücklich keine Labels neu vergeben.
Die Feature-Tabelle bleibt numerisch, nicht byteidentisch reproduzierbar.

## Bestehende Split-Zugehörigkeit

Die Prüfung verwendet ausschließlich die gespeicherten Split-IDs, keine neue
Zufallsaufteilung. Die Manifestdateien wurden auf unveränderte Hashes geprüft.

| Messreihe | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| messreihe_1 | 13.721 | 3.356 | 4.227 |
| messreihe_2 | 13.587 | 3.510 | 4.325 |
| messreihe_3 | 6.734 | 1.645 | 2.087 |
| Gesamt | 34.042 | 8.511 | 10.639 |

Weitere aus Dateinamen oder Koordinaten abgeleitete Gruppierungen:

| Gruppierung | Gruppen | Über mehrere Splits | In allen drei Splits |
| --- | ---: | ---: | ---: |
| source_dataset | 3 | 3 | 3 |
| Scanbezeichnung im Dateinamen | 3 | 3 | 3 |
| Messreihe + Aufnahmedatum | 6 | 6 | 6 |
| Messreihe + Orientierung + Mittelungszahl | 4 | 3 | 3 |
| Messreihe + durch Zählerrücksetzungen abgeleitete Session-Kandidaten | 3 | 3 | 3 |
| Messreihe + 10×10-Koordinatenblock | 681 | 680 | 672 |

Die räumlichen Blöcke sind eine diagnostische Einteilung, keine freigegebene
neue Split-Definition. Datum, Dateinamensbezeichnung und Zählerrücksetzungen
sind keine autoritativen Session-IDs. Die Einteilung stellt keine unbekannten
Kalibrierungs-, Instrumenten- oder Prüfkörper-Metadaten nachträglich her.
Orientierungs-/Mittelungsgruppen allein beweisen keine Abhängigkeit; die
Scan-/Quell-Überlappung ist dagegen direkt nachgewiesen.

### Räumliche Nähe

Nächster Trainingspunkt innerhalb derselben Messreihe, euklidisch in den
aufgezeichneten x/y-Koordinateneinheiten (keine zusätzliche physikalische Einheit
oder Korrelationslänge angenommen):

| Menge | Punkte | Trainingspunkt in Abstand ≤1 | Anteil | Median | Maximalabstand |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 8.511 | 8.307 | 97,6031 % | 1 | 2 |
| Test | 10.639 | 10.443 | 98,1577 % | 1 | 2 |

Es gibt keine exakt gleichen Koordinaten innerhalb einer Quelle in mehreren
Splits. Unmittelbar benachbarte Messpunkte können jedoch sehr ähnliche lokale
Materialstrukturen, Oberflächen, Aufnahmebedingungen und Messartefakte teilen.
Diese Nähe quantifiziert das Abhängigkeitsrisiko, nicht dessen exakten Einfluss
auf Accuracy oder eine gemessene Signal-Korrelationslänge.

Die Labelverteilung unterscheidet sich zudem stark zwischen Messreihen:

| Messreihe | Label 0 | Label 1 | Label 2 | Label 3 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.158 | 4.909 | 2.008 | 11.229 |
| 2 | 3.136 | 2.860 | 1.641 | 13.785 |
| 3 | 1.020 | 207 | 67 | 9.172 |

Ein Modell kann damit implizite Quell-/Aufnahmecharakteristika nutzen, auch
wenn die Quellen-ID nicht direkt als Feature übergeben wird. Ob und wie stark
das tatsächlich geschieht, ist ohne unabhängige Gruppenauswertung nicht
quantifiziert. Tausende benachbarte Punkte sind nicht gleichbedeutend mit
Tausenden unabhängigen experimentellen Einheiten.

## Doppelte Messungen und weitere Leakage-Quellen

Nachgewiesen beziehungsweise ausgeschlossen im untersuchten Bestand:

- Keine doppelten IDs oder Dateipfade, keine mehrfachen Quell-x/y-Koordinaten,
  keine wiederverwendeten Sequenznummern innerhalb einer Messreihe.
- Keine byteidentischen Rohdateien, auch nicht über Split-Grenzen: alle
  53.192 aktuellen Rohdatei-Hashes sind verschieden.
- Keine identischen vollständigen Vektoren der 52 frisch extrahierten Features.
  Die zugehörige Feature-Ausgabe, ihr Quellenstand und sämtliche Rohdatenhashes
  wurden vor Wiederverwendung geprüft. Exakt gleiche vom Loader gelesene x/y-
  Arrays müssten in dieser deterministischen Extraktion identische Features
  ergeben. Deshalb gab es keine Kandidaten für eine zusätzliche Vollsignal-
  Prüfung formatverschiedener, numerisch identischer Kopien.
- Dies ist **keine** erschöpfende Suche nach nahezu identischen, skalierten,
  zeitverschobenen oder nur im Amplitudenarray gleichen Signalen. Räumlich
  korrelierte Nachbarn bleiben trotz fehlender exakter Duplikate relevant.

Lesende Codeprüfung:

- Die HGB-Baseline übergibt genau 52 Feature-Spalten; Label, ID, Pfad,
  Quellenzuordnung und räumliche Metadaten werden ausgeschlossen.
- Median-Imputation und balancierte Sample-Gewichte werden auf dem äußeren
  Trainingssatz bestimmt. Die pro Messung berechneten Features verwenden keine
  Statistiken anderer Split-Mengen. Kein direkter Label- oder äußeren
  Testdaten-Zugriff im HGB-Training wurde gefunden.
- Die Pipeline-Schritte von LR/MLP und den geprüften unüberwachten Verfahren
  fitten Imputer/Scaler auf Trainingsdaten. Die Transformer-Skalierung verwendet
  ebenfalls Trainingsmittelwert und -standardabweichung. Es wurden keine
  entsprechenden Fit-auf-Gesamtdaten-Probleme im geprüften Ablauf gefunden.
- Die ursprünglichen Zufallssplits in anderen Experimenten sowie
  `StratifiedKFold` in `optimize_models.py` berücksichtigen keine Gruppen.
  Damit gilt das Gruppenrisiko auch für innere CV und Hyperparametervergleiche.
  HGB-interne Early-Stopping-Validierung ist ebenfalls nicht gruppenbewusst.
- `optimize_thresholds.py` verwendet Validation, nicht die Testmenge.
  `final_evaluation.py` wertet Testdaten aus, enthält aber keine automatische
  Schwellenwertsuche auf dieser Menge. Eine manuelle Auswahl nach Einsicht in
  historische Testergebnisse ist aus dem Code nicht auszuschließen.
- Unüberwachte Verfahren verwenden teilweise eine andere 80/20-Aufteilung.
  Deren genaue Mitgliedschaften wurden hier nicht separat rekonstruiert; die
  zeilenweise, gruppenunbewusste Aufteilung bleibt aber ein methodisches Risiko.

Keine dieser Feststellungen beweist universelle Leakage-Freiheit. Vollständige
Aufnahme-/Kalibrierungsprotokolle und die tatsächlichen unabhängigen Prüfkörper-
Identitäten fehlen als autoritative Gruppenmetadaten.

## Auswirkungen und geeignete Alternativen

Die historische Baseline bleibt als **reproduzierbare punktweise Within-Scan-
Baseline** erhalten. Ihre Kennzahlen dürfen nicht als unabhängig validierte
Leistung auf neuen Scans/Prüfkörpern bezeichnet werden. Mögliche Folgen sind
optimistische Modell-/Schwellenwertwahl und zu enge Unsicherheitsintervalle.
Die Größe dieser Verzerrung wurde nicht geschätzt; dafür wäre eine separat
freigegebene gruppenunabhängige Auswertung erforderlich.

Vor einer wissenschaftlichen Freigabe für neue Aufnahmen:

1. Ziel der Generalisierung festlegen und echte experimentelle Gruppenschlüssel
   dokumentieren, vorzugsweise unabhängige Prüfkörper und vollständige Scans.
2. Nach ausdrücklicher Freigabe einen separaten gruppenbasierten Evaluationsstand
   aufbauen, beispielsweise mit `GroupShuffleSplit` oder `StratifiedGroupKFold`.
   Gruppen müssen auch in innerer CV und Modellauswahl respektiert werden.
3. Bei nur drei Quellgruppen sind stabile Gruppen-Train/Validation/Test-Sätze
   und Schätzungen begrenzt. Ein 1/1/1-Gruppensplit entspricht nicht den bisherigen
   Punktanteilen; Stratifizierung kann die starken Quellenunterschiede nicht
   beliebig ausgleichen. Leave-one-group-out wäre eine mögliche ergänzende
   Sensitivitätsanalyse, ersetzt aber keine zusätzlichen unabhängigen Gruppen.
4. Für räumliche Interpolation gegebenenfalls vorher definierte räumliche Blöcke
   mit Pufferzonen prüfen; angrenzende Blöcke allein garantieren keine
   Unabhängigkeit. Blockgröße/Puffer nicht nach Testleistung auswählen.
5. Schwellenwerte nur auf der vorgesehenen Validierung wählen und den
   gruppenunabhängigen Test bis zum abschließenden Vergleich unangetastet lassen.

Diese Alternativen wurden nur dokumentiert. Kein bestehender Split wurde
angepasst und kein Modell neu trainiert.

## Artefakte, Tests und Commit-Entscheidung

Neue Ausgabe ausschließlich unter:

```text
results/reproducibility_checks/catalog_leakage_20261102T213059Z_b67d2f19/
```

Enthalten sind zwei neue Katalog-Rekonstruktionen mit Manifesten,
`reconstruction.json`, `leakage.json`, Gruppen-/Labelhäufigkeiten,
`spatial_train_proximity.csv`, leere Duplikatkandidatenlisten, Prüfskript und
Unverändertheitsnachweis. Originalannotation, gespeicherte Split-Manifeste und
137 geschützte historische Dateien einschließlich MLflow-Datenbank blieben
unverändert.

Alle Tests ausgeführt: **92 bestanden, 8 bekannte sklearn-Warnungen**.
Neue Tests prüfen reine Importierbarkeit, unveränderte Rundung/Zuordnung,
Enumeration-unabhängige Ordnung, Mehrdeutigkeitsfehler, überschreibsichere
Schreibpfade, deterministische Kataloge und korrekte Gruppen-/Näheanalysen.

**Commit der technischen Änderungen und transparenten Befunde: ja.** Es gibt
keinen Grund, die reproduzierbare historische Baseline zu überschreiben.
**Freigabe als leakage-freie, auf unabhängigen Aufnahmen validierte Baseline:
nein.** Dafür ist eine wissenschaftliche Entscheidung über Gruppierung und
separate Evaluation erforderlich. Die frühere technische Commit-Bereitschaft
ist um diese Einschränkung zu ergänzen. Große Audit-CSV-Dateien und lokale
Modell-/MLflow-Artefakte sollten nicht pauschal mitcommittet werden. Es wurde
in dieser Einzelprüfung kein Commit, Tag oder Push ausgeführt. Der anschließende
Sessionabschluss ist separat dokumentiert.
