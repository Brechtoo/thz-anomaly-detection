# Überschneidung der 52 eigenen und 701 tsfresh-Features

**Sechs Features berechnen jeweils dieselbe Größe auf demselben vollständigen
Rohsignal.** Prüfung der lokalen Extraktionsfunktionen und der installierten
tsfresh-0.21.2-Calculatoren, zusätzlich numerischer Vergleich aller 53.192 IDs:

| Eigenes Feature | tsfresh-Feature | Definition |
| --- | --- | --- |
| y_max | y__maximum | Maximum |
| y_min | y__minimum | Minimum |
| y_mean | y__mean | arithmetischer Mittelwert |
| y_median | y__median | Median |
| y_std | y__standard_deviation | Populationsstandardabweichung (ddof=0) |
| y_energy | y__abs_energy | Summe der quadrierten Amplituden |

Eine Suche über alle 52 × 701 Spaltenpaare ergab keine weiteren numerischen
Übereinstimmungen: Vorauswahl auf 32 Zeilen, vollständige Überprüfung jedes
Kandidaten auf allen Zeilen, ID-basiert ausgerichtet. Toleranz rtol=1e-10,
atol=1e-12. Numerische Übereinstimmung allein gilt nicht als Formelbeweis.
Die sechs Formeln sind anhand der Implementierungen bestätigt und zusätzlich
mit deterministischen synthetischen Signalen getestet. Die maximalen absoluten
Abweichungen liegen für die fünf Lage-/Streuungsgrößen unter 1,8e-15; bei Energie
bei 1,46e-11 (Summe vs. Skalarprodukt und CSV-Rundung). Es geht um mathematisch
identische Größen, nicht um bitidentische Ergebnisse.

Ähnlich benannte Größen sind keine zusätzlichen Duplikate: Extremwertpositionen
verwenden bei uns die physische Signal-X-Achse, tsfresh normierte Indexpositionen.
Drittelenergien sind absolute Energien in drei Segmenten, tsfresh hier relative
Energien in zehn Segmenten. Die 35 Erweiterungsfeatures verwenden unter anderem
oberflächenausgerichtete, rauschkorrigierte Post-Peak-Fenster; tsfresh arbeitet
auf dem vollständigen Rohsignal. Ableitungen, Integrale und Peakstatistiken
haben dadurch andere Definitionen. Peak-to-Peak ist aus Maximum und Minimum
ableitbar, aber kein weiteres identisches Einzel-Feature. Auch Varianz/Standard-
abweichung oder RMS/Energie sind verwandt, jedoch keine identischen Größen.

Damit sind **6/52 ≈ 11,5 %** der eigenen Features bereits in den 701 enthalten.
Eine Kombination umfasst nach Entfernung dieser sechs direkten Duplikate
**747 statt 753 Spalten**. Das bedeutet nicht 747 statistisch unabhängige Größen.
Es wurden keine Feature-Spalten entfernt und keine Selektion oder Modelle verändert.
Die 701 sind die historisch selektierten Features aus ursprünglich 777 extrahierten.

Reproduktion:

```sh
.venv/bin/python -m src.analysis.feature_overlap \
  --baseline results/feature_tables/feature_table.csv \
  --tsfresh results/feature_tables/tsfresh_features.csv \
  --output /path/to/new_feature_overlap.json
```

Maschinenlesbarer Befund mit Eingangschecksummen und Abweichungen:
`data/metadata_reference_excluded/feature_overlap.json`.
