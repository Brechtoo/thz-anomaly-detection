# TSFEL und physikalisch motivierte THz-Merkmale

Beide Tabellen enthalten die 52.220 referenzfreien Messungen aus
`specimen_holdout_v1` mit unveränderten IDs, Prüfkörpern, Labels, Splits und
Trainings-Folds. Jeder Merkmalswert hängt ausschließlich vom betreffenden
Signal ab. Auch der Referenzpuls stammt aus derselben Messung.

## Signalachse und Merkmalsgruppen

Die Extraktion prüft die Abstände der X-Achse pro Messung. Die Messungen
haben jeweils 4.001 Punkte bei ΔX = 0,05, entsprechend 20 Abtastungen pro
aufgezeichneter X-Einheit. Kein Signal musste umgerechnet oder interpoliert
werden. Da eine Kalibrierung der aufgezeichneten X-Einheit nicht belegt ist,
werden Frequenzen als inverse X-Einheiten interpretiert. Bei künftig leicht
unregelmäßigen Abständen interpoliert der Runner auf ein äquidistantes
Raster gleicher Länge und Ausdehnung. Er markiert nicht monotone Achsen und
Lücken über dem 1,5-Fachen des typischen Schritts als fehlerhaft.

**TSFEL** analysiert den medianzentrierten Gesamttrace in drei Domänen.
Statistische Merkmale beschreiben die Verteilung relativ zur Basislinie,
zeitliche Merkmale unter anderem Differenzen und die Autokorrelationsdauer.
Spektrale Merkmale erfassen Spektrum, gemittelte Spektrogramm-Bins und
CWT-Skalen. Die Berechnung nutzt die einzelnen TSFEL-Funktionen und erhält
numerisch undefinierte Werte als fehlend. Die Frequenzargumente verwenden
den gemessenen Kehrwert des X-Abstands. [TSFEL-Funktionsreferenz](https://tsfel.readthedocs.io/en/latest/descriptions/feature_list.html)

**THz-physikalisch** verwendet denselben frühen Oberflächenanker wie die
bisherigen manuellen Varianten. Das Oberflächenfenster reicht von −2,5 bis
+4,5 relativ zum Anker. Nach dem bisherigen adaptiven Ringing-Ausschluss
(+6 bis +9) reicht das Post-Fenster bis +80. Der Anker ist eine
Signalheuristik. Die relative Lage der Fenster kann sich bei einem falsch
erkannten Oberflächenimpuls verschieben.

| Gruppe | Anzahl | Inhalt |
| --- | ---: | --- |
| Fenster-FFT | 14 | Logarithmische Band-Amplitudenverhältnisse zwischen Oberfläche und stärkstem Post-Fenster sowie zwischen frühem und spätem Post-Bereich. Dazu Abstände der Bandverteilungen und relative zeitliche Verteilung über überlappende Hann-Fenster. |
| Stationäre Haar-Details | 10 | Relative Detailenergien auf fünf Skalen, Entropiedifferenz zur Oberfläche, zeitliche Überlappung und Konzentration feiner und grober Details. Gültige Filterzentren ohne Randauffüllung. |
| Echoform | 6 | Hüllkurven-Crest-Faktor im Post-Fenster und relative Halbwertsbreite, Asymmetrie, bipolare Spanne, Polaritätsbalance und Krümmung am plausibelsten Echo. |
| Kreuzkorrelation | 7 | Normierte Suche nach der Oberflächen-Pulsform derselben Messung. Trefferdichte, Vorzeichen, relative Breite, Konkurrenz und Abstand eines zweiten Treffers sowie Verteilung und Rauigkeit des Residuals nach Pulsanpassung. |

Die FFT-Bandgrenzen liegen bei 0,125 / 0,4 / 0,8 / 1,4 / 2,4 / 4,0 pro
X-Einheit. Relative spektrale Pegel sind Beobachtungsmerkmale und erlauben
ohne Referenzkalibrierung keine materialbezogenen Dämpfungskoeffizienten.
Die Pulssuche verlangt einen Korrelationsbetrag von mindestens 0,45,
Prominenz 0,08 und eine lokale RMS-Amplitude von mehr als 2 % des
Oberflächenfensters. Zwei Treffer liegen mindestens zwei X-Einheiten
auseinander. Die Schwellen sind feste, signalbezogene Verarbeitungsschritte.
Eine Phase oder Gruppenlaufzeit aus unaufgelösten Echos wäre mehrdeutig und
wurde ausgelassen. Die Haar-Berechnung nutzt NumPy, ohne zusätzliche
Wavelet-Abhängigkeit für das physikalische Set.

## Redundanz und numerische Behandlung

Die manuelle Liste mit 52 Merkmalen und die archivierte tsfresh-Liste
lieferten den Vergleich für absolute Amplituden, Energien,
Peak-/Delay-Merkmale, globale Statistiken und FFT-/CWT-Koeffizienten.
Die physikalischen Merkmale verwenden stattdessen konditionale
Echoform-Vergleiche, relative Fenster- und Bandverteilungen und
mehrskalige Lokalisation. Die bisherige Einzelpeak-Korrelation
`second_peak_surface_corr` bleibt eigenständig. Die neue normierte
Korrelation sucht mehrere plausible Treffer über das Post-Fenster.

Bei TSFEL wurden unter anderem Rohpegel, absolute Energie, Standardabweichung
und bekannte Differenzstatistiken ausgelassen. Die Vollprüfung der
Kandidaten bestätigte zehn konstante ECDF-Rangspalten, zwei exakte
Duplikate zu tsfresh-Peakzahlen und ein TSFEL-internes Duplikat zwischen
Maximalfrequenz und Roll-off. Zusätzlich wurden mittlere Differenz,
Schiefe und Kurtosis wegen der äquivalenten bestehenden Berechnungen
ausgeschlossen. Audiospezifische und periodizitätsbezogene Einstellungen
wie menschliches Hörband, MFCC, LPCC und Grundfrequenz bleiben deaktiviert.

Die physikalische Extraktion fand für 9.950 Messungen keinen plausiblen
primären Pulsform-Treffer. Bei 33.909 Messungen fehlen Merkmale, die einen
zweiten Treffer benötigen. Drei weitere Halbwertsbreiten bzw.
Asymmetrien waren nicht bestimmbar. Ein schwach belegtes oberes
Frequenzband führte in 51 Zeilen zu einem fehlenden Oberflächen-Echo-Verhältnis.
Alle IDs bleiben erhalten. Die Rohlücke wird als Parquet-Nullwert abgelegt,
und der Diagnostik-Sidecar gibt die Anzahl fehlender Merkmale pro Zeile an.

## Reproduktion und spätere Integration

Aus dem Projektstamm mit der existierenden Python-3.12-Umgebung ausführen.
Die neue optionale Bibliothek ist in `requirements-tsfel.txt` fixiert.
Die archivierten Originalwellenformen und deren prüfsummengesicherte
Zuordnung aus `specimen_holdout_v1/features` werden benötigt.

```sh
.venv/bin/python -m pip install -r requirements-tsfel.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42
.venv/bin/python -m src.features.extract_specimen_features tsfel --workers 5 \
  --output results/feature_tables/specimen_tsfel_repeat.parquet
.venv/bin/python -m src.features.extract_specimen_features physical --workers 3 \
  --output results/feature_tables/specimen_thz_physical_repeat.parquet
```

Für Kombinationen die Dateien eins zu eins über `instance_id` verbinden
und die Übereinstimmung der Labels und Metadaten prüfen. Als Modellinputs
dienen ausschließlich die im jeweiligen JSON als `retained_features`
aufgeführten Spalten. Imputation, Varianzprüfung und Skalierung müssen
bei späteren Trainings innerhalb des jeweiligen Fold-Trainings erfolgen.
Die fehlenden Echoformen tragen die Information „kein hinreichender
Treffer“. Eine Imputation mit foldintern berechneten Werten und einem
optionalen Fehlindikator lässt sich separat beurteilen.

## Vollauswertung und Prüfung

| Set | Merkmale | Vollständige Zeilen | Laufzeit |
| --- | --- | ---: | ---: |
| `specimen_tsfel_v1` | **72** (7 statistisch, 8 zeitlich, 57 spektral) | 52.220 | 6 min 45 s bei 5 Workern |
| `specimen_thz_physical_v1` | **37** (14 FFT, 10 Haar, 6 Echoform, 7 Kreuzkorrelation) | 18.269 | 2 min bei 3 Workern |

Die Laufzeiten umfassen Einlesen, Prüfung der Rohdaten-Hashes, Extraktion
und Speichern. Beim TSFEL-Pilotlauf auf 64 Trainingsmessungen war die
spektrale Domäne mit rund 20,5 ms pro Signal (Median bei paralleler Systemlast)
der aufwendigste Teil. Statistik und Zeitdomäne lagen bei rund 1,2 bzw.
0,7 ms. Die physikalische Extraktion lag in einem separaten Pilotlauf bei
rund 3,0 ms ohne Einlesen. Die Pilotzeiten dienen der Einordnung des
Rechenaufwands. Die Vorhersagegüte wurde in diesem Schritt nicht gemessen.

In beiden Sets gab es 0 strukturelle Extraktionsfehler und 0 Zeilen ohne
Merkmalswerte. In den endgültigen Tabellen blieben keine leeren oder
exakt konstanten Spalten. Der Vollabgleich fand auch keine exakten
Duplikate innerhalb der Sets oder gegenüber frühen manuellen und
tsfresh-Merkmalen. Pro Set wurden 48 Originalsignale erneut extrahiert
und numerisch gegen die gespeicherten Zeilen geprüft. Außerdem wurden
Dateihashes, der vollständige ID-/Split-/Fold-Abgleich und zulässige
Wertebereiche für physikalische Verteilungs- und Korrelationsmerkmale
geprüft. **198 Tests und Ruff bestanden.**

Die beiden Parquet-Dateien liegen unter `results/feature_tables/` neben
ihren gleichnamigen JSON-Manifesten und `.diagnostics.csv`-Dateien.
