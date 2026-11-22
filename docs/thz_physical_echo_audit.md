# Echo-Audit und physikalisches Feature-Set V2

**Für spätere Trainings ist V2 die physikalisch besser begründete
Versuchsvariante.** Die Zweitmerkmale verwenden disjunkte Messfenster.
Die Auswertung misst weiterhin Ähnlichkeit zur Oberflächenpulsform.
Die Zuordnung einzelner Kandidaten zu realen Grenzflächen bleibt offen.
V1 ist als Referenz für einen gepaarten Vergleich verfügbar.

## Echo-Erkennung

V1 wählt den ersten hinreichend ausgeprägten Betragspeak im
aufgezeichneten Bereich 960–1030 als Frühanker. Der Referenzpuls
stammt aus derselben Messung (−2,5 bis +4,5, 141 Samples).
Das Suchfenster beginnt hinter der adaptiven Ringing-Sperrzone
(+6 bis +9) und endet bei +80. Die Suche nutzt den Betrag der lokal
mittelwertbereinigten normierten Kreuzkorrelation. Als Treffer gelten
innere Peaks ab 0,45 mit Prominenz ab 0,08, einer lokalen RMS-Amplitude
über 2 % der Referenz und einem Mindestabstand von zwei X-Einheiten.
Die Zentrierung und Normierung erlauben Vergleiche der Pulsform und
eine explizite Behandlung invertierter Treffer. Sie begrenzen den
Einfluss des absoluten Pegels. Der RMS-Grenzwert ist ein relativer
Amplitudenfilter, keine kalibrierte Rausch- oder Nachweisgrenze.

Die Fenstervorgaben begrenzen die tatsächlichen auswertbaren
Ausrichtungen stärker als das Post-Fenster allein. Die vollständige
Pulsstütze muss beim Verschieben in das Post-Fenster passen und
`find_peaks` schließt die beiden äußeren Positionen aus. Die Anker
zulässiger Treffer liegen je nach Sperrzone ungefähr bei
+8,55–11,55 bis +75,45. Damit können frühe, späte, schwache, verzerrte und
überlagerte Rückläufe unerkannt bleiben. Die Referenz selbst kann
Ringing oder zusätzliche Signalanteile enthalten. Bei 5 % der
stichprobenartig untersuchten Referenzfenster liegen mindestens 8,9 %
der global medianzentrierten Fensterenergie in den ersten oder letzten
zehn Samples. Dieser Befund beschreibt die Randbelegung des Fensters.
Eine eindeutige Zuordnung dieser Energie zu einer zusätzlichen
Reflexion ist daraus nicht möglich.

### Vollkohorte und Originalsignale

Von 52.220 Messungen haben 9.950 keinen primären Kandidaten, 23.959
genau einen und 18.311 mindestens zwei. Bei **7.554 der 18.311
V1-Zweitmerkmale (41,3 %)** teilen sich die beiden verwendeten
Fenster mindestens ein Sample. Die Überlappung beweist weder eine
Fehldetektion noch zwei getrennte Pulse.

| Gruppe | Messungen | Zweitmerkmal fehlt V1 | Zweitmerkmal fehlt V2 |
| --- | ---: | ---: | ---: |
| Klasse 0 | 7.125 | 50,9 % | 59,4 % |
| Klasse 1 | 7.976 | 58,8 % | 66,0 % |
| Klasse 2 | 3.716 | 72,6 % | 81,2 % |
| Klasse 3 | 33.403 | 68,6 % | 76,9 % |
| Messreihe 1 | 20.916 | 70,1 % | 79,0 % |
| Messreihe 2 | 21.044 | 67,1 % | 73,7 % |
| Messreihe 3 | 10.260 | 49,9 % | 60,1 % |

Die fehlenden V1-Zweitmerkmale reichen je Prüfkörper von **21,7 %**
(`messreihe_3_pk_07`) bis **90,9 %** (`messreihe_2_pk_03`).
Über Training, Validation und Test sind es 61,4 %, 66,8 % und 75,0 %.
Label 0 hat die niedrigste Fehlquote beim Zweitmerkmal. Messreihe 3
hat die niedrigste Fehlquote und den höchsten Anteil überlappender
Paare unter ihren Zweittreffern (3.039/5.140). Diese Zusammenhänge
zeigen, dass Detektierbarkeit auch Messbedingungen, Signalqualität
und Prüfkörper erfassen kann. Die Labels beschreiben Hohlraumklassen
und keine annotierten Echoereignisse.

Die Audit-Stichprobe umfasst 1.014 Originalsignale aus allen 35
Körpern. Je Kombination von Körper, Label und V1-Detektionsstatus
wurden bis zu drei deterministisch verteilte Messungen gewählt.
Rekonstruierte V1-Korrelationswerte, Zweitpeak-Verhältnisse und
relative Abstände stimmten mit den archivierten Werten überein.
Bei den 664 Stichproben mit fehlendem Zweitmerkmal liefern einzeln
veränderte Kriterien folgende Zahlen zusätzlicher Zweittreffer.

| Diagnostische Änderung | Zweittreffer in 664 Stichproben |
| --- | ---: |
| Mindestkorrelation 0,45 → 0,35 | 578 |
| Mindestprominenz 0,08 → 0,04 | 0 |
| RMS-Grenze 2 % → 1 % | 43 |
| Suchbeginn direkt nach Referenzfenster | 31 |
| Suchende +80 → +100 | 32 |

Die Stichprobe ist nach Detektionsstatus geschichtet. Die Zahlen sind
Sensitivitätsbefunde, keine Hochrechnung und keine nachgewiesenen
zusätzlichen Reflexionen. Die lockeren Kriterien kamen nur in der
Diagnostik zum Einsatz. Die sechs Originalsignal-Beispiele zeigen
Rohsignal, Referenz-/Suchfenster sowie die Korrelationskurve vor und
nach der RMS-Maske. Rot markiert sind die akzeptierten V1-Maxima.
[Originalsignal-Grafik](../results/analysis/physical_echo_review_v1/signal_examples.png) ·
[Beispiel-IDs](../results/analysis/physical_echo_review_v1/signal_examples.csv)

### Kontrollierte Puls-Einsätze

Für jede Konfiguration wurden dieselben 225 gemessenen
Oberflächenfenster aus den 23 Trainingskörpern auf flachem Hintergrund
eingesetzt. Ausgewählt wurde je vorhandener Kombination aus
Messreihe, Körper, Label und V1-Detektionsstatus ein Fenster
mit Seed 20261120. Gemessen wurde auch, ob alle Sollpositionen
innerhalb ±0,30 X getroffen wurden.

| Einsatz | Mehrere V1-Maxima | Disjunkter V2-Kandidat | Alle Sollpositionen V1 / V2 |
| --- | ---: | ---: | ---: |
| Ein exakter Puls | 62 | 0 | 225 / 225 |
| Ein Hann-gewichteter Puls | 16 | 0 | 225 / 225 |
| Zwei Pulse, Abstand 1,8 X | 52 | 0 | 0 / 0 |
| Zwei Pulse, Abstand 5 X | 225 | 43 | 225 / 0 |
| Zwei Pulse, Abstand 19 X, exakt | 225 | 225 | 225 / 225 |
| Zwei Pulse, Abstand 19 X, Hann-gewichtet | 225 | 225 | 225 / 225 |

Außerhalb der vollständigen Suchstütze (+6 oder +79) und bei einem
Einsatz mit 1 % Verstärkung lokalisierte V1 in diesen Kontrollen
keinen Sollpuls. Die Kontrollen zeigen zugleich, warum ein fehlendes
Merkmal keine Aussage über das Fehlen eines Echos ist und ein
vorhandener Peak keine Geometrie bestätigt. V2 unterdrückt in den
Einzelpulskontrollen Zweitmaxima, verliert aber auflösbare nahe
Pulse. Die 43 zusätzlichen disjunkten Treffer beim Einsatzabstand
5 X lagen außerhalb mindestens einer vorgegebenen Zielposition.
Die Kontrollen auf flachem Hintergrund liefern keine reale
Fehlalarm- oder Nachweisrate.

## Versionierter Export und Bewertung

V2 übernimmt 35 Merkmale bitgenau aus V1. Es ersetzt das
Zweitpeak-Verhältnis und den relativen Abstand durch die Werte des
höchstrangigen V1-Kandidaten mit einem vollständig disjunkten
Fenster zum Primärkandidaten. Dazu kommen der maximale
Korrelationsbetrag über alle RMS-zulässigen Positionen vor der
Peakselektion und der Anteil dieser Positionen. In dem Maximum
können Randpositionen eingehen. Es ist ein Maß für Ähnlichkeit
unter dem Amplitudenfilter. Der zulässige Anteil misst die
Verfügbarkeit ausreichend starker Fenster. Beide Größen sind
kontinuierlich und für die gesamte Kohorte endlich.

| Prüfung | V1 | V2 |
| --- | ---: | ---: |
| Messungen | 52.220 | 52.220 |
| Merkmale | 37 | 39 |
| Zweitmerkmale vorhanden | 18.311 | 14.016 |
| Zweitmerkmale fehlend | 33.909 (64,9 %) | 38.204 (73,2 %) |
| Zeilen mit allen Merkmalen | 18.269 | 13.976 |

Bei **3.259** überlappenden V1-Paaren fand V2 einen anderen,
disjunkten Kandidaten. Bei **4.295** entfiel das Zweitmerkmal.
Wenn die beiden ursprünglichen Fenster bereits disjunkt waren,
blieben Verhältnis und Abstand exakt gleich. Das neue Verhältnis
liegt über alle gültigen Zeilen zwischen 0,488 und 1, der Abstand
bei mindestens einer vollen Fensterlänge. Der maximale
Korrelationsbetrag liegt zwischen 0,226 und 0,980, der
RMS-zulässige Anteil zwischen 0,714 und 1. Alle 52.220
V1-Trefferdichten wurden beim V2-Lauf exakt reproduziert.
Alle Rohdateien wurden gegen das Archiv gehasht. IDs, Labels,
Gruppen, Splits, Folds sowie die 35 übernommenen Merkmale wurden
direkt verglichen. V1-Tabelle, Manifest, Diagnostik und Extraktor
stimmen mit den vorhandenen Prüfsummen überein.

Detektionsindikatoren wären eine mögliche spätere Ablation. Die
Primärinformation ist bereits durch Trefferdichte null bzw. die
NaN-Maske abbildbar. Für das Zweitmerkmal ist die NaN-Maske nach der
Imputation als expliziter Indikator verfügbar. Eine zusätzliche
statische Kopie wurde nicht exportiert. Beim späteren Training
sollte V2 mit ungefüllten Lücken für geeignete Modelle oder mit
foldintern gefitteter Imputation eingesetzt werden. Ein
Vergleichsarm ohne Zweitmerkmale und ein Arm mit Indikatoren
können zeigen, welcher Anteil des Signals aus der Detektierbarkeit
stammt. Dafür zuerst die eingefrorenen Trainingskörper-Folds
nutzen und kritische Fehler sowie Fehlalarme pro Körper und
Messreihe prüfen. V1 und V2 sollten wegen der 35 gemeinsamen
Merkmale als Vergleichsarme verwendet werden.

## X-Achse und Frequenzrechnung

**Gesichert:** Alle 52.220 Dateien haben 4.001 Werte von X=960 bis
X=1160, mit ΔX=0,05. Die V1-Gleichabstandsprüfung benötigte keine
Interpolation. Die Stichprobe aus 36 Rohdateien (zwölf pro
Messreihe) enthält zwei unmittelbar numerische Spalten ohne
Einheitenheader. Auch die Scan-ZIP-Inhalte und die benachbarten
Luftreferenzen liefern keine Exportbeschreibung. Die
`delta[...mm-...deg]`-Tokens gehören zum Messpunktnamen und
beschreiben die Poseabweichung. Der bestehende Loader benennt
die Spalten `x` und `y` und rechnet keine Einheit um. Die
Ankerung, Fenster und Ableitungen in der Verarbeitung arbeiten
auf der aufgezeichneten X-Skala. [Loader](../src/features/loading.py) ·
[V1-Spezifikation](specimen_tsfel_thz_features.md)

**Plausibel, unbelegt für diesen Export:** Die ältere lokale
Projektdatei `../thz-anomaly-project/notes` diskutiert
Echolaufzeiten in ps (Zeile 48) und FFT-Bänder in THz (Zeile 55).
Das Fraunhofer-HHI-Datenblatt zum TeraWave zeigt einen
Beispieltrace mit einer ps-Achse. Die Publikation zur
Isenhagen-Skulptur beschreibt dieses Gerätesystem und diskutiert
Echolaufzeiten in ps. Die Quellen stellen keinen Bezug zwischen
Gerät, Exporteinstellung und den drei Hohlraum-Prüfkörperscans
her. [HHI-Datenblatt](https://www.hhi.fraunhofer.de/fileadmin/PDF/PC/THZ/20181217_Terawave-Time-Domain-Spectrometer-web.pdf) ·
[Mach et al., 2026](https://doi.org/10.1038/s40494-026-02873-1)

Die FFT verwendet den gemessenen Schritt als `d` in `rfftfreq`.
Daraus folgen 20 Samples je X-Einheit und Nyquist bei 10 X⁻¹.
Für die 141-Punkt-Fenster beträgt das Binraster
1/(141 · 0,05) = **0,141844 X⁻¹**.
Die fünf verwendeten Bänder enthalten 2, 3, 4, 7 und 12 Bins.
Die Hann-Fensterwerte sind damit grobe Bandvergleiche. Das feinere
Volltrace-Raster von etwa 0,005 X⁻¹ gilt hier nicht. Bei einer
später bestätigten ps-Einheit entspräche dies 50 fs Abtastschritt,
20 THz Abtastrate und 0,142 THz FFT-Binabstand. Bis zur
Kalibrierbestätigung bleiben Angaben in X bzw. X⁻¹.
NCC, Energieverhältnisse und die normierten Abstände sind
dimensionslos. Die Auswahl der Fenster und Bänder hängt weiterhin
von der X-Skala ab. Aus diesen Merkmalen lassen sich ohne
Messgeometrie, Materialparameter und Laufzeitreferenz keine
sicheren Schichtdicken oder Material-Dämpfungskoeffizienten ableiten.
Für die Einheit braucht es die Exportspezifikation der konkreten
Session oder eine dokumentierte Referenzmessung mit bekannter
optischer Wegänderung.

## Reproduktion

V2 wurde in 89 Sekunden mit fünf Workern einschließlich
Rohdatenprüfung geschrieben. Die bereits vorliegenden numerischen
Audit-Tabellen wurden beim Abschluss wiederverwendet. Für
Grafiken wurde `Agg` gesetzt. Die Audit-CSV-Dateien, die
Grafik, der Manifest- und Prüfsummennachweis liegen lokal unter
`results/analysis/physical_echo_review_v1/`. Die beiden
Exportversionen liegen unter `results/feature_tables/`.
**201 Tests und Ruff bestanden.** Es wurde kein Modell trainiert.

```sh
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m src.features.thz_physical_v2 --workers 5 \
  --output results/feature_tables/specimen_thz_physical_v2_repeat.parquet
.venv/bin/python -m src.analysis.physical_echo_review \
  --output results/analysis/physical_echo_review_repeat \
  --v2 results/feature_tables/specimen_thz_physical_v2_repeat.parquet
```
