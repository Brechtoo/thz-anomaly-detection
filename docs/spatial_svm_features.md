# Räumliche Eingabefeatures für die RBF-SVM

## Fragestellung und Vergleich

Die Untersuchung verwendet die 34.235 Messpunkte der 23 eingefrorenen
Trainingsprüfkörper. Die drei äußeren Prüffolds umfassen 8/8/7 Prüfkörper
und 11.975/11.841/10.419 Messpunkte. Innerhalb jedes äußeren Trainingsblocks
werden die bereits gespeicherten drei inneren Gruppen-Folds verwendet.
Validation und Test bleiben geschlossen.

Für Manuell, Manuell + TSFEL und Manuell + TSFEL + Physik V2 vergleichen
wir jeweils dieselbe RBF-SVM mit und ohne räumliche Kanäle, auf exakt
denselben Messpunkten. Die im
[Nested-SVM-Lauf](svm_nested_hpo.md) regulär gewählten Werte für C,
Gamma und Klassengewichte stammen aus den jeweiligen inneren Folds.
Diese Parameter werden für die kontrollierte Feature-Ablation eingefroren.
Native innere und äußere Entscheidungen der dazu passenden historischen
Fits sind die Punktkontrolle. Der Ansatz isoliert die Wirkung der
Eingaberepräsentation bei gegebener SVM-Konfiguration.

## Feature-Ablation und Schutz vor Leakage

| Arm | Eingaben und Entscheidung |
| --- | --- |
| `point_only` | Unverändertes jeweiliges Feature-Set, native Vierklassenentscheidung |
| `tight` | Punktfeatures plus räumliche Kanäle im Radius von 1,5 Messabständen |
| `wide` | Dieselben räumlichen Kanäle im Radius von vier Messabständen |
| `selected` | Wahl zwischen Punktkontrolle und beiden Radien anhand der inneren OOF-Entscheidungen |
| `prior_rescue` | Früheres [räumliches Postprocessing](spatial_postprocessing.md) mit denselben nativen SVM-Ausgangsentscheidungen |
| `axis_tight` | Manuell + TSFEL, im engen Radius nur der nächste gemessene Punkt je X-/Y-Richtung |

Acht vorab festgelegte Kanäle stammen aus dem vorhandenen manuellen
Signalblock. Sie beschreiben Spitze-zu-Spitze-Amplitude, Streuung im frühen
Fenster, Amplitude, Abstand, SNR, Oberflächen-Amplitudenverhältnis und
Oberflächenkorrelation des zweiten Peaks sowie die Streuung nach dem
zweiten Peak. Je Kanal kommen die gewichtete lokale Standardabweichung
und die signierte Differenz Zentralwert minus Nachbarmittel hinzu. Damit
entstehen pro Variante 16 Zusatzspalten. Alle räumlichen Arme verwenden
dieselben Kanäle und Statistiken.

Die Distanz wird in den gemessenen X/Y-Koordinaten berechnet. Pro
Prüfkörper und Messreihe skaliert der Median des nächsten Punktabstands
den Radius. In der verwendeten Kohorte beträgt dieser Abstand eine
Koordinateneinheit. Die Gewichte folgen innerhalb des Radius einem
Gauß-Kern mit Sigma gleich halbem Radius. Der Achsenarm nutzt den engen
Radius und denselben Kern. Er berücksichtigt je Richtung höchstens
einen tatsächlich gemessenen Punkt mit demselben X- oder Y-Wert.
In der Kohorte entspricht das dem Ausschluss der diagonalen Nachbarn.
Je Prüfkörper beträgt der Median vier Nachbarn, kein Messpunkt ist im
Achsenarm isoliert.
Ein Punkt zählt nicht als eigener Nachbar, Referenzflächen tragen keine
Nachbarn bei. An Rändern normalisiert die Berechnung über die
vorhandenen Nachbarn. Fehlende Werte werden erst in der fold-lokalen
Vorverarbeitung behandelt. Die Berechnung setzt benachbarte Punkte
eines vollständig aufgenommenen Körpers zur Inferenz voraus.
Koordinaten, Körper-ID, wahre Labels und Wahrscheinlichkeiten früherer
Modelle sind keine Prädiktoren. Dadurch ist keine zusätzliche
Kalibration oder gestapelte OOF-Pipeline erforderlich.

Der historische Punktblock wird pro Fit wie zuvor imputiert, auf
Konstanten und exakte Duplikate geprüft und standardisiert. Für den
Zusatzblock werden Mediane, Konstantenmaske und Skalierung ausschließlich
aus den Fit-Körpern gelernt. Beide standardisierten Blöcke werden
zusammengefügt und mit `sqrt(d_punkt / d_erweitert)` multipliziert.
Damit bleibt die erwartete quadrierte Distanz zwischen unabhängigen
standardisierten Zeilen bei unverändertem Gamma auf der Skala der
Punktkontrolle.

Die adaptive Wahl benutzt die bestehende Nested-SVM-Auswahlfunktion
für native Entscheidungen. Innere Mindestwerte und relative
Qualitätsschranken verhindern die Wahl eines Arms allein aufgrund
weniger Freigaben. Danach priorisiert die Auswahl das Ziel unter 2 %,
den bestehenden Safety-Score aus kritischer und 1→3-Rate sowie
Prüfkörper-/Fold-Komponenten, dann Vierklassenqualität und Fehlalarme.
Da die inneren Labels bereits an der Wahl der SVM-Parameter beteiligt
waren, dient die innere Auswertung der Auswahl. Die Bewertung
der kombinierten Wahl erfolgt auf dem jeweiligen äußeren Prüffold.
Die frühere Nachbearbeitung wird erst danach zum Bericht hinzugefügt.

## Eigenständige Parametersuche

Für Manuell + TSFEL stand die vollständige innere Radius-Ablation bereits
beim Wiederanlauf zur Verfügung. Für dieses Feature-Set prüft ein
separater Lauf, welchen Zusatznutzen eine eigenständige SVM-Abstimmung
liefert. Je äußerem Fold wird aus den beiden räumlichen Radien anhand
der verfügbaren inneren OOF-Entscheidungen und der vorhandenen
Qualitätsschranken ein Radius gewählt. Danach prüft die Suche für diesen
Radius alle 32 Konfigurationen des historischen SVM-Gitters für C,
Gamma und Klassengewichte auf denselben drei inneren Gruppen-Folds.
Die bereits berechnete räumliche Konfiguration mit den Punktparametern
wird wiederverwendet. Der äußere Fold bewertet die so gewählte räumliche
Konfiguration. Ein zweiter Arm lässt anhand derselben inneren Kriterien
auch die Punktkontrolle als Rückfalloption zu.

Der direkte Vergleich mit demselben Radius und eingefrorenen Parametern
misst den Effekt der neuen Abstimmung. Der Vergleich mit der Punktkontrolle
misst den gemeinsamen Effekt von Nachbarschaftsfeatures und Abstimmung.
Eine zusätzliche Punktkontrolle spielt das historische Punktgitter unter
denselben relativen Qualitätsschranken erneut durch. Sie verwendet die
288 archivierten inneren Fits. Ihre Wahl weicht nur in Fold 0 von der
historischen Wahl ab. Für diesen Fold wird ein zusätzlicher äußerer
Punktfit gerechnet.
Die Aussage über erreichbaren Zusatznutzen gilt für dieses Feature-Set,
das vorgegebene Gitter und das innere Screening. Sie deckt keine gemeinsame
Vollsuche über beide Radien, andere Feature-Sets oder neue Messreihen ab.

## Äußere Gruppen-CV

Die kritische Rate bezieht sich auf 7.600 Punkte wahrer Klasse 1 oder 2.
Die Teilraten verwenden 5.297 bzw. 2.303 Punkte, die Fehlalarmrate
22.082 Punkte wahrer Klasse 3. Freigabe ist der Anteil aller Vorhersagen
mit Klasse 3. Der gepaarte Bootstrap zieht 2.000-mal mit Seed 42
ganze Prüfkörper und erhält die Paarung der Entscheidungen. Seine
Intervalle beschreiben die Streuung bei festgehaltenen Fits.

| Feature-Set | Arm | 1→3 | 2→3 | Kritisch | Fehlalarm % | Freigabe % | Macro-F1 | Bal. Acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Manuell | Punkt | 397 | 242 | 639 | 20,79 | 49,47 | 0,6788 | 0,7580 |
| Manuell | Eng | 398 | 322 | 720 | 19,82 | 50,79 | 0,7096 | 0,7851 |
| Manuell | Weit | 366 | 297 | 663 | 19,29 | 51,17 | 0,7161 | 0,7927 |
| Manuell | Innere Wahl | 383 | 244 | 627 | 20,04 | 49,81 | 0,6904 | 0,7711 |
| Manuell | Postprocessing | 393 | 242 | 635 | 20,81 | 49,44 | 0,6787 | 0,7581 |
| Manuell + TSFEL | Punkt | 478 | 268 | 746 | 18,41 | 51,51 | 0,7003 | 0,7730 |
| Manuell + TSFEL | Eng | 449 | 349 | 798 | 18,10 | 52,31 | 0,7270 | 0,7980 |
| Manuell + TSFEL | Weit | 418 | 338 | 756 | 17,22 | 52,81 | 0,7310 | 0,8010 |
| Manuell + TSFEL | Innere Wahl | 417 | 273 | 690 | 17,82 | 52,29 | 0,7221 | 0,7944 |
| Manuell + TSFEL | Postprocessing | 424 | 267 | 691 | 18,49 | 51,30 | 0,7017 | 0,7754 |
| Manuell + TSFEL | Achsenarm | 472 | 339 | 811 | 18,06 | 52,26 | 0,7214 | 0,7922 |
| Manuell + TSFEL + Physik V2 | Punkt | 310 | 321 | 631 | 15,78 | 54,90 | 0,7666 | 0,8325 |
| Manuell + TSFEL + Physik V2 | Eng | 313 | 349 | 662 | 14,49 | 55,77 | 0,7786 | 0,8399 |
| Manuell + TSFEL + Physik V2 | Weit | 364 | 366 | 730 | 13,12 | 56,67 | 0,7800 | 0,8372 |
| Manuell + TSFEL + Physik V2 | Innere Wahl | 378 | 330 | 708 | 14,69 | 55,84 | 0,7708 | 0,8311 |
| Manuell + TSFEL + Physik V2 | Postprocessing | 254 | 311 | 565 | 16,30 | 54,37 | 0,7655 | 0,8343 |

### Fehler nach Prüfkörper und Region

Für Manuell ergibt die innere Wahl in den äußeren Folds 0/1/2 die
Folge Punkt/weit/Punkt. Die kritische Rate fällt gegenüber der
Punktkontrolle um 0,16 Prozentpunkte (deskriptives 95-%-Prüfkörperintervall
−0,50 bis +0,07). Der Rückgang entsteht vollständig im mittleren Fold.
`messreihe_1_pk_09` trägt zehn der zwölf eingesparten Fehler.
Drei Körper verbessern sich, einer verschlechtert sich und 18 mit
Klasse-1/2-Annotationen bleiben gleich.

Für Manuell + TSFEL ergibt die innere Wahl weit/weit/Punkt.
Das gepaarte deskriptive Prüfkörperintervall der kritischen Rate
beträgt gegenüber der Punktkontrolle −1,93 bis +0,13 Prozentpunkte.
Gegenüber der Nachbarschaftsrettung reicht es von −0,32 bis +0,36.
Der Nettorückgang gegenüber der Punktkontrolle umfasst 127 verhinderte
und 71 neue kritische Fehler. Sieben der 22 Prüfkörper mit Klasse 1/2
verbessern sich, drei verschlechtern sich und zwölf bleiben gleich.
`messreihe_1_pk_04` und `messreihe_1_pk_09` tragen gemeinsam den
gesamten Nettorückgang. Ohne beide Körper beträgt die Änderung
null unter den übrigen 6.402 Klasse-1/2-Punkten. Bei
`messreihe_3_pk_06` bleiben 108 von 132 kritischen Punkten.

Die adaptive Radiuswahl ändert die Zahl kritischer Fehler weder bei
isolierten Klasse-1-Punkten (12 von 31) noch bei Komponenten
mit zwei bis vier Punkten (16 von 41). Bei größeren Klasse-1-Komponenten sinken die
kritischen Fehler von 450 auf 389. Bei Klasse-1/2-Punkten in räumlicher
Nachbarschaft zu Klasse 3 steigen sie von 479 auf 487. Abseits solcher
Übergänge sinken sie von 267 auf 203. Die Auswertung beschreibt
Annotationszusammenhänge im Messraster.

Bei Manuell + TSFEL liefert der Achsenarm 811 kritische Fehler
(472 Fälle 1→3, 339 Fälle 2→3). Der gleich große Kreis mit diagonalen
Nachbarn liefert 798 (449 und 349). Die gepaarte Differenz beträgt
+0,17 Prozentpunkte (deskriptives 95-%-Prüfkörperintervall
−0,27 bis +0,65). Macro-F1 sinkt von 0,7270 auf 0,7214 und Balanced
Accuracy von 0,7980 auf 0,7922. Der Achsenarm liegt bei beiden
Vierklassenkennzahlen über der Punktkontrolle. Seine kritische Rate
steigt gegenüber der Punktkontrolle von 9,82 % auf 10,67 %. Im dritten
Fold entfallen 206 statt 140 Fehler auf 2→3. Die Begrenzung auf
Achsenpunkte zeigt unter diesen Bedingungen keinen Sicherheitsgewinn.

Für Manuell + TSFEL + Physik V2 wählt die innere Suche eng/weit/Punkt.
Die kritischen Fehler steigen gegenüber der Punktkontrolle von 631
auf 708, obwohl die Fehlalarmrate von 15,78 % auf 14,69 % sinkt.
Die Nachbarschaftsrettung erreicht 565 kritische Fehler. Gegenüber
dieser Referenz liegt die adaptive Radiuswahl um +1,88 Prozentpunkte
höher (95-%-Prüfkörperintervall +0,01 bis +5,11).
`messreihe_1_pk_09` trägt 65 der 77 zusätzlichen Fehler gegenüber
der Punktkontrolle. Bei `messreihe_3_pk_06` bleiben 117 von 132
und bei `messreihe_2_pk_08` 11 von 11 kritischen Fehlern bestehen.

Die festen Kreisradien liefern in allen drei Feature-Sets mehr
2→3-Fehler und höhere Macro-F1-Werte als die jeweilige Punktkontrolle.
Damit ändern sie die Art der Fehler und den Betriebspunkt. Die
Freigabequoten unterscheiden sich zwischen den Armen. Die Vergleiche
beantworten die Frage nach der Güte der jeweiligen nativen
Vierklassenentscheidung bei ihrer eigenen Freigabequote.

### Äußeres Ergebnis der Nested-Parametersuche

Alle zusätzlichen Arme verwenden Manuell + TSFEL. Die direkte
HPO-Kontrolle verwendet in beiden Zeilen die Radien weit/weit/eng.
Bei der adaptiven HPO-Wahl steht zusätzlich die Punktkontrolle zur
Verfügung. Sie ergibt wie die ursprüngliche Radiuswahl weit/weit/Punkt.

| Arm | 1→3 | 2→3 | Kritisch | Kritisch % | Fehlalarm % | Freigabe % | Macro-F1 | Bal. Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gleiche Radien, Punktparameter | 414 | 342 | 756 | 9,95 | 17,65 | 52,45 | 0,7296 | 0,8011 |
| Gleiche Radien, eigene HPO | 450 | 332 | 782 | 10,29 | 19,34 | 51,53 | 0,7119 | 0,7836 |
| Adaptive HPO-Wahl mit Punktoption | 453 | 263 | 716 | 9,42 | 19,50 | 51,37 | 0,7043 | 0,7769 |
| Punkt-HPO mit gleichen Qualitätsschranken | 504 | 261 | 765 | 10,07 | 20,76 | 49,94 | 0,6753 | 0,7497 |

Alle räumlich gewählten Kandidaten verwenden balancierte
Klassengewichte. In Fold 0 bleibt C bei 0,3. Die Suche senkt Gamma von
0,0003 auf 0,0001, den niedrigsten getesteten Wert. Damit fällt die
kritische Rate auf den inneren Vorhersagen von 16,21 % auf 14,16 %.
Im zugehörigen äußeren Fold steigt sie von 2,31 % auf 3,24 %.
Fold 1 bestätigt C = 0,3 und Gamma = 0,0003. Für den engen Radius in
Fold 2 bestätigt die Suche C = 1 und Gamma = 0,001. Bei der adaptiven
Wahl setzt sich in Fold 2 der Punktarm durch.

Bei gleichen Radien erhöht die eigene HPO die gepoolte kritische Rate
um 0,34 Prozentpunkte (95-%-Prüfkörperintervall −0,30 bis +1,37) und
senkt Macro-F1 um 0,0177 (−0,0400 bis −0,0033). Die 26 zusätzlichen
kritischen Fehler bestehen aus 36 zusätzlichen 1→3 und zehn entfallenen
2→3. `messreihe_1_pk_04` trägt netto 36 zusätzliche Fehler bei,
während vier andere Körper zusammen zehn kritische Fehler einsparen.
Auch gegenüber der adaptiven Radiuswahl mit eingefrorenen Parametern
verschlechtert die HPO-Wahl die Bilanz von 690 auf 716 kritische Fehler.
Die nach denselben Schranken optimierte Punktkontrolle erreicht 765
kritische Fehler gegenüber 746 bei der historischen Punktkontrolle.
Gegenüber dieser Punktkontrolle fällt die kritische Rate der adaptiven
HPO-Wahl um 0,64 Prozentpunkte (−1,74 bis +0,17). Gegenüber dem früheren
Postprocessing steigt sie um 0,33 Prozentpunkte (−0,39 bis +1,57).

### Schlussfolgerung

Das Beibehalten der Punktparameter ist für die kontrollierte Ablation
geeignet. Für Manuell + TSFEL bestätigt die eigenständige Suche die
übernommenen Parameter in zwei Folds und erzielt im dritten mit einer
anderen Wahl auf den äußeren Prüfkörpern schlechtere Werte. Das
untersuchte Gitter liefert damit keinen belegten Zusatzgewinn durch
Retuning. Die festen räumlichen Features verbessern die allgemeine
Vierklassengüte häufig. Der Rückgang kritischer Fehler bei adaptiver
Radiuswahl ist für Manuell und Manuell + TSFEL klein, auf wenige Körper
konzentriert und mit Intervallen vereinbar, die null einschließen.
Der niedrigste beobachtete Wert bleibt bei der Physik-V2-Punkt-SVM
mit Nachbarschaftsrettung bei 7,43 %. Das Ziel unter 2 % wird gepoolt
von keinem Arm erreicht.

Die äußeren Prüfkörper sind innerhalb ihrer Folds von Fit und Auswahl
getrennt. Die Trainingsmengen der drei äußeren Fits überlappen, und die
Folds enthalten Körper aus den vorhandenen Messreihen. Das Experiment
belegt die Leistung unter diesen Messbedingungen. Eine Übertragung auf
andere Messreihen oder eine betriebliche Freigabegrenze ist damit offen.

## Reproduktion

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m src.experiments.spatial_feature_cv \
  --workers 1 --output-dir results/experiments/specimen_spatial_features_repeat
```

Die beiden ergänzenden Läufe beziehen sich auf die Checkpoints der
Radius-Ablation. Der kombinierte Bericht liest die abgeschlossenen
Läufe und prüft die Übereinstimmung der zugehörigen MLflow-Artefakte.

```sh
.venv/bin/python -m src.experiments.spatial_feature_axis \
  --fixed results/experiments/specimen_spatial_features_v1

.venv/bin/python -m src.experiments.spatial_feature_hpo \
  --frozen results/experiments/specimen_spatial_features_v1 --jobs 2

.venv/bin/python -m results.experiments.specimen_spatial_features_hpo_v1.point_comparator.reproduce

.venv/bin/python -m src.analysis.spatial_feature_review \
  --fixed results/experiments/specimen_spatial_features_v1 \
  --axis results/experiments/specimen_spatial_features_axis_v1 \
  --hpo results/experiments/specimen_spatial_features_hpo_v1 \
  --point-comparator results/experiments/specimen_spatial_features_hpo_v1/point_comparator \
  --output-dir results/experiments/specimen_spatial_features_review_repeat
```

Für neue Wiederholungen erhält jeder Lauf einen eigenen Zielordner und
einen lokalen MLflow-Store. `--resume` prüft den eingefrorenen Plan und die Parquet-Hashes
abgeschlossener Fits. Die Eingabeprüfung verlangt die ursprüngliche
Train-Kohorte, Gruppen-Folds, SVM-Fit-Artefakte und den verifizierten
Postprocessing-Lauf. Seed 42 sowie die Hashes der Daten, Folds,
Archivvorhersagen und des Quellcodes stehen in den Laufplänen. Die Pläne
enthalten außerdem die verwendete Python- und Paketversion. Die dichten
RBF-Kernel belegen bei äußeren Fits zeitweise rund 6 GB auf dem lokalen
Laufwerk.

Beim Wiederanlauf lagen 31 Fit-Checkpoints im Hauptlauf und sechs im
parallelen Nebenlauf vor. Die ursprünglichen Aufrufe, die Übereinstimmung
der Eingabe- und Code-Hashes, die Vorhersagemetadaten, die Vorverarbeitung
und das aufgelöste Gamma wurden geprüft. Der Audit
`resume_audit_20261124.json` im Hauptlauf hält die 37 übernommenen
Prüfsummen fest. Die beiden unvollständigen Kernel-Arbeitsstände
enthielten keine abgeschlossenen Fit-Checkpoints.

Der [Radius-Hauptlauf](../results/experiments/specimen_spatial_features_v1/)
enthält 72 verifizierte räumliche Fit-Checkpoints und 616.230 gepaarte
Vorhersagezeilen. Seine MLflow-Run-ID ist
`e3cfbb9544f246a080d51f3efb0076d5`. In
`verified_20261124.log` ist die Prüfung der 117 Eingabe- und
Referenz-Hashes sowie der Fold- und Auswahlkonsistenz festgehalten.
Die MLflow-Run-IDs des Achsenarms und der zusätzlichen Punktkontrolle
sind `28d160a0e5ee46f4b4dae0c5bdf66611` und
`055b8f66a3524c4fa950efda9180cd37`.
Die [Nested-Suche](../results/experiments/specimen_spatial_features_hpo_v1/)
mit Run-ID `55bb8db5bb83488f9bdc36e92960f870` verwendet neun
vorhandene innere Fits und ergänzt 279. Dazu kommt ein neuer äußerer
räumlicher Fit. Alle 280 neuen Vorhersage-Checkpoints wurden anhand
ihrer Prüfsummen verifiziert. Das Gitter, die inneren Kennzahlen,
Entscheidungen und Checkpoint-Prüfsummen liegen im jeweiligen Lauf.

Der [kombinierte Bericht](../results/experiments/specimen_spatial_features_review_v1/)
enthält `pooled.csv`, `fold.csv` und `specimen.csv` mit den
klassenweisen Kennzahlen, `confusion.csv` mit den Konfusionsmatrizen,
`diagnostic.csv` und `paired_error_changes.csv` mit Fehleränderungen
nach Körper und räumlichem Ausschnitt sowie `paired_bootstrap.csv`
mit den gepaarten Intervallen. Die
[Fehlerkarte](../results/experiments/specimen_spatial_features_review_v1/specimen_error_maps.png)
zeigt kritische Fehlertypen und Fehlalarme für zwei zuvor identifizierte
schwierige Prüfkörper in Messkoordinaten. Das Manifest verbindet die
Prüfsummen aller neun Berichtdateien mit den vier lokalen MLflow-Läufen.
