# Verschachtelte Gruppen-CV für RBF-SVM und Freigaberegel

Ziel der inneren Suche ist, die RBF-SVM mit ihrer regulären
Vierklassenentscheidung und mit einer sicherheitsorientierten
Freigaberegel unter denselben äußeren Prüfbedingungen zu vergleichen.
Das regulär ausgewählte Modell, der separat sicherheitsorientiert
ausgewählte Fit ohne Regel und dessen adaptierte Vierklassenentscheidung
werden jeweils separat ausgewiesen. Eine explizite Prüfung ist, falls
berechnet, eine eigene Entscheidung und wird mit ihrer Prüfquote
ausgewertet.

## Abschluss des Laufs vom 21. November 2026

Alle 864 inneren Fits und neun äußeren Auswertungen wurden in
2 Stunden und 1 Minute abgeschlossen. Ein gezielter Neustart
übernahm die bereits geschriebenen Fits. Es gibt keine
protokollierten Fitfehler. Das MLflow-Experiment
`thz-specimen-rbf-svm-nested-hpo` enthält die Suche und
Auswertung unter dem Parent `8f88f65c4cd74cab9c8df25b771fc832`.

| Feature-Set | Kritisch Grid / regulär / adaptiert | Fehlalarme regulär / adaptiert | Macro-F1 regulär / adaptiert | Balanced Accuracy regulär / adaptiert |
| --- | ---: | ---: | ---: | ---: |
| Manuell + TSFEL | 8,16 / 9,82 / 5,86 % | 18,41 / 29,94 % | 0,7003 / 0,6907 | 0,7730 / 0,7850 |
| Manuell + TSFEL + Physik V2 | 8,78 / 8,30 / 5,96 % | 15,78 / 33,64 % | 0,7666 / 0,7038 | 0,8325 / 0,7996 |
| Manuell | 8,59 / 8,41 / 7,93 % | 20,79 / 21,87 % | 0,6788 / 0,6814 | 0,7580 / 0,7631 |

1. **V2 regulär ist der erste Kandidat für eine spätere
   unabhängige Vierklassen-Auswertung.** Gegenüber seiner
   historischen Referenz sinken 1 → 3 von 382 auf 310,
   während 2 → 3 von 285 auf 321 steigen. Insgesamt sind es
   36 kritische Fehler weniger. Fehlalarme, Macro-F1 und Balanced
   Accuracy verbessern sich gleichzeitig. Die reguläre Auswahl
   verwendet in zwei Folds `C=10, gamma=0.001` und in einem Fold
   `C=1, gamma=0.0003`. `C=10` lag am oberen Rand der Suche.
2. **Manuell + TSFEL adaptiert ist der Sicherheitsvergleich.**
   445 statt 620 kritische Fehler ergeben die niedrigste gepoolte
   Rate. Die Fehlalarmrate steigt gegenüber der Grid-Referenz um
   12,90 Prozentpunkte. V2 adaptiert hat mit 250 statt 282
   Ereignissen die niedrigste Rate für 1 → 3, dafür mehr Fehler
   2 → 3 und mehr Fehlalarme als Manuell + TSFEL adaptiert.
   Die manuelle Variante bleibt Kontrollbaseline.
3. **Die Sicherheitsregeln erreichen das 2-%-Ziel gepoolt nicht.**
   Ihre Fold-Raten liegen für Manuell + TSFEL bei 1,22 / 9,77 /
   6,88 % und für V2 bei 0,83 / 8,67 / 9,23 %. Beide Varianten
   lassen auf `messreihe_3_pk_06` weiterhin 65 von 132 kritischen
   Messungen als Klasse 3 durch. Bei `messreihe_2_pk_08` sind
   es für Manuell + TSFEL 6 von 11 und für V2 11 von 11.
   Die deskriptiven, gepaarten Prüfkörper-Perzentile der
   Änderung zur Grid-Referenz betragen für die gepoolte kritische
   Rate −5,91 bis −0,47 Prozentpunkte bei Manuell + TSFEL und
   −5,99 bis −0,90 bei V2. Sie erfassen die Variation bei
   Resampling dieser Körper mit den beobachteten Vorhersagen.

Alle 18 inneren Auswahlen erfüllten ihre Qualitätsgrenzen und
verwendeten `balanced`. Die adaptierten Entscheidungen nutzten
stets die native SVM-Klasse als Ausgangspunkt, mit je fünf
One-vs-rest- und vier multinomialen Kalibrationen. Acht der neun
Regeln routeten unterhalb der Schwelle auf Klasse 2, eine auf das
größte (p_0, p_1, p_2). Die Schwellen lagen zwischen 0,75 und
0,95, bei den beiden fusionierten Sets zwischen 0,875 und 0,95.
Eine explizite Prüfung statt der Klassenumleitung ergab bei
Manuell + TSFEL 10,15 % und bei V2 12,85 % Prüfentscheidungen
über alle Messungen. Die kritischen Raten bleiben bei gleicher
Freigabeentscheidung gleich. Die später unabhängig zu
bewertenden Regeln und ihre betrieblich zulässige Prüfquote
bleiben festzulegen.

[Ergebnisbericht](../results/experiments/specimen_svm_nested_hpo_v1/report/status.md) ·
[Auswahl je Fold](../results/experiments/specimen_svm_nested_hpo_v1/selections.csv) ·
[MLflow-Run](../results/experiments/specimen_svm_nested_hpo_v1/run.json)

## Versuchsgrenzen und Vergleich

Die Auswertung verwendet ausschließlich die eingefrorenen 34.235
Trainingsmessungen aus 23 Prüfkörpern und deren drei äußere Folds.
Validation und Test bleiben geschlossen. Die derzeit untersuchten
Feature-Kandidaten stammen aus dem vorangegangenen Feature-Vergleich
und der explorativen Schwellenanalyse auf derselben OOF-Kohorte.
Die verschachtelte CV trennt die aktuelle Parameter- und Regelauswahl
von der Auswertung der jeweiligen äußeren Prüfgruppe. Die bereits
getroffene Feature-Vorauswahl begrenzt weiter die Aussagekraft für
unbekannte Messreihen. Die Ergebnisse sind explorativ.

Die historischen Grid-Entscheidungen werden als feste Referenz
herangezogen, sobald für die betreffenden äußeren Folds Ergebnisse
vorliegen. Die Auswertung prüft die Vorhersagehashes und ordnet
Messungs-ID, Prüfkörper, Messreihe, Label und Fold exakt zu. Bei
Teilergebnissen wird die Grid-Referenz auf dieselben äußeren Folds
beschränkt. In Zwischenberichten können verschiedene Feature-Kandidaten
unterschiedliche äußere Teilkohorten umfassen.

Der Grid-Referenz liegen fest konfigurierte RBF-Modelle zugrunde. Die
frühere Schwellenanalyse auf deren OOF-Dateien optimierte und bewertete
Betriebspunkte auf derselben OOF-Kohorte. Daraus beobachtete Schwellen
oder Betriebspunkte dienen nicht als Vorgabe der neuen inneren Suche.
[Historischer Feature-Vergleich](specimen_feature_grid.md) ·
[Explorative Schwellenanalyse](rbf_svm_oof_thresholds.md)

## Innere Auswahl

Der Startbereich des RBF-Rasters enthält die 27 Kreuzkombinationen aus
C = 0,1 / 0,3 / 1, Gamma = 0,0003 / 0,001 / 0,003 und drei festgelegten
Gewichtungsschemata. Hinzu kommen Breitenproben mit balancierten
Gewichten bei C = 0,03 / 3 / 10 und Gamma = 0,001 sowie bei Gamma
= 0,0001 / 0,01 und C = 0,3. Das Gewichtungsschema `balanced` folgt den
Klassenhäufigkeiten des jeweiligen Fits. `uniform` gewichtet alle
Klassen gleich und `cavity_cost` verwendet die Gewichte 1,0 / 2,5 /
1,6 / 0,85 in der Reihenfolge 0 bis 3. Die innere Gruppierung verwendet
den Seed 42 plus die äußere Fold-Nummer. Für die Kalibration kommen bis
zu drei Fits mit höchster Vierklassen-Güte, bis zu vier geeignete Fits
mit niedrigen kritischen Fehlerraten und der feste RBF-Referenzfit in
die Vorauswahl.

Ein innerer Score-Fit hält den gesamten inneren Validierungsblock
zurück. Für die Kalibration eines inneren Validierungskörpers werden
Scores anderer Körper desselben zurückgehaltenen Blocks verwendet.
Der Score-Fit hat deshalb weder diesen Zielkörper noch die
Kalibrationskörper dieses Schritts gesehen. Für die Anwendung auf
dem äußeren Fold verwendet der finale Kalibrator die inneren
OOF-Scores aus dessen Trainingskörpern. Die Schwelle und die Regel
werden innerhalb des jeweiligen äußeren Trainings bestimmt. Der
äußere Fold liefert jeweils die Bewertung der so festgelegten
Konfiguration.

Beide Auswahlpfade prüfen Mindestwerte für Macro-F1 (0,60),
Balanced Accuracy (0,64), Recall der Klassen 0 bis 3 (0,45 / 0,55 /
0,15 / 0,55), Präzision der Klasse 3 (0,82) und den schlechtesten
inneren Fold-Recall für Klasse 3 (0,45). Macro-F1 und Balanced Accuracy
dürfen zusätzlich höchstens 0,04 und der Recall der Klasse 3
höchstens 0,15 unter dem inneren Referenzfit liegen. Unter zulässigen
Optionen hat eine gepoolte kritische Rate unter 2 % Vorrang. Eine
gewichtete Kombination aus gepoolter kritischer Rate, der Rate für
1 → 3, einer nach Körper-Unterstützung stabilisierten Rate und der
schlechtesten Fold-Rate bestimmt die weitere Sicherheitsrangfolge.
Danach entscheiden Vierklassen-Güte und Fehlalarme. Fehlt eine
zulässige Option, greift die reguläre SVM-Entscheidung mit der
geringsten Unterschreitung der Mindestwerte. Die konkreten inneren
Ergebnisse und Auswahlgründe stehen in `selections.csv`.

## Lesen der Ergebnisse

- Kritische Fehlerraten zählen wahre Klasse 1 oder 2, die als 3
  freigegeben wurde. Die Teilraten nutzen jeweils ihre wahre Klasse
  als Nenner. Fehlalarme zählen ausschließlich 3 → 1/2.
- Die Freigabequote bezieht sich auf alle ausgewerteten Messungen.
  Die Nichtfreigabe wahrer Klasse 3 zählt zusätzlich Ausgaben 0 und
  gegebenenfalls explizite Prüfungen.
- Explizite Prüfungen werden separat als `-1` ausgewiesen. Die
  zugehörige Prüfquote und die Quote der Prüfungen unter wahren 3
  werden ohne Vierklassen-F1 oder Balanced Accuracy berichtet.
- `outcomes_by_true_class.csv` berichtet Präzision, Recall und F1 je
  Klasse und Auswertungseinheit für alle Vierklassenentscheidungen.
  Die Kennzahlen liegen zwischen 0 und 1. Zellen ohne definierten
  Nenner bleiben leer.
  [scikit-learn zu Präzision, Recall und F1](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
- Abweichungen zwischen historischer Entscheidung, regulärem Fit,
  Sicherheitsfit ohne Regel und adaptiertem Betriebspunkt stehen
  gepaart je Feature-Set, Fold und Körper zur Verfügung.
- `paired_specimen_intervals.csv` enthält das 2,5- und 97,5-Perzentil
  gepaarter Differenzen in Prozentpunkten aus 2.000 Prüfkörperziehungen
  mit Zurücklegen innerhalb der äußeren Folds (Seed 42). Die gepoolten
  Raten entstehen aus den gezogenen Fehleranzahlen und Nennern.
  Ziehungen ohne Nenner werden je Kennzahl ausgelassen. Die Spannen
  beschreiben die beobachtete Prüfkörperstreuung bei festgehaltenen Fits
  und Folds. Sie erlauben keine inferenzstatistische Aussage über neu
  trainierte Modelle oder unbekannte Messreihen.
- Die ungewichtete Verteilung der kritischen Körperquoten umfasst
  die Prüfkörper mit mindestens einer Messung der wahren Klasse 1
  oder 2. Hohe Raten bei kleinem Nenner bleiben in den Körperdateien
  sichtbar. Die gepoolte Rate gewichtet nach Messungen.
- Log-Loss, Multiklassen-Brier und Expected Calibration Error für
  `p_3` beschreiben die kalibrierten Wahrscheinlichkeiten der
  beiden Fits. Der ECE verwendet zehn gleich breite, nach Anzahl
  gewichtete Intervalle. Die Wahrscheinlichkeitsgüte validiert
  keine einzelne Freigabeentscheidung.

Der Bericht wird aus den gespeicherten äußeren OOF-Checkpoints
generiert. `status.md` enthält die gepoolten Betriebspunkte,
Fold-Entscheidungen sowie die Robustheit nach Körpern.
`decision_metrics.csv`, `paired_deltas.csv`,
`body_robustness.csv`, `outcomes_by_true_class.csv`,
`confusion_matrices.csv` und `probability_metrics.csv` halten
die Ergebnisse nach ihren jeweiligen Auswertungseinheiten fest.
`search_progress.csv` zählt erwartete und gespeicherte innere Fits,
vollständig bewertete Kandidaten und ausgewertete äußere Folds. Die
gelisteten Fehlerdateien bleiben auch nach einem erfolgreichen
Wiederholungsversuch als Verlauf sichtbar.
Die beiden mit `Agg` gerenderten Grafiken zeigen die Betriebspunkte
und die Differenzen der kritischen Raten pro Körper.

Die Auswertung lässt sich ohne zusätzliche Modellfits wiederholen.

```sh
.venv/bin/python -m src.analysis.svm_nested_report \
  --output-dir <Laufordner> --source results/experiments/specimen_feature_grid_v1
```

Das Training lässt sich mit denselben Pfaden fortsetzen. Vollständige
Checkpoints werden wiederverwendet.

```sh
.venv/bin/python -m src.experiments.svm_nested_hpo --jobs 2
```

Die scikit-learn-Dokumentation erläutert das methodische Prinzip
der getrennten inneren Auswahl und äußeren Bewertung, die verwendete
innere Gruppierung und die Kalibration auf separaten Daten.
[Verschachtelte CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) ·
[StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) ·
[Wahrscheinlichkeitskalibration](https://scikit-learn.org/stable/modules/calibration.html)
