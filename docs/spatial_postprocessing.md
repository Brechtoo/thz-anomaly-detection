# Räumliche Nachbearbeitung von SVM-Entscheidungen

Die konservative Nachbarschaftsregel verbesserte die Ergebnisse der
Prüfkörper-CV für Manuell + TSFEL + Physik V2 am deutlichsten. Sie senkte dort
kritische Freigaben von 631 auf 565 bei nahezu gleicher Vierklassen-Güte.
Die kritische Rate blieb mit 7,43 % vom Ziel unter 2 % entfernt. Die beiden
besonders schwierigen Körper und die isolierten Klasse-1-Punkte profitierten
in dieser Variante nicht.

## Eingaben und Trennung

Die Kohorte umfasst genau 34.235 Messpunkte aus 23 Trainingsprüfkörpern. Die
eingefrorenen äußeren Gruppen umfassen 8/8/7 Körper und 11.975/11.841/10.419
Punkte. Verwendet werden die nativen `prediction_regular`-Entscheidungen
und ihre gespeicherten `regular_p_0..3` aus der
[verschachtelten RBF-SVM-Suche](svm_nested_hpo.md) für alle drei dort
ausgewerteten Feature-Sets. Damit sind die Baselines gepaart und ihre
jeweiligen Modelle und Parameter ohne den äußeren Prüffold bestimmt.
Die frühere gepoolte Feature-Grid-OOF und die explorativ darauf
optimierten Schwellen wurden für die räumliche Auswahl nicht verwendet.
Grid-OOF aus einem anderen äußeren Fold kann von einem Modell stammen, das
auf dem gerade zurückgehaltenen Körper trainiert wurde.

Je äußerem Fold werden die inneren Hard Predictions und Scores des zuvor
innerlich ausgewählten regulären SVM-Kandidaten geladen. Ein innerer
Prüfkörper wird mit Scores anderer Körper aus demselben inneren Prüfblock
kalibriert. Somit haben weder der gemeinsame Score-Fit noch der Kalibrator
diesen Zielkörper gesehen. Die gespeicherten äußeren Wahrscheinlichkeiten
wurden anhand eines Kalibrators aus den zulässigen inneren OOF-Scores
überprüft. Die inneren Labels waren bereits an der SVM-Kandidatenauswahl
beteiligt. Die inneren Kennzahlen dienen daher zur Regelauswahl. Die
unabhängige Bewertung der kombinierten Auswahl erfolgt in den äußeren Folds.

IDs, Koordinaten, Labels, Körperzuordnung und Hashes werden je Artefakt
geprüft. Die X/Y-Position stammt aus dem ursprünglichen Akquisitionsdateinamen
der eingefrorenen Trainingskohorte. Keine wahren Labels fließen in die
räumliche Transformation ein. Validation und Test bleiben geschlossen.
Der Lauf trainiert keine SVMs und extrahiert keine Merkmale.

## Raum und Entscheidung

Nachbarn stammen ausschließlich aus demselben Prüfkörper und derselben
Messreihe. Referenzflächen werden ausgeschlossen. Die Distanz ist euklidisch
in den aufgezeichneten Koordinaten. Je Körper werden die Radien 1,5 / 2,5 / 4
mal dem Medianabstand zum nächsten Messpunkt geprüft. Hier beträgt die
Median-Distanz für jeden Körper eine Koordinateneinheit. Die minimalen und
medianen Nachbarzahlen sind 3/8, 6/20 und 13/48 für die drei Radien.
Die Mittelwerte verwenden gleiches Gewicht oder einen auf den Radius
begrenzten Gauß-Kernel mit Sigma gleich dem halben Radius. Ränder und Lücken
haben weniger Beiträge. Die Regeln setzen zwei tatsächlich gemessene
Nachbarn voraus und verwenden den Zielpunkt nur als separate Evidenz.

| Arm | Entscheidungsregel |
| --- | --- |
| Baseline | Unveränderte native Vierklassenentscheidung |
| Punktweise Kontrolle | Bei nativer Klasse 3 nur freigeben, wenn das eigene p(3) die Schwelle erreicht. Sonst Klasse 2. |
| Glättung | Eigene und benachbarte Wahrscheinlichkeiten mischen. Änderung nur bei ausreichendem Vorsprung und Nachbarunterstützung. |
| Lokaler Konsens | Bei ausreichendem gewichtetem Stimmenanteil der Nachbarn ändern, wenn der Zielpunkt die neue Klasse hinreichend unterstützt. |
| Nachbarschaftsrettung | Nur native Klasse 3 prüfen. Eigenes p(1)+p(2) muss mindestens 0,10 betragen. Das Nachbarmittel muss Schwelle und Mehr-Evidenz erfüllen. Eigene und benachbarte Wahrscheinlichkeit bestimmen dann die Entscheidung zwischen 1 und 2. |

Die räumlichen Regeln lassen keinen direkten Übergang von nativer Klasse
1/2 zu 3 zu. Bei eigener Evidenz p(1)+p(2) ≥ 0,65 schützen sie eine native
1/2-Entscheidung und sperren eine neue Freigabe. Der Test umfasst 79
Kandidaten (24 je räumlicher Familie plus sieben Punktregeln).
Die Auswahl priorisiert weniger 1→3 und dann weniger 1/2→3. Sie verlangt
eine Verbesserung in mindestens einer dieser Größen, ohne die andere zu
verschlechtern. Auf den inneren Gruppen dürfen Macro-F1, Balanced Accuracy
und Recall 1 jeweils höchstens drei Prozentpunkte sinken. Für Recall 3,
Präzision 3 und Freigabequote gelten Grenzen von zwölf, zwei und zehn
Prozentpunkten. Für Recall 3 je innerem Fold gilt eine Grenze von 20
Prozentpunkten. Fehlalarme und 0→3 dürfen höchstens um zwölf bzw. zwei
Prozentpunkte steigen. Der Baseline-Fallback greift je Strategie und Fold,
wenn keine Regel die Auswahlbedingungen erfüllt. Die rein innerliche
Auswahl zwischen den drei räumlichen Familien (`selected_spatial`) ergab
stets dieselben Entscheidungen wie die Nachbarschaftsrettung mit Fallback.

## Ergebnisse auf zuvor zurückgehaltenen Körpern

Alle Zeilen je Feature-Set verwenden dieselben Messungs-IDs. Die kritische
Rate bezieht sich auf 7.600 wahre Punkte der Klassen 1 und 2. Die getrennten
Raten beziehen sich auf 5.297 bzw. 2.303 Punkte. Fehlalarme sind 3→1/2 unter
22.082 wahren Klasse-3-Punkten. Freigabe ist der Anteil aller Vorhersagen
mit Klasse 3. Raten stehen in Prozent, F1 und Balanced Accuracy auf der
Skala 0–1.

| Feature-Set | Arm | 1→3 % | 2→3 % | Kritisch % | Fehlalarm % | Freigabe % | Macro-F1 | Bal. Acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Manuell | Baseline | 7,49 | 10,51 | 8,41 | 20,79 | 49,47 | 0,6788 | 0,7580 |
| Manuell | Punktweise | 7,46 | 10,51 | 8,38 | 21,25 | 49,16 | 0,6770 | 0,7568 |
| Manuell | Glättung/Konsens¹ | 7,49 | 10,51 | 8,41 | 20,79 | 49,47 | 0,6788 | 0,7580 |
| Manuell | Nachbarschaftsrettung | 7,42 | 10,51 | 8,36 | 20,81 | 49,44 | 0,6787 | 0,7581 |
| Manuell + TSFEL | Baseline | 9,02 | 11,64 | 9,82 | 18,41 | 51,51 | 0,7003 | 0,7730 |
| Manuell + TSFEL | Punktweise | 8,06 | 11,51 | 9,11 | 19,05 | 50,94 | 0,6976 | 0,7718 |
| Manuell + TSFEL | Glättung/Konsens¹ | 9,02 | 11,64 | 9,82 | 18,41 | 51,51 | 0,7003 | 0,7730 |
| Manuell + TSFEL | Nachbarschaftsrettung | 8,00 | 11,59 | 9,09 | 18,49 | 51,30 | 0,7017 | 0,7754 |
| Manuell + TSFEL + Physik V2 | Baseline | 5,85 | 13,94 | 8,30 | 15,78 | 54,90 | 0,7666 | 0,8325 |
| Manuell + TSFEL + Physik V2 | Punktweise | 5,81 | 13,94 | 8,28 | 15,78 | 54,89 | 0,7666 | 0,8325 |
| Manuell + TSFEL + Physik V2 | Glättung¹ | 5,85 | 13,94 | 8,30 | 15,78 | 54,90 | 0,7666 | 0,8325 |
| Manuell + TSFEL + Physik V2 | Lokaler Konsens | 5,85 | 13,94 | 8,30 | 15,77 | 55,03 | 0,7670 | 0,8319 |
| Manuell + TSFEL + Physik V2 | Nachbarschaftsrettung | 4,80 | 13,50 | 7,43 | 16,30 | 54,37 | 0,7655 | 0,8343 |

¹ Alle betreffenden Folds fielen auf die Baseline zurück. Bei Physik V2
änderte der lokale Konsens in einem Fold 56 Entscheidungen. Die kritischen
Freigaben blieben gleich, während 0→3 von 248 auf 257 stieg.
`inner_trials.csv` enthält alle Suchversuche, einschließlich nicht
zugelassener Konfigurationen.

- **Physik V2:** Die Rettung verhindert 56 Fälle 1→3 und zehn Fälle 2→3.
  Sie verursacht 116 zusätzliche Fehlalarme und reduziert Freigaben um
  182 von 18.795. Recall 3 sinkt von 81,13 auf 80,61 %, die Präzision
  der erteilten Freigaben steigt von 95,32 auf 95,63 %. Die kritischen
  Fold-Raten sinken von 3,49/8,38/14,58 % auf 2,88/6,70/14,44 %.
  Sieben Körper zeigen einen Rückgang. Allein
  `messreihe_1_pk_09` trägt 44 der 66 verhinderten Ereignisse bei.
  Gegenüber der Punktregel beträgt der gepaarte Unterschied −0,84
  Prozentpunkte (deskriptives 95-%-Prüfkörper-Bootstrap-Intervall −2,10
  bis −0,11). Die Punktregel gab 54,89 % frei. Die Betriebspunkte
  unterscheiden sich, daher belegt der Vergleich keinen Vorteil bei
  exakt gleicher Freigabequote.
- **Manuell + TSFEL:** 55 kritische Freigaben entfallen gegenüber der
  Baseline. Die Punktregel verhindert 54, produziert dabei aber 141
  zusätzliche Fehlalarme. Die Nachbarschaftsregel verursacht 18
  zusätzliche Fehlalarme. Der Unterschied der kritischen Rate zwischen
  beiden Regeln beträgt −0,01 Prozentpunkte, das deskriptive Intervall
  reicht von −0,29 bis +0,43. Für **Manuell** sind die vier verhinderten
  Fälle auf einen einzigen Körper konzentriert.
- **Kleine Befunde und Übergänge:** Eine nachgelagerte Auswertung
  verwendet einen festen Radius von 1,5 Messabständen und verbundene
  Klasse-1-Annotationen. Bei Physik V2 bleiben 22/31 kritische
  Einzelfälle unverändert. In Komponenten von zwei bis vier Punkten
  sinken sie von 26/41 auf 23/41, in größeren von 262/5.225 auf
  209/5.225. Bei wahren 1/2-Punkten neben Klasse 3 sinken die kritischen
  Fehler von 527 auf 497, abseits solcher Übergänge von 104 auf 68.
  Die Fehlalarmrate an der Klasse-3-Seite des Übergangs steigt von
  42,77 auf 44,94 %. Diese Schnitte beschreiben Annotationen im
  Messraster und erlauben keine Aussage über die physische Größe.
- **Schwierige Körper:** Bei Physik V2 bleiben 117/132 kritische
  Punkte in `messreihe_3_pk_06` und 11/11 in `messreihe_2_pk_08`
  unverändert. Auch die übrigen Baselines zeigen dort durch die
  Nachbarschaftsrettung keine Abnahme. Die Regel hilft vor allem, wenn
  die umliegenden Vorhersagen noch Hohlraum- oder Unsicherheitsevidenz
  enthalten. Systematische Fehler, die sich flächig wiederholen,
  bleiben bestehen.

Die Prüfkörper-CV stützt einen Nutzen unter diesen Messbedingungen,
vor allem für lokale Korrekturen auf Basis der Physik-V2-Wahrscheinlichkeiten.
Der Bootstrap mit 2.000 Ziehungen und Seed 42 resampelt gepaarte
Körperentscheidungen. Er beschreibt Unterschiede für diese beobachteten
Fits. Die drei äußeren Trainingsmengen überlappen, und die Messreihen sind
in mehreren Folds vertreten. Die Feature-Vorauswahl stammt zudem aus
früherer Arbeit an derselben Kohorte. Unabhängige neue Messreihen und eine
betriebliche Freigabeschwelle bleiben unbewertet.

## Reproduktion und Artefakte

```sh
.venv/bin/python -m src.experiments.spatial_postprocessing \
  --output-dir results/experiments/specimen_spatial_postprocessing_repeat
```

Ein neuer Zielordner ist erforderlich. Der
[verifizierte Lauf](../results/experiments/specimen_spatial_postprocessing_v1_verified/)
enthält `plan.json` mit Eingabe- und Code-Hashes, `selections.csv`,
`inner_trials.csv`, gepaarte `predictions.parquet` und einen separaten
lokalen MLflow-Store. MLflow-Run-ID ist `f9815a33a5954e08b7c580ea4690d178`.
Im [Report-Ordner](../results/experiments/specimen_spatial_postprocessing_v1_verified/report/)
liegen alle gepoolten Kennzahlen einschließlich Precision, Recall und F1
für jede Klasse sowie Fold-, Prüfkörper- und Raumdiagnosen und Intervalle
gegen Baseline und Punktregel. Der zweite Lauf erzeugte identische
616.230 gepaarte Entscheidungszeilen wie der erste Lauf. Für die berichtete
Ausführung stimmen die gespeicherten Hashes aller 96 Eingabeartefakte und
der fünf erfassten Code-Dateien.
