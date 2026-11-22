# Explorative Schwellenwertanalyse der HeRAI-RBF-SVMs

**Manuell + TSFEL bietet beim gepoolten Ziel unter 2 % die höchste Freigabequote
(36,81 %).** Am beobachteten Betriebspunkt fällt die kritische Fehlerrate von
8,16 % auf 1,987 %. Dafür gehen 15,60 % aller Messungen zusätzlich in die Prüfung.
Fold 2 bleibt bei 4,16 %. Eine strengere Regel liegt in allen drei beobachteten
Folds unter 2 % und gibt nur noch 23,08 % der Messungen frei. Einzelne Körper
haben auch an diesem Punkt hohe kritische Fehlerraten.

## Daten und Entscheidungsregeln

Die Analyse umfasst für jedes Modell genau **34.235 OOF-Messungen aus
23 Trainingsprüfkörpern** (Fold-Größen 11.975, 11.841 und 10.419). Die wahren
Klassen enthalten 4.553, 5.297, 2.303 und 22.082 Messungen in der Reihenfolge
0, 1, 2, 3. Für die neun Fold-Checkpoints stimmen Messungs-ID, Prüfkörper,
Messreihe, Label und Fold mit der eingefrorenen Trainingskohorte überein.
Vorhersage-Hashes stimmen mit den lokalen MLflow-Artefakten überein. Alle
Wahrscheinlichkeiten sind endlich, liegen in [0, 1] und summieren sich innerhalb
der Prüftoleranz zu eins. Die explizite Klassenreihenfolge lautet `p_0` bis
`p_3`. Alle drei SVMs verwenden das archivierte Setup mit `C=0.3`,
`gamma=0.001`, balancierten Gewichten und Seed 42. Es wurden keine Fits
ausgeführt. Die Analyse liest ausschließlich Trainings-OOF-Daten.

Die primäre Regel übernimmt `SVC.predict` und gibt dessen Klasse 3 nur bei
`p_3 >= t` frei. Als zweite Ausgangsentscheidung dient der Argmax aller vier
Wahrscheinlichkeiten, auf den dieselbe Schwelle angewendet wird. Bei beiden
Regeln bleibt eine Ausgangsentscheidung 0, 1 oder 2 erhalten. Wird Ausgangsklasse
3 durch die Schwelle zurückgehalten, führen drei Alternativen zur expliziten
Prüfung, zu Klasse 2 oder zur Klasse mit dem größten Wert unter
`p_0, p_1, p_2`. Die Prüfung hat den separaten Entscheidungscode `-1`.

Berechnet wurden 101 feste Schwellen von 0 bis 1 in 0,01-Schritten. Zusätzlich
wurde pro Modell und Ausgangsregel die kleinste Schwelle mit **streng unter
2 %** ermittelt, einmal gepoolt und einmal als Maximum der drei jeweils
erforderlichen Fold-Schwellen. Die exakten Punkte behandeln gleiche
Wahrscheinlichkeiten gemeinsam. Die Werte im Bericht sind gerundet. Auswahl und
Bewertung dieser Punkte verwenden dieselben OOF-Daten und sind explorativ.

Die kritische Fehlerrate bezieht sich auf wahre Klasse 1 oder 2, getrennte Raten
auf die jeweilige wahre Klasse. Fehlalarme sind ausschließlich 3 → 1/2, bezogen
auf wahre Klasse 3. Freigabe und zusätzliche Prüfung beziehen sich auf alle
Messungen, sofern keine wahre Klasse angegeben ist. Die Nichtfreigabe wahrer
Klasse 3 zählt auch 3 → 0 und explizite Prüfungen.

## Ergebnisse und Kosten der Nichtfreigabe

Die gepoolte Grenze lässt höchstens 151 kritische Fehler unter 7.600 Messungen
der Klassen 1 und 2 zu. Die drei Modelle erreichen diese Grenze mit den folgenden
beobachteten Betriebspunkten für `predict` und explizite Prüfung.

| Features | Kritisch ohne Schwelle | Schwelle ≈ | Kritisch | 1 → 3 (n / %) | 2 → 3 (n / %) | Freigabe | Prüfung alle | Prüfung wahre 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Manuell + TSFEL | 620 (8,16 %) | 0,95170 | 151 (1,987 %) | 97 / 1,831 | 54 / 2,345 | 36,81 % | 15,60 % | 20,82 % |
| Manuell | 653 (8,59 %) | 0,95291 | 151 (1,987 %) | 105 / 1,982 | 46 / 1,997 | 34,55 % | 16,16 % | 21,19 % |
| Manuell + TSFEL + Physik V2 | 667 (8,78 %) | 0,96877 | 151 (1,987 %) | 108 / 2,039 | 43 / 1,867 | 35,72 % | 18,07 % | 24,73 % |

Bei der festen Schwelle 0,90 liegen die kritischen Raten bei 3,95 %, 4,12 %
und 5,17 %. Selbst bei 0,95 bleiben alle drei über 2 %. Auf dem festen Raster
unterschreiten die ersten beiden das Ziel bei 0,96 und V2 bei 0,97.
Bei Schwelle 1 gibt keines der drei Modelle eine Messung frei. Die Freigabequote
ist deshalb notwendig, um die Fehlerreduktion einzuordnen.

| Features | Fehlalarme bei Prüfung | Fehlalarme bei größtem p₀–p₂ | Fehlalarme bei Klasse 2 | Wahre 3 insgesamt ohne Freigabe |
| --- | ---: | ---: | ---: | ---: |
| Manuell + TSFEL | 17,04 % | 25,49 % | 37,86 % | 44,29 % |
| Manuell | 20,08 % | 26,58 % | 41,26 % | 47,78 % |
| Manuell + TSFEL + Physik V2 | 18,38 % | 33,97 % | 43,11 % | 45,57 % |

Das Routing ändert bei gleicher Ausgangsregel und Schwelle die Entscheidung
über Freigabe und damit die kritische Rate nicht. Für die Prüfstrategie bleibt
die formale Fehlalarmrate auf dem Ausgangswert. Der zusätzliche Aufwand wird
als Prüfung gezählt. Das größte `p_0`–`p_2` erhöht auch die Rate 3 → 0
(z. B. von 6,43 % auf 18,80 % bei Manuell + TSFEL). Die Klasse-2-Strategie
zählt jede durch die Schwelle zurückgehaltene wahre 3 als Fehlalarm.

Der Wahrscheinlichkeits-Argmax ohne Zusatzschwelle hat 24,84–27,78 %
kritische Fehler. Bei den gepoolten Zielpunkten gibt er dieselben Messungen
frei wie die entsprechende `predict`-Regel. Die formale Fehlalarmrate mit
Prüfentscheidung sinkt auf 2,54–2,88 %, während die Prüfquote auf 32,98–36,45 %
aller Messungen und 39,17–43,12 % der wahren 3 steigt. Diese Variante verlagert
damit erheblichen Aufwand in den Prüfpfad.

**Die Rangfolge hängt vom Vergleichspunkt ab.** Auf den harten OOF-Vorhersagen
liegen Manuell + TSFEL, Manuell und V2 in dieser Reihenfolge. Bei gleicher
gepoolter Zielrate bietet V2 mehr Freigaben und weniger formale Fehlalarme als
Manuell. Manuell + TSFEL führt weiterhin. Bei der gemeinsamen Schwelle 0,97
sind die kritischen Fehler von Manuell und Manuell + TSFEL nahezu gleich
(85 gegenüber 86), bei 0,98 haben beide 58. V2 hat bei 0,98 mehr Freigaben
(29,76 %) und mehr kritische Fehler (1,39 % gegenüber jeweils 0,76 %).

## Stabilität über Folds und Prüfkörper

| Features | Fold 0 / 1 / 2 am gepoolten Ziel | Schwelle ≈ für alle Folds | Kritisch gepoolt | Freigabe | Prüfung alle | Prüfung wahre 3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Manuell + TSFEL | 0,50 / 1,83 / 4,16 % | 0,98241 | 0,645 % | 23,08 % | 29,33 % | 41,27 % |
| Manuell | 0,29 / 1,98 / 4,25 % | 0,98553 | 0,592 % | 22,09 % | 28,63 % | 39,66 % |
| Manuell + TSFEL + Physik V2 | 0,43 / 0,92 / 5,45 % | 0,99216 | 0,566 % | 18,34 % | 35,44 % | 51,02 % |

Fold 2 bestimmt die strengere Schwelle und bleibt dort bei jeweils 41 kritischen
Fehlern unter 2.092 kritischen Messungen (1,960 %). Für wahre Klasse 1 liegt er
dabei weiterhin bei 2,83–2,91 %. V2 erzielt an seinem strengeren Punkt die
niedrigste gepoolte Rate, bei der geringsten Freigabequote.

Von 23 Körpern haben 22 mindestens eine Messung der Klassen 1 oder 2.
Am gepoolten Ziel verbessern sich gegenüber `predict` 15 Körper bei
Manuell + TSFEL, 13 bei Manuell und 18 bei V2. Die jeweils drei stärksten
Verbesserungen machen 79 %, 78 % bzw. 68 % der verhinderten kritischen Fehler aus.
Am gepoolten Ziel liegen 9, 9 bzw. 8 der 22 Körper bei mindestens 2 %.
Die ungewichteten Mittelwerte über diese 22 Körper betragen 9,01 %, 13,11 %
und 9,95 %. Die gepoolte Kennzahl gewichtet nach Anzahl der Messungen.

`messreihe_3_pk_06` hat am gepoolten Ziel je nach Modell 63–79 verbleibende
kritische Fehler unter 132 kritischen Messungen (47,73–59,85 %). Auch bei den
strengeren Schwellen haben alle Modelle jeweils drei Körper bei mindestens 2 %.
Auf `messreihe_3_pk_06` verbleiben 29–30 von 132 Fehlern. Auf
`messreihe_2_pk_08` werden bei allen drei Modellen 11 von 11 kritischen
Messungen freigegeben. Kleine Nenner sind pro Körper ausgewiesen.

## Aussagekraft der Wahrscheinlichkeiten

| Features | Log-Loss | Multiklassen-Brier | ECE p₃ gepoolt | ECE p₃ über die Folds | Mittleres p₁+p₂ / beobachtete 1/2 unter predict = 3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Manuell + TSFEL | 0,4068 | 0,2080 | 1,68 Pp. | 2,18–4,13 Pp. | 2,12 / 3,46 % |
| Manuell | 0,4336 | 0,2169 | 1,83 Pp. | 2,75–4,18 Pp. | 1,97 / 3,76 % |
| Manuell + TSFEL + Physik V2 | 0,3375 | 0,1711 | 1,40 Pp. | 2,00–2,99 Pp. | 2,23 / 3,62 % |

Der Brier-Wert summiert die vier quadrierten Fehler je Messung. Der
Expected Calibration Error (ECE) verwendet zehn gleich breite Intervalle
und ihre jeweilige Besetzung als Gewicht. V2 hat hier die beste globale
Wahrscheinlichkeitsgüte. Bei allen drei Modellen unterschätzt die
mittlere Wahrscheinlichkeit `p_1+p_2` das beobachtete Risiko unter den
ursprünglich freigegebenen Messungen. Im obersten `p_3`-Intervall
([0,9; 1]) betragen die mittleren Scores 97,23–97,61 %, die beobachteten
Anteile wahrer Klasse 3 liegen bei 95,99–96,70 %. Auch einige Hohlräume
bekommen sehr hohe Scores. Das 95. Perzentil von `p_3` unter den
ursprünglichen Fehlern 1 → 3 liegt je nach Modell bei 0,991–0,996.

`SVC.predict` und Wahrscheinlichkeits-Argmax liefern für 16,37–21,80 %
der Messungen unterschiedliche Klassen. Die archivierte native LibSVM-
Kalibration läuft punktweise innerhalb des jeweiligen Fold-Trainings.
Die OOF-Auswertung ist nach Prüfkörpern getrennt. Die Kalibration selbst
wurde nicht nach Prüfkörpern angepasst. Die Ergebnisse stützen die Nutzung
der Scores zur Rangordnung und zeigen die Grenzen absoluter
Sicherheitsinterpretationen. Beispielsweise werden an den gepoolten
Zielpunkten weiterhin 1,25–3,25 % der wahren Klasse 0 freigegeben.

## Empfehlung für die folgende Optimierung

**Manuell + TSFEL als Hauptkandidat, V2 als Vergleich und Manuell als
Aufwandsbaseline.** V2 rechtfertigt den Vergleich durch seine globale
Wahrscheinlichkeitsgüte und die niedrigere Klasse-2-Fehlerrate am gepoolten
Zielpunkt. Manuell + TSFEL erreicht dort den besseren Kompromiss aus
Freigabequote und Prüfaufwand. Die explizite Prüfung eignet sich als
primärer Rückfallpfad, sofern die betrieblich mögliche Prüfquote zuvor
festgelegt wird.

Die eingefrorenen drei Gruppen-Folds als äußere Bewertung verwenden.
`C`, `gamma`, gegebenenfalls Klassengewichte, Score und Entscheidungsregel
gemeinsam in einer **inneren Gruppen-CV auf den jeweiligen äußeren
Trainingskörpern** auswählen. Vorverarbeitung und optionale gruppengetrennte
Kalibration gehören ebenfalls in diese innere Auswahl. Native Scores und
gruppenkalibrierte Scores getrennt vergleichen. Die je Außenfold gewählte
Regel genau einmal auf dessen zurückgehaltene Körper anwenden. Kriterien für
eine sinnvolle Mindestfreigabe oder maximale Prüfquote vor der Suche festlegen.
Innerhalb dieser Grenzen kritische Fehler vorrangig minimieren, anschließend
Fehlalarme und Prüfaufwand vergleichen. Zusätzlich Fehler je Klasse, Fold und
Körper sowie Freigaben aus Klasse 0 ausweisen. Die drei Feature-Sets sind
bereits anhand dieser OOF-Kohorte vorausgewählt. Die verschachtelte CV prüft
die weitere Auswahl von Fit- und Entscheidungsparametern innerhalb des
Entwicklungsdatensatzes. Die Prüfung auf dem eingefrorenen Testsplit folgt
erst nach Festlegung der endgültigen Konfiguration.

## Artefakte und Wiederholung

- [Trade-off-Kurven](../results/analysis/rbf_svm_oof_thresholds_v1/tradeoffs.png) und
  [Kalibration und Score-Verteilungen](../results/analysis/rbf_svm_oof_thresholds_v1/wahrscheinlichkeiten.png).
- [Schwellenraster](../results/analysis/rbf_svm_oof_thresholds_v1/sweep.csv),
  [Betriebspunkte global, je Fold und je Prüfkörper](../results/analysis/rbf_svm_oof_thresholds_v1/operating_points.csv) und
  [Entscheidungen nach wahrer Klasse](../results/analysis/rbf_svm_oof_thresholds_v1/outcomes_by_true_class.csv).
- [Kalibrationsgüte](../results/analysis/rbf_svm_oof_thresholds_v1/calibration.csv),
  [Reliabilitätsintervalle](../results/analysis/rbf_svm_oof_thresholds_v1/reliability_bins.csv) und
  [Quantile von p₃](../results/analysis/rbf_svm_oof_thresholds_v1/p3_quantiles.csv).
- [Eingabe- und Artefakt-Hashes](../results/analysis/rbf_svm_oof_thresholds_v1/manifest.json) und
  [MLflow-Run](../results/analysis/rbf_svm_oof_thresholds_v1/mlflow_run.json) im separaten
  Experiment `thz-rbf-svm-oof-thresholds`.

Wiederholung aus dem Projektstamm in einen neuen Ausgabeordner.

```sh
.venv/bin/python -m src.analysis.rbf_svm_oof_thresholds --no-log \
  --output results/analysis/rbf_svm_oof_thresholds_repeat
```
