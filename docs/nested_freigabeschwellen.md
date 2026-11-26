# Konservative Freigabe auf den Nested-HPO-Checkpoints

**Die 2-%-Marke ist gepoolt mit allen sechs Familien erreichbar. Der Preis ist die Prüfung vieler Messungen und der Gewinn hängt stark vom Prüfkörper ab.** Der Vergleich umfasst drei Feature-Sets und dieselben 34.235 äußeren OOF-Messungen von 23 Trainingsprüfkörpern. 7.600 Messungen tragen Label 1 oder 2. Validation und Test blieben geschlossen.

## Warum die bisherigen Ergebnisse auseinanderliegen

Die frühere feste Grid-SVM mit manuellen Features und TSFEL erreichte 151/7.600 kritische Fehler (1,987 %) bei 36,81 % Freigaben und 15,60 % expliziter Prüfung. Dieselben äußeren OOF-Labels bestimmten und bewerteten die Schwelle von rund 0,95170. Fold 2 lag bei 4,16 %. Damals wurden LibSVM-Wahrscheinlichkeiten verwendet. Die Nested-HPO wählte Fit, gruppenbasierte Kalibration und Vierklassenregel im inneren Training aus. Die absoluten Schwellen sind zwischen beiden Verfahren nicht übertragbar. [Historische Schwellenanalyse](rbf_svm_oof_thresholds.md)

In **allen 54** inneren Auswahlen (6 Familien × 3 Feature-Sets × 3 äußere Folds) enthielten die gespeicherten kalibrierten Versuche Betriebspunkte unter 2 %. Jeder solche Versuch unterschritt die jeweilige Mindest-Sensitivität für Klasse 3. Die Nested-Auswahl verlangte mindestens 55 % Recall für Klasse 3 und höchstens 15 Prozentpunkte Verlust gegenüber der inneren Referenz, neben weiteren Qualitätsgrenzen. Das Ausweichen von Klasse 3 auf Klasse 2 oder eine Prüfung ändert bei gleicher Freigabemenge nur Fehlalarme und Prüfaufwand. Für weniger kritische Fehler muss die Freigabemenge selbst enger werden.

## Schwellen und Vergleich

Pro äußerem Fold blieb der schon innerlich ausgewählte adaptive Fit samt Ausgangsentscheidung, Kalibrationsmethode und Mindestschwelle fix. Auf seinen inneren OOF-Scores wurde die niedrigste zusätzliche Schwelle für strikt unter 2 % bestimmt, einmal **gepoolt** und einmal **je Fold** (Maximum der drei inneren Mindestschwellen). Gleiche Scores werden gemeinsam behandelt. Die innere Kalibration jeder Messung verwendet nur andere Prüfkörper aus ihrem inneren Prüffold. Für äußere Vorhersagen dient die Kalibration aus dem gesamten inneren OOF-Satz. Ausgangsklasse 3 unterhalb der Schwelle geht in explizite Prüfung. Die reproduzierte ursprüngliche Freigabemenge stimmt in allen 54 Fällen mit dem archivierten adaptiven Ergebnis überein. Es gab keine neuen Basismodell-Fits.

**Manuelle Features + TSFEL**, derselbe Vergleichspunkt wie beim früheren 2-%-Ergebnis. Zahlenpaare zeigen die gepoolt bzw. je innerem Fold gewählte Schwelle. Alle Angaben sind Prozentsätze auf äußeren Prüfkörpern. Prüfung und Freigabe beziehen sich auf alle Messungen, Fehlalarme auf wahre Klasse 3. Die übrigen Messungen behalten die Klassenentscheidung 0, 1 oder 2. Die Prüfquote zählt explizit zurückgestellte Entscheidungen vor jeder menschlichen Nachprüfung.

| Familie | Kritisch gepoolt / je Fold | Prüfung gepoolt / je Fold | Freigabe gepoolt / je Fold | Fehlalarm bei beiden |
|---|---:|---:|---:|---:|
| RBF-SVM | 1,07 / 0,61 | 39,94 / 48,00 | 14,21 / 6,15 | 15,32 |
| MLP | 0,68 / 0,55 | 47,83 / 51,13 | 15,44 / 12,13 | 6,33 |
| Logistische Regression | 0,93 / 0,55 | 39,51 / 48,57 | 20,84 / 11,78 | 8,97 |
| HGB | 1,32 / 1,16 | 52,09 / 54,77 | 14,63 / 11,95 | 4,13 |
| LightGBM | 1,38 / 1,00 | 43,66 / 49,80 | 18,70 / 12,56 | 9,09 |
| XGBoost | 1,79 / 1,61 | 41,83 / 46,56 | 18,10 / 13,37 | 11,68 |

Über alle drei Feature-Sets liegen bei gepoolter innerer Wahl 17/18 Einzelvarianten außen strikt unter 2 %. XGBoost mit manuellen Features erreicht genau 152/7.600 = 2 %. Bei der strengeren Wahl sind es 18/18, wobei die manuelle SVM **keine einzige Messung** freigibt. Mit manuellen Features + TSFEL überschreiten bei gepoolter Wahl alle sechs Einzelmodelle in Außenfold 2 das Ziel (2,29–6,17 %). Selbst die strenge logistische Regression liegt dort bei 42/2.092 = 2,008 %, das strenge MLP bei 41/2.092 = 1,960 %.

## Was gemeinsame Fehler über Ensembles zeigen

Zwei feste Vetoregeln verwenden die SVM als Ausgangsmodell. Eine SVM-Freigabe geht in Prüfung, wenn das MLP oder, bei der zweiten Regel, mindestens eine der fünf anderen Familien die Freigabe verweigert. Für alle Mitglieder gilt jeweils dieselbe Strategie zur inneren Schwellenwahl. Die zusätzliche SVM-Kontrolle wählt ihre Schwelle ohne äußere Labels so, dass sie innen höchstens ebenso viele Messungen freigibt wie das jeweilige Ensemble. Damit lässt sich der Nutzen eines zweiten Scores vom Effekt einer strengeren SVM-Schwelle unterscheiden.

| M+TSFEL, gepoolte innere Wahl | 1→3 / 2→3 | Kritisch % | Prüfung % | Freigabe % | Fehlalarm % |
|---|---:|---:|---:|---:|---:|
| SVM + MLP | 24 / 1 | 0,33 | 47,85 | 6,30 | 15,32 |
| SVM bei innerem Prüfaufwand des Paars | 43 / 17 | 0,79 | 49,69 | 4,46 | 15,32 |
| Einstimmigkeit aller sechs | 13 / 0 | 0,17 | 50,03 | 4,12 | 15,32 |
| SVM bei innerem Prüfaufwand aller sechs | 7 / 3 | 0,13 | 53,59 | 0,56 | 15,32 |

- SVM und MLP teilen 25 kritische Fehler, weitere 56 betreffen nur die SVM und 27 nur das MLP (Jaccard 0,23). HGB und LightGBM teilen 86 Fehler (Jaccard 0,72). Der MLP-Vetoentscheid verhindert auch gegenüber der SVM mit angeglichenem innerem Aufwand 35 kritische Fehler, bei 630 weniger äußeren Prüfungen. Gegenüber dem MLP allein spart das Paar 27 Fehler bei fast gleicher Prüfquote, erhöht aber die Fehlalarmrate von 6,33 % auf 15,32 %.
- Alle 25 verbleibenden Fehler des Paars stammen aus Fold 2, davon 19/132 kritische Sollmessungen auf `messreihe_3_pk_06` und 6/11 auf `messreihe_2_pk_08`. Mit allen sechs Modellen sinkt die erste Zahl auf 7/132, die zweite bleibt bei 6/11. Alle zwölf zusätzlich verhinderten Fehler entfallen also auf denselben Körper. In Fold 1 gibt das Paar nur 2 von 11.841 Messungen frei, der Sechserkonsens eine einzige.
- Das Paar bietet hier einen beobachteten Vorteil gegenüber einer strengeren SVM. Ein allgemeiner Ensemble-Vorsprung folgt daraus nicht. Bereits mit M+TSFEL bei der strengeren Einzel-Fold-Wahl hat die auf Paar-Aufwand eingestellte SVM weniger kritische Fehler als das Paar (10 gegenüber 17), bei geringerer Freigabe (0,54 % gegenüber 3,11 %). Ein betrieblich tragbarer Prüfaufwand wurde nicht vorgegeben. Feature-Vorauswahl und der Vergleich dieser Strategien auf denselben 23 Körpern begrenzen die Übertragbarkeit.

[Außenmetriken je Feature-Set, Fold und Prüfkörper](../results/analysis/nested_release_review_v1/metrics.csv) · [Paarweise Fehlerüberlappung](../results/analysis/nested_release_review_v1/overlap.csv) · [Innere Schwellen und Freigabezahlen](../results/analysis/nested_release_review_v1/thresholds.csv)

Reproduktion aus dem Projektstamm mit `.venv/bin/python -m src.analysis.nested_release_review --output results/analysis/nested_release_review_repeat`. Der lokale MLflow-Lauf `4f25216fc1014e61b75d3520f3fcffe8` protokolliert beide Plan-Prüfsummen, Seed 42, Parameter, Tabellen und gepoolte Kennzahlen.
