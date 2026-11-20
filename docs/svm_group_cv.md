# RBF-SVM auf eingefrorenen Prüfkörper-Splits

## Versuchsaufbau

Beide Arme verwenden die vier Originalklassen 0/1/2/3, die eingefrorenen drei
Folds auf 34.235 Trainingspunkten aus 23 Prüfkörpern und die bestehenden Holdouts
mit 7.521 Validierungspunkten aus 5 bzw. 10.464 Testpunkten aus 7 Prüfkörpern.
Die Zuordnung der Trainingspunkte wird gegen die historischen, punktgenauen
Fold-Zuordnungen geprüft. Die Holdouts gehen ausschließlich in die Auswertung
der nach der CV festgelegten Modelle ein.

- **Manuell, 52 Features**. Der historische Cache `early_legacy_windows` enthält
  die 17 ankerunabhängigen und 35 früh verankerten Merkmale mit den bisherigen
  Zeitfenstern. Für die Holdout-Körper verwendet derselbe Extraktor je Signal
  die unveränderte Früherkennungsheuristik. Der Median-Imputer fittet je Fold
  auf dessen Trainingskörpern und für den Endfit auf den 23 Trainingskörpern.
- **tsfresh**. Die Rohbatches des referenzfreien Prüfkörper-Holdouts liefern
  je Signal alle 777 Features aus dem `efficient`-Preset. Imputation und
  überwachte Merkmalsauswahl werden mit der bestehenden tsfresh-Funktion
  für jeden CV-Fold auf dessen Trainingskörpern bestimmt. Beim Endfit kommt
  die manifestgeprüfte, auf allen 23 Trainingskörpern vorverarbeitete Tabelle
  mit 708 Features aus dem bestehenden Baseline-Vergleich zum Einsatz.

Der vorhandene Registry-Eintrag `rbf_svm` liefert Varianzfilter, StandardScaler
und RBF-SVC. Varianzfilter und Skalierung werden für jeden Fit frisch auf dessen
Training gelernt. Je Feature-Set werden acht Kombinationen aus
`C ∈ {0.3, 3.0}`, `gamma ∈ {0.001, "scale"}` und
`class_weight ∈ {None, "balanced"}` getestet. `scale` hängt vom
Trainingsfold und der Eingabedimension ab. Die Rangfolge verwendet zuerst die
gepoolte Zahl kritischer OOF-Fehler, bei Gleichstand höheren Macro-F1 und dann
weniger Fehlalarme. Die Candidate-ID entscheidet vollständige Gleichstände.
Seed ist 42. Der RBF-Kernel-Cache umfasst bis zu 768 MiB pro Fit.
Für die tsfresh-CV berechnet der Runner pro Fold und Gamma dieselbe
RBF-Kernelmatrix blockweise und verwendet sie für alle vier Kombinationen aus
`C` und Gewichtung. Die temporäre Trainingsmatrix benötigt bis zu rund
4,6 GB Speicherplatz im Laufordner. Für den Endfit verwenden beide Arme
die direkte RBF-Implementierung der Registry.

Für die Vergleichbarkeit werden die sieben historischen frühen manuellen
CV-Baselines und die sieben historischen tsfresh-Holdout-Baselines verwendet.
Für den frühen manuellen Holdout trainiert der Runner HGB und MLP als zusätzliche
Kontrollen mit den bestehenden Registry-Parametern auf derselben Feature-Matrix.
Die historischen manuellen Holdout-Ergebnisse verwendeten den bisherigen
Oberflächenanker. Der alte punktweise tsfresh-Lauf verwendete 701 Merkmale
und andere Splits.

Die Auswertung verwendet `four_label_v2`. Ein kritischer Fehler ist eine wahre
1 oder 2 mit Vorhersage 3. Der Nenner für die Gesamtrate ist die Zahl wahrer
1/2. Die Klassenraten nutzen die jeweilige wahre Klasse als Nenner. Fehlalarme
sind 3→1/2, bezogen auf alle wahren 3. Die Ausgaben enthalten auch Accuracy,
Balanced Accuracy, Macro-/Weighted-F1, Recall je Klasse, Konfusionsmatrizen
sowie Ergebnisse je Körper und Messreihe.

## Reproduktion

Aus dem Projektstamm mit dem vorhandenen HeRAI-Interpreter und den
unveränderten lokalen Feature-Caches aus den beiden früheren Versuchen
ausführen. [Manueller Anker-Cache](experimental_anchor_group_cv.md) ·
[tsfresh-Caches](specimen_holdout.md)

Für die frühen manuellen Holdout-Features
müssen die archivierten Originalsignale an den Pfaden aus `raw_inputs.csv`
erreichbar sein. Die Befehle benötigen jeweils einen neuen Ausgabeordner.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42
.venv/bin/python -m src.experiments.svm_group_cv manual52_early --jobs 2 \
  --output-dir results/experiments/svm_group_cv_manual52_repeat
.venv/bin/python -m src.experiments.svm_group_cv tsfresh --jobs 2 \
  --output-dir results/experiments/svm_group_cv_tsfresh_repeat
```

`plan.json` hält Grid, Datenprüfsummen, Seed, Git-Stand, Interpreter und
Bibliotheksversionen fest. `fold_metrics.csv` dokumentiert alle 24 Fits je
Feature-Set und `cv_ranking.csv` die gepoolten Kandidatenergebnisse.
`selected.json`, `final_features.json`, `runtime.json`, die Modelle und
die ID-verknüpften Vorhersagen liegen in den jeweiligen Laufordnern.
Die ursprünglichen lokalen MLflow-Stores enthalten je Feature-Set einen CV-Run
und je Endmodell einen Holdout-Run. Alle sechs Runs sind außerdem in der
Projekt-UI unter `svm-specimen` sichtbar. Die nachträglichen Kopien tragen
`herai.imported=true` und `herai.source_run_id`. Neue Läufe erscheinen
automatisch im selben Experiment der zentralen `mlflow.db`.
Für tsfresh-Modellinferenz auf Rohmerkmalen werden
zusätzlich die archivierten Imputationswerte und die Spaltenreihenfolge der
bestehenden tsfresh-Holdout-Aufbereitung benötigt.

## Ergebnisse mit den frühen manuellen Features

### Gruppen-CV auf denselben Trainingskörpern

| Modell | Accuracy | Balanced Accuracy | Macro-F1 | Kritisch 1/2→3 | Fehlalarme 3→1/2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RBF-SVM, ausgewählt | 75,33 % | 77,08 % | 0,6953 | 653 (8,59 %) | 4.433 (20,08 %) |
| MLP | 86,39 % | 75,69 % | 0,7640 | 1.676 (22,05 %) | 906 (4,10 %) |
| Logistische Regression | 87,25 % | 73,79 % | 0,7593 | 1.732 (22,79 %) | 673 (3,05 %) |
| XGBoost | 88,06 % | 75,34 % | 0,7758 | 1.843 (24,25 %) | 720 (3,26 %) |
| HGB | 88,29 % | 75,72 % | 0,7807 | 1.862 (24,50 %) | 702 (3,18 %) |
| LightGBM | 88,17 % | 75,22 % | 0,7752 | 1.873 (24,64 %) | 698 (3,16 %) |
| Random Forest | 88,26 % | 73,54 % | 0,7646 | 1.985 (26,12 %) | 517 (2,34 %) |
| Extra Trees | 87,84 % | 72,50 % | 0,7542 | 2.037 (26,80 %) | 493 (2,23 %) |

Die SVM mit `C=0.3`, `gamma=0.001` und balancierten Klassengewichten
hat 1.023 kritische Fehler weniger als das historische MLP. Das geht mit
3.527 zusätzlichen Fehlalarmen und einem um 0,0688 niedrigeren Macro-F1 einher.
Die ungewichtete SVM mit `C=3` und `gamma="scale"` erreicht im selben Raster
einen Macro-F1 von 0,7753 bei 25,01 % kritischen Fehlern und 2,16 % Fehlalarmen.
Diese Kombination zeigt, wie stark die Wahl der Klassengewichte und Kernelparameter
den Betriebspunkt verschiebt. Nur der nach dem vorab festgelegten Kriterium
ausgewählte Kandidat wurde auf den Holdouts evaluiert.

### Prüfkörper-Holdouts

| Split | Modell | Accuracy | Balanced Accuracy | Macro-F1 | Kritisch 1/2→3 | 1→3 | 2→3 | Fehlalarme |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | RBF-SVM | 74,48 % | 77,69 % | 0,7090 | 160 (8,80 %) | 7,67 % | 10,81 % | 944 (20,69 %) |
| Validation | HGB | 86,54 % | 75,83 % | 0,7734 | 363 (19,97 %) | 7,49 % | 42,01 % | 242 (5,30 %) |
| Validation | MLP | 87,48 % | 78,48 % | 0,7963 | 321 (17,66 %) | 6,80 % | 36,83 % | 208 (4,56 %) |
| Test | RBF-SVM | 79,48 % | 82,74 % | 0,7544 | 39 (1,72 %) | 1,25 % | 2,65 % | 1.468 (21,72 %) |
| Test | HGB | 90,57 % | 80,12 % | 0,8198 | 413 (18,16 %) | 6,52 % | 41,53 % | 174 (2,57 %) |
| Test | MLP | 90,46 % | 79,29 % | 0,8124 | 408 (17,94 %) | 7,11 % | 39,68 % | 122 (1,81 %) |

Die Validierungsverbesserung kommt aus Klasse 2. Die SVM gibt dort 71 von
657 Punkten fälschlich als Klasse 3 frei, das MLP 242. Bei Klasse 1 produziert
sie auf diesen fünf Körpern 89 kritische Fehler gegenüber 79 beim MLP. Für die
sieben Testkörper sinken beide klassenbezogenen Fehlerraten. Der Klasse-2-Recall
im Test liegt bei 90,48 % gegenüber 47,35 % beim HGB und 42,33 % beim MLP. Das MLP hat beim
unveränderten 200-Iterationen-Limit eine Konvergenzwarnung ausgegeben.

Die gepoolte manuelle Testfehlerrate verdeckt Körperunterschiede.
Auf `messreihe_3_pk_02` gibt die SVM alle fünf Klasse-2-Punkte fälschlich frei,
bei null Punkten der Klasse 1. Auf `messreihe_1_pk_06` und
`messreihe_2_pk_01` meldet sie 92,97 % bzw. 91,99 % der wahren Klasse-3-Punkte
als Klasse 1 oder 2.

## Vergleich beider SVMs in der Gruppen-CV

| Feature-Set | Balanced Accuracy | Macro-F1 | Kritisch 1/2→3 | 1→3 | 2→3 | Fehlalarme |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 52 manuelle, früher Anker | 77,08 % | 0,6953 | 653 (8,59 %) | 7,57 % | 10,94 % | 4.433 (20,08 %) |
| tsfresh | 73,48 % | 0,6753 | 880 (11,58 %) | 7,06 % | 21,97 % | 3.674 (16,64 %) |

Auch tsfresh wählt `C=0.3`, `gamma=0.001` und balancierte Klassengewichte.
Die foldweise Selektion behält 700, 704 und 684 Features. Die frühen manuellen
Merkmale halbieren gegenüber tsfresh die kritische Klasse-2-Rate ungefähr,
bei 759 zusätzlichen Fehlalarmen und 27 zusätzlichen kritischen Klasse-1-Fehlern.
Die beiden SVM-Zeilen zeigen die Auswahl-CV. Die dort verwendeten OOF-Punkte
sind jeweils aus dem Training des zugehörigen Fits ausgeschlossen, gehen aber
in die Kandidatenwahl ein. Die Holdouts liefern die Leistungsprüfung für die
ausgewählten Konfigurationen. Die historischen tsfresh-Referenzwerte für dieses
Feature-Set stammen aus den Holdouts.

In beiden Rastern senkt jede balancierte Kombination die kritische Fehlerrate
gegenüber jeder ungewichteten Kombination. Die Fehlalarmrate steigt jeweils.
Der Vergleich mit den fest konfigurierten Baselines bewertet diese konkreten
Betriebspunkte. Die getesteten Baselines hatten keine entsprechende Suche nach
Klassengewichten oder Fehlalarmgrenzen.

## tsfresh auf denselben Prüfkörper-Holdouts wie die Baselines

Alle Modelle in der Tabelle verwenden im Endfit dieselbe Imputation, dieselben
708 ausgewählten Merkmale und dieselben Prüfkörper-Splits. Die Baselines
stammen aus dem abgeschlossenen, versionierten Vergleich. [Baseline-Metriken](evidence/specimen_holdout_v1/baseline/results.csv)

| Modell | Validation Macro-F1 | Validation kritisch | Validation Fehlalarme | Test Macro-F1 | Test kritisch | Test Fehlalarme |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RBF-SVM | 0,7066 | 164 (9,02 %) | 493 (10,80 %) | 0,7377 | 312 (13,72 %) | 995 (14,72 %) |
| Logistische Regression | 0,7270 | 529 (29,10 %) | 108 (2,37 %) | 0,7369 | 507 (22,30 %) | 215 (3,18 %) |
| MLP | 0,7247 | 395 (21,73 %) | 281 (6,16 %) | 0,7197 | 522 (22,96 %) | 353 (5,22 %) |
| LightGBM | 0,6987 | 516 (28,38 %) | 87 (1,91 %) | 0,7144 | 717 (31,53 %) | 87 (1,29 %) |
| HGB | 0,6913 | 542 (29,81 %) | 50 (1,10 %) | 0,7025 | 738 (32,45 %) | 77 (1,14 %) |
| XGBoost | 0,7108 | 522 (28,71 %) | 109 (2,39 %) | 0,6979 | 739 (32,50 %) | 77 (1,14 %) |
| Random Forest | 0,6410 | 494 (27,17 %) | 63 (1,38 %) | 0,6847 | 839 (36,90 %) | 46 (0,68 %) |
| Extra Trees | 0,6240 | 507 (27,89 %) | 63 (1,38 %) | 0,6756 | 868 (38,17 %) | 39 (0,58 %) |

Die tsfresh-SVM erreicht in Validation 76,92 % Accuracy, 75,22 % Balanced
Accuracy und 0,7823 Weighted-F1. Im Test sind es 80,51 %, 78,59 % und 0,8257.
Gegenüber der logistischen Regression sinkt im Test die Zahl kritischer Fehler
um 195, bei nahezu gleichem Macro-F1 und 780 zusätzlichen Fehlalarmen.
Von den kritischen Testfehlern entfallen 206 auf Klasse 1 (13,57 %) und 106 auf
Klasse 2 (14,02 %). Der Klasse-2-Recall beträgt 76,06 %. Auf
`messreihe_2_pk_14` kommen 209 der 312 kritischen Testfehler zusammen
(209/217 auf diesem Körper, gegenüber 21/217 mit den frühen manuellen Features).

Auf denselben fünf Validierungskörpern liegen die beiden SVMs bei 160 bzw. 164
kritischen Fehlern. tsfresh hat dort 451 Fehlalarme weniger, aber prognostiziert
bei 607 wahren Klasse-3-Punkten eine Wiederholungsmessung (Klasse 0), gegenüber
406 bei der manuellen SVM. Auf den sieben Testkörpern hat die frühe manuelle SVM
273 kritische Fehler weniger und einen um 0,0167 höheren Macro-F1 als die
tsfresh-SVM. Sie erzeugt 473 zusätzliche Fehlalarme. Beide geben auf
`messreihe_3_pk_02` alle fünf Fälle der Klasse 2 als Klasse 3 frei.

## Laufzeit und Entscheidung

| SVM-Feature-Set | Gruppen-CV, alle 24 Fits | Finale Feature-Bereitstellung | Endfit | Vorhersage Validation / Test | Gesamtlauf | Support-Vektoren |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manuell, früh | 16,0 min | 75,9 s | 66,0 s | 21,3 s / 25,9 s | 21,4 min | 19.186 |
| tsfresh | 15,9 min | 1,5 s | 208,0 s | 58,2 s / 77,7 s | 24,3 min | 16.218 |

Gemessen auf macOS/ARM64 mit Python 3.12.5 und scikit-learn 1.9.1, einem
Thread je Fit und bis zu zwei parallelen Kandidaten. Die Läufe überlappten
zeitweise. Die Gesamtlaufzeit enthält Auswertung, MLflow, Laden der Modelle zur
Vorhersageprüfung und im manuellen Arm die Kontrollmodelle. Die manuelle
Bereitstellung enthält die Extraktion für 17.985 Holdout-Signale mit null
Extraktionsausfällen. tsfresh verwendet bereits vorhandene Extraktionsartefakte.
Die historischen tsfresh-Endfits lagen bei 10,8–80,2 s, darunter 22,8 s
für logistische Regression. Im aktuellen frühen manuellen Holdout brauchte
HGB 13,7 s zum Fitten und 0,58 s für die Testvorhersagen, das MLP 43,5 s und
0,014 s. Die Zeiten beschreiben die jeweiligen Läufe mit ihrer Systemlast.
Ein tsfresh-Vorversuch mit direktem Kernel pro CV-Kandidat wurde wegen der
Laufzeit beendet. Die abgeschlossene tsfresh-CV verwendete gemeinsame
Kernelwerte bei unverändertem Versuchsplan.

**Die SVM mit frühen manuellen Features sollte als Kandidat für die
weitere Entwicklung berücksichtigt werden.** Sie reduziert die Gesamtzahl
kritischer Fehler in Gruppen-CV, Validation und Test deutlich und erzielt
im manuellen Test die höchste Balanced Accuracy der drei verglichenen Modelle.
Dafür sinkt Macro-F1 gegenüber HGB und MLP, und etwa jeder fünfte wahre
Klasse-3-Testpunkt führt zu einem Fehlalarm. tsfresh bleibt nach kritischen
Fehlern gegenüber den eigenen Baselines konkurrenzfähig, hat aber einen rund
dreimal so langen Endfit und im Test mehr kritische Fehler als die manuelle
SVM. Für die Weiterentwicklung ist der Vergleich bei gemeinsamem
Fehlalarmbudget entscheidend. Die sieben Testkörper und die starke
Körperabhängigkeit begrenzen Aussagen über den Einsatz auf neuen Prüfkörpern.

## Artefakte und Abschlussprüfung

- `results/experiments/svm_group_cv_manual52_v1/` und
  `results/experiments/svm_group_cv_tsfresh_v2/` enthalten MLflow-Stores,
  Konfigurationen, vollständige Vorhersagen, Körper-/Quellenberichte und Modelle.
- `results/experiments/svm_group_cv_comparison_v1/` enthält die gemeinsamen
  CSV-Tabellen für CV, Holdouts und Laufzeiten. Die historischen Vorhersagen
  wurden per ID, Label und Prüfkörper gegen die eingefrorenen Kohorten geprüft
  und mit den bestehenden Metrikfunktionen nachgerechnet.
- Beide Runner haben die Holdout-Vorhersagen nach erneutem Laden der Modelle
  exakt bestätigt. 186 pytest-Tests und Ruff sind bestanden. Die Tests prüfen
  unter anderem Fold-Isolation, Auswahlpriorität und Vorhersagegleichheit
  des gemeinsamen Kernels mit der direkten RBF-SVM auf synthetischen Daten.
  Eine zweite vollständige Wiederholung der realen Trainings wurde nicht
  durchgeführt.
