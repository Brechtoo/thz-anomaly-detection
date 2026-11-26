# Nicht räumliche Modelle im Vergleich zur optimierten RBF-SVM

Abschluss mit 15 von 15 vollständig über drei äußere Gruppen-Folds bewerteten Modell-/Feature-Kombinationen. Jede Tabellenzeile umfasst dieselben 34.235 Messungen der 23 Trainingsprüfkörper. 7.600 Messungen tragen das Label 1 oder 2, 22.082 das Label 3. M steht für manuelle Features, T für TSFEL und P für Physik V2.

Die Spalten 1→3 und 2→3 zählen kritische Fehler. Kritisch % bezieht sich auf alle Messungen der Klassen 1 und 2. Fehlalarm % zählt Entscheidungen 1 oder 2 unter allen wahren 3. BA bezeichnet Balanced Accuracy. Die gepaarte Differenz Δ vergleicht Modell minus SVM bei gleichem Feature-Set und gleicher Entscheidungsart. Negative Δ bei Fehlerquoten sind günstig.

## Adaptive Freigabe

| Modell | Features | 1→3 | 2→3 | Kritisch % | Fehlalarm % | Macro-F1 | BA | Δ kritisch in Prozentpunkten [95-%-Intervall] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HGB | M | 310 | 405 | 9,41 | 16,07 | 0,750 | 0,808 | 1,47 [-1,57; 3,55] |
| LR | M | 336 | 301 | 8,38 | 18,58 | 0,723 | 0,795 | 0,45 [-1,07; 1,63] |
| LightGBM | M | 270 | 504 | 10,18 | 14,50 | 0,760 | 0,804 | 2,25 [-2,98; 5,79] |
| MLP | M | 281 | 394 | 8,88 | 19,49 | 0,725 | 0,793 | 0,95 [-3,27; 4,22] |
| RBF-SVM | M | 390 | 213 | 7,93 | 21,87 | 0,681 | 0,763 | Referenz |
| XGBoost | M | 289 | 568 | 11,28 | 15,57 | 0,749 | 0,792 | 3,34 [-1,38; 7,94] |
| HGB | M+T | 202 | 406 | 8,00 | 17,92 | 0,744 | 0,810 | 2,14 [-2,36; 5,62] |
| LR | M+T | 316 | 300 | 8,11 | 15,23 | 0,740 | 0,808 | 2,25 [0,79; 5,15] |
| LightGBM | M+T | 203 | 392 | 7,83 | 17,10 | 0,750 | 0,807 | 1,97 [-2,52; 5,66] |
| MLP | M+T | 144 | 285 | 5,64 | 25,37 | 0,713 | 0,795 | -0,21 [-5,80; 3,32] |
| RBF-SVM | M+T | 282 | 163 | 5,86 | 29,94 | 0,691 | 0,785 | Referenz |
| XGBoost | M+T | 267 | 381 | 8,53 | 16,89 | 0,753 | 0,817 | 2,67 [-0,42; 6,21] |
| HGB | M+T+P | 276 | 423 | 9,20 | 15,37 | 0,769 | 0,826 | 3,24 [1,21; 5,93] |
| LR | M+T+P | 318 | 303 | 8,17 | 15,12 | 0,753 | 0,825 | 2,21 [0,87; 4,23] |
| LightGBM | M+T+P | 172 | 383 | 7,30 | 17,20 | 0,760 | 0,827 | 1,34 [-2,55; 4,35] |
| MLP | M+T+P | 180 | 326 | 6,66 | 24,40 | 0,728 | 0,801 | 0,70 [-3,63; 3,92] |
| RBF-SVM | M+T+P | 250 | 203 | 5,96 | 33,64 | 0,704 | 0,800 | Referenz |
| XGBoost | M+T+P | 236 | 413 | 8,54 | 17,61 | 0,755 | 0,814 | 2,58 [-0,58; 6,06] |

## Reguläre Entscheidung

| Modell | Features | 1→3 | 2→3 | Kritisch % | Fehlalarm % | Macro-F1 | BA | Δ kritisch in Prozentpunkten [95-%-Intervall] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HGB | M | 405 | 483 | 11,68 | 14,32 | 0,763 | 0,813 | 3,28 [0,46; 7,01] |
| LR | M | 394 | 611 | 13,22 | 10,42 | 0,744 | 0,778 | 4,82 [2,20; 7,81] |
| LightGBM | M | 404 | 493 | 11,80 | 14,97 | 0,762 | 0,812 | 3,39 [0,70; 7,06] |
| MLP | M | 429 | 465 | 11,76 | 13,10 | 0,752 | 0,810 | 3,36 [1,41; 5,60] |
| RBF-SVM | M | 397 | 242 | 8,41 | 20,79 | 0,679 | 0,758 | Referenz |
| XGBoost | M | 406 | 510 | 12,05 | 15,10 | 0,758 | 0,806 | 3,64 [1,07; 7,66] |
| HGB | M+T | 371 | 552 | 12,14 | 13,94 | 0,765 | 0,809 | 2,33 [-2,30; 6,49] |
| LR | M+T | 324 | 296 | 8,16 | 15,05 | 0,748 | 0,818 | -1,66 [-4,88; 0,61] |
| LightGBM | M+T | 371 | 497 | 11,42 | 14,01 | 0,767 | 0,815 | 1,61 [-2,82; 5,71] |
| MLP | M+T | 391 | 758 | 15,12 | 7,80 | 0,758 | 0,778 | 5,30 [-0,66; 10,24] |
| RBF-SVM | M+T | 478 | 268 | 9,82 | 18,41 | 0,700 | 0,773 | Referenz |
| XGBoost | M+T | 365 | 514 | 11,57 | 14,70 | 0,763 | 0,811 | 1,75 [-2,32; 6,25] |
| HGB | M+T+P | 315 | 441 | 9,95 | 14,10 | 0,779 | 0,834 | 1,64 [0,31; 3,32] |
| LR | M+T+P | 339 | 338 | 8,91 | 14,83 | 0,763 | 0,829 | 0,61 [-0,71; 2,33] |
| LightGBM | M+T+P | 337 | 555 | 11,74 | 12,70 | 0,779 | 0,822 | 3,43 [1,23; 6,29] |
| MLP | M+T+P | 476 | 995 | 19,36 | 5,90 | 0,781 | 0,778 | 11,05 [8,08; 15,28] |
| RBF-SVM | M+T+P | 310 | 321 | 8,30 | 15,78 | 0,767 | 0,832 | Referenz |
| XGBoost | M+T+P | 336 | 392 | 9,58 | 16,99 | 0,763 | 0,825 | 1,28 [-0,21; 3,46] |

## Einfluss einzelner Prüfkörper

Je Familie zeigt die Tabelle das Feature-Set mit der niedrigsten beobachteten adaptiven kritischen Rate. Diese Auswahl dient der Beschreibung. Die erste Zählspalte enthält ausschließlich Körper mit kritischen Sollklassen. Die letzte Spalte gibt die verbleibende gepoolte Ratendifferenz an, wenn je ein Körper mit allen seinen Messungen aus der Auswertung entfällt. Ein Vorzeichenwechsel zeigt, dass schon ein einzelner Körper die Richtung des Vergleichs ändern kann.

| Modell | Features | Weniger/gleich/mehr Fehler | Kleinste Fehlerdifferenz | Größte Fehlerdifferenz | Δ kritisch ohne je einen Körper (Prozentpunkte) |
| --- | --- | --- | --- | --- | --- |
| MLP | M+T | 6/1/15 | messreihe_1_pk_09 (-162) | messreihe_1_pk_08 (+48) | -0,96 bis 2,05 |
| LightGBM | M+T+P | 6/5/11 | messreihe_1_pk_09 (-100) | messreihe_1_pk_03 (+55) | 0,65 bis 2,83 |
| HGB | M+T | 3/2/17 | messreihe_1_pk_09 (-119) | messreihe_1_pk_03 (+60) | 1,41 bis 3,95 |
| LR | M+T | 2/6/14 | messreihe_1_pk_09 (-7) | messreihe_1_pk_03 (+69) | 1,40 bis 2,54 |
| XGBoost | M+T | 4/3/15 | messreihe_1_pk_09 (-69) | messreihe_1_pk_03 (+72) | 1,80 bis 3,81 |

## Klassenweise Güte der beschriebenen Betriebspunkte

| Modell | Features | F1 Klasse 0 | F1 Klasse 1 | F1 Klasse 2 | F1 Klasse 3 | Freigaben % |
| --- | --- | --- | --- | --- | --- | --- |
| MLP | M+T | 0,817 | 0,879 | 0,341 | 0,815 | 47,60 |
| LightGBM | M+T+P | 0,869 | 0,864 | 0,434 | 0,873 | 53,70 |
| HGB | M+T | 0,849 | 0,838 | 0,425 | 0,862 | 53,41 |
| LR | M+T | 0,805 | 0,828 | 0,460 | 0,867 | 53,97 |
| XGBoost | M+T | 0,845 | 0,862 | 0,438 | 0,867 | 54,31 |
| RBF-SVM | M | 0,710 | 0,839 | 0,369 | 0,807 | 49,01 |
| RBF-SVM | M+T | 0,786 | 0,880 | 0,325 | 0,773 | 44,00 |
| RBF-SVM | M+T+P | 0,856 | 0,882 | 0,314 | 0,763 | 42,80 |

## Unsicherheit und Interpretation

Die 95-%-Perzentilintervalle verwenden 2.000 gepaarte Bootstrap-Ziehungen ganzer Prüfkörper innerhalb der äußeren Folds bei festen OOF-Vorhersagen. Der gepoolte Schätzer gewichtet Messungen nach ihrem Vorkommen. Prüfkörper mit vielen kritischen Messungen haben entsprechend größeren Einfluss. Die Intervalle beschreiben die Variation der untersuchten Körper. Überlappende Trainingsmengen, Modelltraining auf anderen Kohorten, Vorauswahl der Familien auf historischen OOF-Ergebnissen derselben Körper und Mehrfachvergleiche begrenzen die Inferenz.

## Innere Auswahl und äußere Überprüfung

Jede Familie umfasst neun Entscheidungen (drei Feature-Sets × drei äußere Folds). Die innere Zielerfüllung bezieht sich auf die zur Auswahl herangezogenen OOF-Entscheidungen. Der äußere Fold prüft den gewählten Betriebspunkt an anderen Körpern. Die neun Bewertungen einer Familie verwenden dieselben drei äußeren Gruppen wiederholt.

| Modell | Innere Qualitätsschranken erfüllt | Kritisch < 2 % innen | Kritisch < 2 % außen | Adaptive Regel weicht von predict ab |
| --- | --- | --- | --- | --- |
| HGB | 9 | 0 | 0 | 9 |
| LightGBM | 9 | 0 | 0 | 9 |
| LR | 9 | 0 | 0 | 8 |
| MLP | 9 | 0 | 1 | 9 |
| XGBoost | 9 | 0 | 0 | 7 |

## Folgerung

0 von 15 adaptiven Varianten erreichen gepoolt eine kritische Fehlerrate unter 2 %. Den niedrigsten beobachteten Wert hat MLP (M+T) mit 429 kritischen Fehlern (5,64 %), 25,37 % Fehlalarmen und einem Macro-F1 von 0,713. Vor einer Freigabeentscheidung sollten Modellfamilie, Feature-Set, Auswahlregel und Akzeptanzgrenzen für beide kritischen Klassen festgelegt und an unabhängigen Prüfkörpern überprüft werden.

Alle gruppierten Kennzahlen, Precision, Recall und F1 je Klasse sowie die harten Entscheidungen der adaptiv gewählten Fits stehen in [decision_metrics.csv](../results/experiments/specimen_tabular_nested_hpo_v1/report/decision_metrics.csv). Die Unsicherheit von Fehlalarmrate, Macro-F1 und BA sowie die absoluten Intervalle stehen in [paired_intervals.csv](../results/experiments/specimen_tabular_nested_hpo_v1/robustness/paired_intervals.csv). Alle einzelnen Ausschlüsse und Fehlerbeiträge stehen in [leave_one_specimen_out.csv](../results/experiments/specimen_tabular_nested_hpo_v1/robustness/leave_one_specimen_out.csv).
