# Freigabe, Fehler und Prüfaufwand auf der Pareto-Front

Die Auswertung umfasst 34.235 äußere OOF-Messungen auf 23 Trainingsprüfkörpern. Von 1.668 untersuchten Verfahren sind 238 Pareto-optimal und ergeben 210 unterschiedliche Kombinationen der vier Kennzahlen. Ein Punkt wird dominiert, wenn ein anderer höchstens ebenso viele kritische Fehler, Fehlalarme und Prüfungen sowie mindestens ebenso viele Freigaben erreicht und mindestens eine Kennzahl verbessert. Kritische Fehler beziehen sich auf 7.600 wahre 1/2, Fehlalarme auf 22.082 wahre 3, Prüfungen und Freigaben auf alle Messungen.

## Welche Betriebspunkte erreichen die Grenzen?

Die Tabelle zeigt die **beobachteten Extreme der äußeren Pareto-Front** bei strikt unterschrittener gepoolter Grenze. Freigabegleichstände werden nach Prüfquote, dann nach Fehlalarmen aufgelöst, Prüfgleichstände nach Fehlalarmen, dann nach Freigabe. Das zuerst genannte Modell liefert die Ausgangsklasse. Eine Freigabe braucht die Zustimmung aller Mitglieder. M = manuell, T = TSFEL, P = Physik V2, LR = logistische Regression. Alle Kennzahlen sind Prozentwerte.

| Grenze | Vorrang | Modelle, Features | Kritisch | 1→3 | 2→3 | Fehlalarm | Prüfung | Freigabe |
|---|---|---|---:|---:|---:|---:|---:|---:|
| < 0,5 % | Freigabe | LR + MLP, M+T | 0,355 | 0,472 | 0,087 | 8,97 | 44,12 | 16,23 |
| < 0,5 % | Wenige Prüfungen | SVM + LR, M+T | 0,474 | 0,547 | 0,304 | 15,32 | 38,63 | 15,52 |
| < 1 % | Freigabe | LR + MLP, M+T+P | 0,947 | 0,982 | 0,868 | 12,26 | 33,04 | 24,60 |
| < 1 % | Wenige Prüfungen | SVM + LR, M | 0,961 | 1,133 | 0,564 | 20,12 | 28,91 | 21,33 |
| < 2 % | Beide Kriterien | SVM + MLP + LR, M+T | 1,921 | 1,491 | 2,909 | 15,32 | 19,61 | 34,54 |

Für die fünf Tabellenzeilen gilt der Reihe nach die zusätzliche innere Grenze von 2 % je kritischer Klasse, 1,5 % je Fold, 3 % je Klasse, 3 % gepoolt und ausschließlich die archivierte Mindestschwelle je Modell. Die Prozentwerte bezeichnen das innere Fehlerbudget. Die daraus ermittelte Score-Schwelle variiert je äußerem Fold.

Unter 1 % erzielt die LR allein 22,40 % Freigaben bei 8,97 % Fehlalarmen und 37,95 % Prüfung. Das freigabestärkste Paar gewinnt 2,20 Prozentpunkte Freigabe bei 3,29 Punkten mehr Fehlalarmen. Für MLP, HGB, LightGBM, XGBoost und SVM allein liegen die maximalen Freigabequoten unter derselben Fehlergrenze bei 20,14 %, 17,18 %, 16,93 %, 13,40 % und 8,91 %. Verlangt man **unter 2 % für beide kritischen Klassen**, führt LR + MLP mit M+T+P bei 32,53 % Freigabe, 25,11 % Prüfung und 12,26 % Fehlalarmen (1→3 1,32 %, 2→3 1,74 %).

## Was eine Auswahl ohne äußere Labels erreicht

Die obigen Extreme verwenden äußere Ergebnisse zum Vergleich. Für die folgende Bewertung wurde je Fold bereits **innen** das gesamte Verfahren einschließlich Familie, Features, Ausgangsmodell und Regel gewählt. Zulässig waren Verfahren mit mindestens einer inneren Freigabe und einer inneren kritischen Rate strikt unter der jeweiligen Grenze. Bei den beiden Prioritäten gelten dieselben Gleichstandsregeln wie oben. Daher können für dieselbe Priorität in verschiedenen Folds unterschiedliche Verfahren zum Einsatz kommen.

| Grenze | Innere Priorität | Kritisch | 1→3 | 2→3 | Fehlalarm | Prüfung | Freigabe | Schlechtester äußerer Fold | Körper ≥ Grenze¹ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| < 0,5 % | Freigabe | 0,368 | 0,491 | 0,087 | 9,86 | 45,56 | 14,88 | 1,24 | 3 |
| < 0,5 % | Wenige Prüfungen | 0,276 | 0,359 | 0,087 | 18,67 | 42,13 | 9,60 | 1,00 | 2 |
| < 1 % | Freigabe | 0,500 | 0,604 | 0,261 | 9,92 | 45,58 | 14,22 | 1,82 | 3 |
| < 1 % | Wenige Prüfungen | 0,934 | 0,982 | 0,825 | 18,67 | 36,59 | 15,13 | 3,25 | 3 |
| < 2 % | Freigabe | 1,645 | 1,303 | 2,432 | 8,30 | 35,65 | 25,66 | 4,92 | 7 |
| < 2 % | Wenige Prüfungen | 1,974 | 1,548 | 2,953 | 18,67 | 22,69 | 29,03 | 5,40 | 7 |

¹ Von 22 Körpern mit kritischen Sollmessungen. Die klassenweisen Nenner betragen 5.297 für Klasse 1 auf 13 Körpern und 2.303 für Klasse 2 auf 21 Körpern. Körper ohne Sollmessungen der jeweiligen Klasse haben dort keine definierte Rate.

**Die Fehler konzentrieren sich unterschiedlich je Klasse.** Bei der inneren Freigabepriorität und dem 0,5-%-Ziel entstehen 26 Fehler der Klasse 1. Davon entfallen 14/121 auf `messreihe_3_pk_06` und 11/11 auf `messreihe_2_pk_08`. Beim 2-%-Ziel entfallen 42 der 56 Klasse-2-Fehler auf `messreihe_1_pk_03` (42/319). Bei Priorität auf wenig Prüfung liegt die gepoolte 2-%-Rate knapp unter der Grenze. Lässt man jeweils einen Körper weg, reicht sie von 1,31 % bis 2,26 %.

**Der Anspruch an Robustheit bestimmt die nutzbare Freigabe.** Müssen nachträglich alle drei äußeren Folds unter 0,5 / 1 / 2 % liegen, sinkt die höchste beobachtete Freigabe auf 9,65 / 14,33 / 22,40 %. Gilt die Grenze auf jedem einzelnen vertretenen Körper, sind es bei allen drei Zielen nur 5,57 %. Diesen Punkt erreicht eine manuelle SVM mit null beobachteten kritischen Fehlern, 44,67 % Prüfung und 20,12 % Fehlalarmen. Die 23 Körper erlauben keine belastbare Fehlerobergrenze für neue Prüfkörper.

## Versuchsrahmen

Alle sechs Einzelmodelle, jedes Paar und das Trio aus MLP/SVM/LR sowie der Sechserkonsens wurden auf den drei eingefrorenen Feature-Sets untersucht. Bei Paar und Trio wurde jedes Mitglied als Ausgangsmodell geprüft, beim Sechserkonsens die SVM. Eine zurückgestellte Ausgangsentscheidung 3 geht in explizite Prüfung; die übrigen Ausgangsklassen bleiben erhalten. Dadurch hängt die formale Fehlalarmrate vom Ausgangsmodell ab. Die Prüfquote misst Messungen im zusätzlichen Prüfpfad, ohne Annahme über Dauer oder Ergebnis einer menschlichen Prüfung.

Grundlage sind die [verifizierten Vorhersagen und dieselbe gruppengetrennte Kalibration](nested_freigabeschwellen.md). Die archivierten Mindestschwellen gelten immer. Die zusätzliche gemeinsame Schwelle auf dem kleinsten Mitglieds-Score folgt aus inneren Fehlerbudgets von 0,1 / 0,25 / 0,5 / 0,75 / 1 / 1,5 / 2 / 3 / 5 / 8 %, jeweils gepoolt, je innerem Fold oder je kritischer Klasse. Bei den drei angefragten Budgets wurden auch separate Schwellen je Ensemblemitglied geprüft. Score-Gleichstände bleiben zusammen. Seed 42, eingefrorene Gruppen, keine neuen Basismodell-Fits. Alle 66 mit der vorherigen Auswertung direkt vergleichbaren Verfahren stimmen in den vier gepoolten Kennzahlen überein. Die Front gilt für die untersuchten Verfahren und die Trainingskohorte. Der Testsplit blieb geschlossen.

[Vollständige Pareto-Front](../results/analysis/nested_release_pareto_v1/pareto.csv) · [Innere Auswahl je Fold](../results/analysis/nested_release_pareto_v1/choices.csv) · [Äußere Prüfkörperzählungen](../results/analysis/nested_release_pareto_v1/specimens.parquet) · [Auswertungscode](../src/analysis/nested_release_pareto.py)
