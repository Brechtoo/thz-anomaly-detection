# Nested-Optimierung tabellarischer Nicht-SVM-Modelle

Der [Modellvergleich](../results/experiments/specimen_tabular_nested_hpo_v1/robustness/vergleich.md)
zeigt beide Entscheidungsarten je Feature-Set, die Fehlerbeiträge einzelner
Prüfkörper und gepaarte Unsicherheitsintervalle. Die Kopfzeile weist den
verifizierten Auswertungsumfang aus.

Die [Analyse konservativer Freigabeschwellen](nested_freigabeschwellen.md)
untersucht das 2-%-Ziel, den zusätzlichen Prüfaufwand und die gemeinsamen
Fehler von SVM und Ensemblepartnern anhand derselben Checkpoints.

Der Lauf ist abgeschlossen. Alle 15 Kombinationen liegen für alle drei äußeren
Folds vor. Die [Ergebnistabellen](tabular_nested_ergebnisse.md) enthalten den
vollständigen Vergleich. Zwei Betriebspunkte unterschreiten die kritische
Fehlerrate der SVM bei gleichem Feature-Set und gleicher Entscheidungsart,
beide mit manuellen Features und TSFEL.

- Bei adaptiver Freigabe spart das MLP gepoolt 16 kritische Fehler ein.
  Ohne `messreihe_1_pk_09` entstehen 146 zusätzliche kritische Fehler.
  HGB, LightGBM, logistische Regression und XGBoost liegen auch an ihrem
  jeweils sichersten adaptiven Betriebspunkt über der zugehörigen SVM.
- Bei regulärer Entscheidung senkt die logistische Regression die kritische
  Rate von 9,82 % auf 8,16 %, die Fehlalarmrate von 18,41 % auf 15,05 %
  und erhöht Macro-F1 von 0,700 auf 0,748. Der kritische Vorsprung bleibt
  beim Ausschluss jedes einzelnen Körpers bestehen (−2,07 bis −0,52
  Prozentpunkte). Er konzentriert sich dennoch auf zwei Körper der ersten
  Messreihe. PK 09 und PK 04 tragen 89 und 56 der eingesparten Fehler bei,
  auf den übrigen Körpern entstehen zusammen 19 zusätzliche. Das gepaarte
  95-%-Prüfkörperintervall der Gesamtdifferenz reicht von −4,88 bis +0,61
  Prozentpunkten.

Die Vorauswahl verwendet ausschließlich die gepoolten Trainings-OOF-Ergebnisse
der acht tabellarischen Familien des Feature-Grids. Je Familie zählt der beste Betriebspunkt auf den drei
Feature-Sets der Nested-SVM. Die Rangfolge priorisiert kritische Fehler.

| Familie | Feature-Set des besten historischen Punkts | Kritische Rate | Macro-F1 |
| --- | --- | ---: | ---: |
| MLP | Manuell + TSFEL | 18,39 % | 0,7637 |
| Logistische Regression | Manuell + TSFEL + Physik V2 | 20,63 % | 0,7934 |
| HistGradientBoosting | Manuell + TSFEL | 23,00 % | 0,7836 |
| XGBoost | Manuell + TSFEL + Physik V2 | 23,18 % | 0,7907 |
| LightGBM | Manuell + TSFEL + Physik V2 | 23,58 % | 0,7881 |

Random Forest (25,53 %) und Extra Trees (26,22 %) lagen dahinter. Diese
Vorauswahl nutzt Resultate derselben 23 Prüfkörper und gehört daher zur
explorativen Festlegung der Modellfamilien. Bewertet werden die anschließend
je Familie und Feature-Set verschachtelt ausgewählten Konfigurationen.
Ein nachträgliches Ranking der äußeren Ergebnisse ist deskriptiv.

## Suchdesign und Vergleich

- Alle Familien nutzen Manuell, Manuell + TSFEL sowie Manuell + TSFEL +
  Physik V2. Die archivierten SVM-Gruppenzuordnungen werden anhand ihrer
  Prüfsumme geladen. Alle 34.235 Messungen gehören zur Trainingskohorte.
- Pro Modell und Feature-Set stehen fünf feste Parameterprofile mit
  uniformen, balancierten und kostenbasierten Fit-Gewichten zur Wahl, dazu
  die Registry-Referenz. Die drei Kostenprofile entsprechen der bisherigen
  SVM-Suche. Das Regressionsgitter variiert C von 0,003 bis 30.
  Das MLP variiert Tiefe, Breite, L2-Penalty und Lernrate. Die
  Boosting-Modelle variieren Blatt- oder Tiefenlimits, Schrumpfung,
  Mindestblattgröße, Iterationszahl und ausgewählte Regularisierungen.
  Alle Parameter stehen im [Suchraum](../src/experiments/tabular_nested_space.py).
- Die bestehende Nested-SVM-Blockvorbereitung beobachtet beim Fit nur
  zulässige Gruppen. Sie lernt Mediane, Konstanten-/Duplikatfilter und
  Skalierung pro Fit. HGB verwendet eine feste Iterationszahl, damit keine
  punktweise interne Early-Stopping-Partition entsteht.
- Das Verfahren nutzt dieselben absoluten Qualitätsschranken, relativen
  Toleranzen, Sicherheitsgewichte und Schwellen wie die SVM. Die relativen
  Grenzen werden an der inneren Registry-Referenz derselben Familie und
  Feature-Kombination verankert. Die reguläre Entscheidung bleibt `predict`.
  Für die adaptive Entscheidung prüft eine innere Vorauswahl von höchstens
  acht Kandidaten Kalibratoren und Freigaberegeln. Die Kalibratoren verwenden
  Log-Wahrscheinlichkeiten als Eingabe. Für einen inneren Zielkörper stammen
  die Kalibrationsdaten ausschließlich von anderen Körpern aus demselben
  inneren Prüffold. Der äußere Kalibrator lernt aus den zulässigen inneren
  OOF-Vorhersagen. Der äußere Prüffold bewertet die festgelegte Wahl.
- Der gepaarte Vergleich stellt native und adaptive Entscheidungen je
  Feature-Set den entsprechenden archivierten SVM-Entscheidungen gegenüber.
  Die Prüfkörperintervalle verwenden 2.000 gepaarte Ziehungen innerhalb
  der äußeren Folds bei festgehaltenen Vorhersagen. Die bisherige
  Feature-Vorauswahl, die überlappenden äußeren Trainingsmengen und die
  vertretenen Messreihen begrenzen die Übertragbarkeit.

Die 15 Modell-/Feature-Kombinationen sind getrennte Experimente. Die
äußeren Ergebnisse schätzen die jeweilige innere Auswahlprozedur. Wird
anschließend der günstigste Tabellenwert ausgewählt, fließen diese
äußeren Labels in die Wahl der Familie oder des Feature-Sets ein.
Die familienabhängige Registry-Referenz kann außerdem zu unterschiedlichen
relativen Qualitätsschranken führen. Die Ergebnisse vergleichen die
tatsächlich untersuchten Suchräume und Budgets.

Die Fehlerrate von unter 2 % wird im inneren Suchverfahren unter den
zulässigen Kandidaten priorisiert. Der anschließende Sicherheitsscore
berücksichtigt zusätzlich die Rate 1→3, die stabilisierte Rate über die
Prüfkörper und den ungünstigsten inneren Fold. Deshalb werden 1→3 und
2→3 im äußeren Vergleich einzeln ausgewiesen. Die Fehlalarmrate erhält
erst nach Sicherheitsscore und allgemeiner Klassifikationsgüte Gewicht.

## Ausführung

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m src.experiments.tabular_nested_hpo --jobs 4
```

Derselbe Aufruf setzt den Lauf anhand des Plans, atomarer Fit-Checkpoints,
Vorhersageprüfsummen und gespeicherter innerer Entscheidungen fort.
Der Plan umfasst 2.160 innere Fits (5 × 3 × 16 × 3 × 3) und 45
äußere Bewertungen. Je Bewertung entstehen bis zu zwei neue äußere
Fits. Die Datenbank im Laufordner verzeichnet Konfiguration,
Fortschritt und Ergebnisse. Die Reproduktion benötigt die bestehende
Projektumgebung und die ursprünglichen Eingabeartefakte.
Der Laufordner enthält eine prüfsummengleiche Kopie der im Plan
referenzierten Projektquellen.

Der [Laufordner](../results/experiments/specimen_tabular_nested_hpo_v1/)
enthält für jede Familien-/Feature-Kombination die inneren Versuche und
die vor der äußeren Auswertung gespeicherte Auswahl. Im Unterordner
`report/` stehen die gepoolten, fold- und prüfkörperspezifischen
Kennzahlen einschließlich Precision, Recall und F1 je Klasse, die
Konfusionsmatrizen und die direkten Differenzen zur SVM.
`report/status.json` zeigt die Zahl abgeschlossener äußerer Bewertungen.
Fehlerraten und Freigabe werden in Prozent ausgegeben, Macro-F1 und
Balanced Accuracy auf der Skala von 0 bis 1.

Das MLP verwendet ein festes Budget von 200 Iterationen. Bei
Konvergenzwarnungen bewertet das Experiment den bis dahin erreichten
Fit. Die Resultate beziehen sich auf dieses Budget und die geprüften
Parameterprofile.

Beim Wiederaufnehmen am 26. November wurden alle 304 vorhandenen Fits
gegen Prüfsumme, Messungsidentität, Labels und Gruppenzuordnung geprüft.
`resume_20261126.json` enthält ihre ursprünglichen Vorhersageprüfsummen.
XGBoost speichert Log-Scores als Float32. Nach den ersten 48 gültigen
XGBoost-Fits überstieg der Rundungsfehler der kalibrierten Zeilensummen
die festgelegte Toleranz. Die Auswertung erweitert die gespeicherten Werte
vor der Kalibrierung verlustfrei auf Float64. Der ursprüngliche Plan,
die ursprünglichen Fit-Dateien, der archivierte Quelltext sowie die
Prüfsummen beider Codeversionen erlauben eine Prüfung dieser Korrektur.

Nach der vollständigen HPO prüft der Abschlusslauf auch den MLflow-Status,
die registrierten Artefakte sowie die äußeren Vorhersagen. Die Auswertung
der gespeicherten Konfusionsmatrizen lässt sich separat reproduzieren.

```sh
.venv/bin/python -m src.analysis.tabular_nested_robustness
.venv/bin/python -m src.analysis.tabular_nested_summary
```

## Wie ein Prüfkörper das Ergebnis des MLP beeinflusst

Für Manuell + TSFEL ergeben die drei äußeren Folds diese Betriebspunkte.

| Modell und Entscheidung | Kritisch | 1→3 | 2→3 | Fehlalarm % | Macro-F1 | Bal. Acc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SVM regulär | 746 | 478 | 268 | 18,41 | 0,7003 | 0,7730 |
| MLP regulär | 1.149 | 391 | 758 | 7,80 | 0,7580 | 0,7779 |
| SVM adaptiv | 445 | 282 | 163 | 29,94 | 0,6907 | 0,7850 |
| MLP adaptiv | 429 | 144 | 285 | 25,37 | 0,7131 | 0,7951 |

Das adaptive MLP senkt die kritische Rate gegenüber der adaptiven
SVM um 0,21 Prozentpunkte (deskriptives 95-%-Prüfkörperintervall
−5,80 bis +3,32). Auf `messreihe_1_pk_09` entfallen 162 weniger
kritische Fehler. Die übrigen Körper weisen zusammen 146 zusätzliche
Fehler auf. Der gepoolte Vorsprung liefert somit keinen Beleg für
einen stabilen Sicherheitsgewinn über die Prüfkörper.

Auf diesem Körper sinkt 1→3 von 193 auf 27, während 2→3 von zwei
auf sechs steigt. Über die übrigen Körper entstehen beim MLP gegenüber
der SVM 28 zusätzliche Fehler der Art 1→3 und 118 der Art 2→3. Wenn
der Körper aus der Auswertung entfällt, beträgt die kritische
Ratendifferenz +2,05 Prozentpunkte (146 zusätzliche Fehler bei 7.134
kritischen Sollmessungen). Unter den 22 Körpern mit kritischen
Sollklassen verbessert sich das MLP bei sechs, verschlechtert sich bei
15 und erreicht auf einem Körper dieselbe Fehlerzahl.

Auch die Differenzen bei Fehlalarmen (−4,57 Prozentpunkte,
95-%-Intervall −14,77 bis +4,89), Macro-F1 (+0,022, Intervall −0,012
bis +0,058) und Balanced Accuracy (+0,010, Intervall −0,017 bis +0,040)
variieren mit der Zusammensetzung der Prüfkörper. Das absolute
Prüfkörperintervall der kritischen MLP-Rate reicht von 2,73 % bis 10,82 %.
