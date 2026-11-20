# Frühe manuelle Merkmale und tsfresh im Prüfkörpervergleich

## Versuchsaufbau

`specimen_manual52_early_tsfresh_fused_v1` verbindet die 52 manuellen
`early_legacy_windows`-Merkmale mit dem referenzfreien tsfresh-Feature-Set.
Die frühen manuellen Trainingswerte stammen aus dem Anker-Cache. Die
Holdout-Werte stammen aus der SVM-Extraktion mit derselben Ankerheuristik und
denselben Zeitfenstern. Feature- und Split-Manifeste sowie die geordnete
ID-Liste sichern die Zuordnung. Das Parquet speichert die Werte für den
Endfit mit `instance_id`, `label` und den Namensräumen `manual__` und
`tsfresh__`. Es enthält **52.220 Zeilen und 759 Features**. Die ursprüngliche
708er-tsfresh-Tabelle wurde mit den 52 manuellen Features verbunden. Die auf
dem Training exakt identische Spalte `y__mean_abs_change` wurde entfernt.
Die tsfresh-Selektion hatte konstante Spalten bereits ausgeschlossen.

Das Protokoll verwendet die bestehenden 23/5/7 Körper für Training,
Validation und Test sowie die drei eingefrorenen Group-CV-Folds innerhalb
des Trainings. Die vier Zielklassen und die Fehlerdefinitionen
`four_label_v2` bleiben erhalten. Kritisch ist wahre 1/2 → 3, ein Fehlalarm
ist wahre 3 → 1/2. Der Nenner ist jeweils die Zahl der wahren Ausgangsklassen.
Das Ranking priorisiert kritische Fehler und bei Gleichstand Fehlalarme.
Validation und Test dienen der Auswertung fester Konfigurationen.

Für jeden CV-Fold wird tsfresh aus allen 777 signalweise extrahierten
`efficient`-Rohmerkmalen mit der bestehenden Imputation und überwachten
Selektion auf den jeweiligen Fold-Trainingskörpern neu aufbereitet. Die
manuellen Mediane stammen aus denselben Fit-Zeilen. Konstanten und exakte
Spaltenduplikate werden ebenfalls nur dort erkannt. Die manuelle Spalte hat
Vorrang, falls eine tsfresh-Spalte dieselben Werte enthält. Der Endfit nutzt
die auf den 23 Trainingskörpern aufbereitete historische tsfresh-Tabelle
und eine separat auf denselben Körpern bestimmte Fusionsmaske. Die
Endfit-Tabelle wird ausschließlich für diesen Fit und seine Holdouts
verwendet. Varianzfilter und StandardScaler aus der Registry fitten jeweils
innerhalb des Trainings.

Der Vergleich umfasst **logistische Regression, Random Forest, HGB,
XGBoost, LightGBM, Extra Trees, MLP und RBF-SVM**. Die sieben
Standardkonfigurationen bleiben gleich. Die RBF-SVM übernimmt den in beiden
vorigen Läufen gewählten Betriebspunkt (`C=0.3`, `gamma=0.001`,
`class_weight="balanced"`). Die Group-CV verwendet dieselbe blockweise
exakte RBF-Kernelberechnung wie der tsfresh-SVM-Vergleich. Der Endfit
verwendet die direkte Registry-Pipeline. Der bisherige Transformer
verarbeitet Rohsignal-Sequenzen. PCA, Isolation Forest und Autoencoder
liefern Anomalie-Scores. Ihre bisherigen Ein- und Ausgaben passen nicht zu
diesem tabellarischen Vierklassenvergleich.

## Wiederholung und Artefakte

Aus dem Projektstamm mit Python 3.12 und den verifizierten lokalen Caches
aus den bisherigen Prüfkörperexperimenten ausführen. Ausgabeordner und
Feature-Datei müssen neue Pfade sein.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=42
.venv/bin/python -m src.experiments.specimen_fusion \
  --output-dir results/experiments/specimen_fusion_repeat \
  --feature-file results/feature_tables/specimen_manual52_early_tsfresh_fused_repeat.parquet
.venv/bin/python -m src.analysis.specimen_feature_comparison \
  --fusion-run results/experiments/specimen_fusion_repeat \
  --output-dir results/experiments/specimen_fusion_comparison_repeat
```

Für die Kernelberechnung in der CV werden temporär bis zu rund 4,6 GB
Speicherplatz benötigt. Die Dateien werden beim Abschluss der
Kernelberechnung entfernt. Das Parquet und sein JSON-Sidecar liegen in
`results/feature_tables/`; die Ergebnisse, Modell-Dateien, Prüfsummen,
Softwareversionen, der Quellstand und die Fusions-/Imputationszustände
liegen im Laufordner unter `results/experiments/specimen_fusion_v1/`.
Der gemeinsame MLflow-Store enthält jeden Fold- und Endfit als Child-Run
unter `thz-specimen-feature-fusion`. Der Parent-Run enthält die
Run-Konfiguration und die gepoolten OOF-Metriken sowie Holdout-Tabellen.
Die Vergleichs-CSV referenziert die bestehenden identischen
Prüfkörper-Vergleiche aus `svm_group_cv_comparison_v1`.

## Vereinheitlichung

Die Modell-Registry bleibt die Quelle für die Klassifikatoren. Fusion,
SVM und Prüfkörper-Baseline teilen sich nun Fit-/Vorhersage-Timing sowie
die gruppierten Berichte mit festen vier Klassen. Für die Fehler je wahrer
Klasse verwenden die Runner eine gemeinsame Funktion. Die verifizierten
Fold- und Feature-Loader werden von Fusion und SVM gemeinsam genutzt.
Die gemeinsamen Berichtsfunktionen verwenden dieselben Implementierungen
für Konfusionsmatrix und Klassifikationsbericht wie die übrigen
überwachten Pfade. Alle aktuellen überwachten Runner mit MLflow-Anbindung
nutzen die lokale Projekt-Datenbank. Historische Ergebnisse, Modelle,
Splits, Rohdaten und reproduktionsrelevante Quellstände bleiben erhalten.

## Gruppen-CV

Gepoolte Vorhersagen für 34.235 Messpunkte. Die Nenner für kritische Fehler
und Fehlalarme sind 7.600 und 22.082.

| Fusion, Modell | Accuracy | Balanced Accuracy | Macro-F1 | Kritisch 1/2→3 | Fehlalarme 3→1/2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RBF-SVM | 79,11 % | 78,38 % | 0,7208 | 776 (10,21 %) | 3.167 (14,34 %) |
| Logistische Regression | 84,64 % | 74,04 % | 0,7398 | 1.542 (20,29 %) | 1.380 (6,25 %) |
| MLP | 83,75 % | 73,06 % | 0,7284 | 1.620 (21,32 %) | 1.076 (4,87 %) |
| HGB | 88,38 % | 74,53 % | 0,7707 | 1.825 (24,01 %) | 515 (2,33 %) |
| XGBoost | 88,32 % | 74,37 % | 0,7698 | 1.839 (24,20 %) | 503 (2,28 %) |
| LightGBM | 88,28 % | 74,09 % | 0,7672 | 1.900 (25,00 %) | 481 (2,18 %) |
| Random Forest | 86,75 % | 70,16 % | 0,7242 | 2.014 (26,50 %) | 406 (1,84 %) |
| Extra Trees | 86,39 % | 69,34 % | 0,7100 | 2.021 (26,59 %) | 337 (1,53 %) |

In den drei Folds verbleiben 751, 755 und 735 Merkmale. Die Fusion-SVM
erzeugt 123 kritische Fehler mehr als die frühe manuelle SVM (653), bei
1.266 weniger Fehlalarmen. Gegenüber der tsfresh-SVM (880) sinken
kritische Fehler um 104 und Fehlalarme um 507. Die historischen SVM-OOF-Werte
gingen in die Auswahl der Konfiguration ein. Bei den fest konfigurierten
Registry-Baselines fallen die Unterschiede je Modell verschieden aus.
Die logistische Regression senkt gegenüber den frühen manuellen Merkmalen
die kritischen Fehler von 1.732 auf 1.542, bei 707 zusätzlichen Fehlalarmen.
HGB erreicht auf dem kombinierten Set den höchsten Macro-F1 der acht
Modelle. Er liegt bei 0,7707 gegenüber 0,7807 für HGB mit frühen manuellen
Merkmalen.

## Prüfkörper-Holdouts

Es gibt 1.818 kritische Fälle in Validation und 2.274 im Test. Für die
Fehlalarmrate sind die Nenner 4.563 und 6.758. Accuracy, Balanced Accuracy,
Weighted-F1, die Aufteilung nach Klasse 1 und 2, Konfusionsmatrizen sowie
Körper- und Messreihenberichte liegen in den Laufartefakten.

| Fusion, Modell | Validation Macro-F1 | Validation kritisch | Validation Fehlalarme | Test Macro-F1 | Test kritisch | Test Fehlalarme |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RBF-SVM | 0,7606 | 164 (9,02 %) | 578 (12,67 %) | 0,7734 | 148 (6,51 %) | 1.063 (15,73 %) |
| MLP | 0,7634 | 362 (19,91 %) | 212 (4,65 %) | 0,7600 | 352 (15,48 %) | 348 (5,15 %) |
| Logistische Regression | 0,7442 | 495 (27,23 %) | 89 (1,95 %) | 0,7849 | 406 (17,85 %) | 178 (2,63 %) |
| HGB | 0,7769 | 487 (26,79 %) | 77 (1,69 %) | 0,7912 | 485 (21,33 %) | 114 (1,69 %) |
| XGBoost | 0,7859 | 479 (26,35 %) | 67 (1,47 %) | 0,7910 | 500 (21,99 %) | 130 (1,92 %) |
| LightGBM | 0,7777 | 484 (26,62 %) | 65 (1,42 %) | 0,7935 | 505 (22,21 %) | 98 (1,45 %) |
| Random Forest | 0,7475 | 456 (25,08 %) | 86 (1,88 %) | 0,7652 | 566 (24,89 %) | 90 (1,33 %) |
| Extra Trees | 0,6883 | 429 (23,60 %) | 58 (1,27 %) | 0,7444 | 677 (29,77 %) | 55 (0,81 %) |

Bei allen sieben Standardmodellen sinken die kritischen Fehler gegenüber
reinem tsfresh sowohl in Validation als auch im Test. Die logistische
Regression und das MLP senken dabei auf beiden Splits auch die Fehlalarme.
Gegenüber dem frühen manuellen Set ist das Bild gemischt. Für MLP steigen
die kritischen Fehler in Validation von 321 auf 362 und fallen im Test von
408 auf 352. Bei HGB steigen sie in beiden Splits von 363 auf 487 und von
413 auf 485. Gepaarte Vergleiche berücksichtigen nur Modelle mit tatsächlich
vorhandenen Ergebnissen unter denselben Kohorten.

### Drei Feature-Sets mit RBF-SVM

| Feature-Set | Validation kritisch | Validation Fehlalarme | Test kritisch | Test Fehlalarme | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 52 manuell, früh | 160 (8,80 %) | 944 (20,69 %) | 39 (1,72 %) | 1.468 (21,72 %) | 0,7544 |
| Fusion, 759 | 164 (9,02 %) | 578 (12,67 %) | 148 (6,51 %) | 1.063 (15,73 %) | 0,7734 |
| tsfresh, 708 | 164 (9,02 %) | 493 (10,80 %) | 312 (13,72 %) | 995 (14,72 %) | 0,7377 |

Die Fusion ist für die tabellarischen Baselines eine interessante Alternative
zu reinem tsfresh. Die frühe manuelle SVM hat bei strikter Priorität für
kritische Fehler weiterhin die niedrigste Quote in CV, Validation und Test.
Die Fusion-SVM erzielt den höchsten Test-Macro-F1 der drei SVMs und
vermeidet gegenüber der manuellen SVM 405 Test-Fehlalarme. Gleichzeitig
kommen 109 kritische Fehler hinzu. Gegenüber tsfresh halbiert die
Fusion-SVM die kritischen Testfehler ungefähr, bei 68 zusätzlichen
Fehlalarmen. In Validation sind die kritischen Fehler der beiden gleich,
bei 85 zusätzlichen Fehlalarmen mit Fusion.

Auf `messreihe_2_pk_14` entfallen 102 der 148 kritischen Fusion-SVM-Testfehler
(102/217 gegenüber 21/217 manuell und 209/217 tsfresh). Auf
`messreihe_3_pk_02` gibt auch die Fusion alle fünf Klasse-2-Punkte fälschlich
frei. Die sieben Testkörper und die Körperabhängigkeit begrenzen die
Übertragbarkeit dieser Ergebnisse.

## Laufzeiten und Prüfung

Der Gesamtlauf dauerte **41,0 min**, davon **25,0 min** für die drei
CV-Folds mit Vorbereitung, Training, Evaluation und Logging. Die
bereitstehenden Caches ermöglichten die Endfit-Feature-Tabelle in 11,9 s.
Endfits dauerten zwischen 16,3 s (Extra Trees) und 237,7 s (RBF-SVM).
Die Fusion-SVM brauchte 87,3 s für die Testvorhersagen, gegenüber 25,9 s
für die frühe manuelle SVM, 77,7 s für tsfresh und 0,08 s für das Fusion-MLP.
Die Zeiten beziehen sich auf die jeweiligen Läufe auf macOS/ARM64 bei
einem Thread je Fit.

Alle **32 Fits und der Parent-Run** sind in MLflow `FINISHED`. Für jeden
Endfit wurde die Modell-Datei erneut geladen und auf 64 gleichmäßig
über jeden Holdout verteilten Punkten geprüft. Ein unabhängiger Check
rechnet die Metriken für 48 Vorhersagedateien nach, prüft die Kohorten,
Trainings-Imputationswerte und die Prüfsummen aller acht Modell-Artefakte
im gemeinsamen MLflow-Store. Eine zweite vollständige Trainingswiederholung
wurde nicht durchgeführt. **191 Tests und Ruff bestanden.**
