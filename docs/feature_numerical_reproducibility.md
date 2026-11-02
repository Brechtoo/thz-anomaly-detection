# Numerische Reproduzierbarkeit der Feature-Pipeline

**Nachfolgende wissenschaftliche Prüfung:** Der Katalog wurde inzwischen ab
Rohmessungen/Annotationen byteidentisch rekonstruiert. Zugleich wurde eine
Überlappung aller drei Messreihen zwischen den bestehenden Splits nachgewiesen.
Die technische Reproduzierbarkeit bleibt bestehen; Aussagen über unabhängige
neue Scans benötigen eine gesonderte gruppenunabhängige Evaluation. Siehe
[Katalog- und Leakage-Bericht](catalog_reconstruction_and_leakage.md).

## Abschließendes Urteil

Die Pipeline ist für den geprüften THz-Datenstand unter der unten definierten
numerischen Toleranz **numerisch reproduzierbar**, aber nicht byteidentisch.
Das bereits trainierte Referenz-HGB liefert auf der neu extrahierten Tabelle
alle 8.511 ursprünglichen Validierungsvorhersagen und Klassenwahrscheinlichkeiten
exakt unverändert. Es erfolgten weder Training noch Optimierung oder erneute
Feature-Extraktion in dieser abschließenden Analyse.

## Daten und numerische Unterschiede

Referenz: `results/feature_tables/feature_table.csv`, SHA256
`03d439403489d319feb7299847b159d9bd96e9398f72fd1d19bd64cb84df97ed`.
Neuaufbau: `results/reproducibility_checks/20261102T210010Z_4ec464c0/feature_rebuild/feature_table.csv`, SHA256
`a9ad6f850c1586761b97e08ecb02677fb2c84bc753f868b2b55832703e149987`.

Schema, Datentypen, Reihenfolge, IDs, Original-Labels und übrige Metadaten sind
identisch. Bei 53.192 Zeilen und 52 Features unterscheiden sich 1.381.899 von
2.765.984 numerischen Zellen in 47 Features. Fünf Positions-/Distanzfeatures
sind exakt identisch. NaN- und Endlichkeitsmasken stimmen ebenfalls überein.

- Maximale absolute Differenz: **2.7966962079517543e-11** bei
  `second_peak_max_width`, ID `messreihe_2_x-62_y88`.
- Maximale relative Differenz: **1.1104164519752798e-10** bei
  `second_peak_max_prominence`, ID `messreihe_2_x-1_y-57`.
  Referenzwert: `3.999303225921125e-06`, Neuaufbau: `3.9993032254770355e-06`.
  Die absolute Differenz beträgt hier nur `4.440892098500626e-16`.

Die relative Differenz ist `abs(neu - referenz) / abs(referenz)`. Bei zwei
Nullwerten gilt sie als null, bei Referenz null und Neuaufbau ungleich null
wäre sie unendlich. Solche Null-zu-Nichtnull-Fälle treten nicht auf.
Die jeweiligen maximalen absoluten und relativen Differenzen pro Feature
müssen nicht dieselbe Messung betreffen.

## Begründete Toleranz

Für diesen Datenstand wird der folgende gemischte Vergleich dokumentiert:

```python
np.isclose(neu, referenz, rtol=1e-11, atol=1e-13, equal_nan=True)
# abs(neu - referenz) <= 1e-13 + 1e-11 * abs(referenz)
```

Zusätzlich müssen Schema, ID-/Zeilenreihenfolge, Labels, Metadaten sowie
NaN-/Endlichkeitsmasken exakt übereinstimmen. Die Toleranz gilt ausschließlich
für numerische Featurewerte, niemals für Labels, IDs oder Modellvorhersagen.

Begründung:

- Eine rein relative Toleranz ist bei sehr kleinen Prominenzen/Korrelationen
  ungeeignet: Der größte relative Unterschied entspricht absolut weniger als
  `5e-16`. Der kleine Absolutanteil `1e-13` berücksichtigt diese Fälle.
- Der relative Anteil `1e-11` berücksichtigt die etwas empfindlichere
  Peak-Breiteninterpolation. Deren beobachtete maximale relative Abweichung
  liegt bei rund `5.08e-12`.
- Alle Zellen erfüllen den Vergleich. Der größte Quotient aus beobachtetem
  Fehler und erlaubtem Toleranzbudget beträgt **0.4991279094577298**, also
  rund 50 %. Der Absolutanteil ist dabei hundertmal strenger als der zuvor
  zusätzlich diagnostisch geprüfte Wert `1e-11`.
- Das eingefrorene HGB bleibt auf sämtlichen Validierungsfällen einschließlich
  Wahrscheinlichkeiten exakt unverändert.

Das ist eine nach Untersuchung fachlich akzeptierte, empirisch validierte
Toleranz. Die frühere Prüfung mit `rtol=atol=1e-12` hatte vier Ausreißer; dieses
historische Prüfergebnis bleibt bestehen. Die neue Toleranz ist kein Beleg für
Bytegleichheit oder eine universelle Grenze für beliebige neue Messungen.
Sie wurde nicht automatisch in den strikten Datenhashschutz des Baseline-
Trainings übernommen. Die ursprüngliche Referenz bleibt eingefroren.

## Modellprüfung ohne Training

Das Modell wurde aus dem Artefakt des Referenz-Runs
`a5b8cf67d02d4214bb8e36ea946e839a` geladen:

```text
mlruns/1/models/m-44665dd6c857481c9cbc8eacc720025c/artifacts/model.pkl
SHA256: 75e30f9d836cb7807fe8d3cf80d3bb4c119d0ea86802e607d65555f87d39411c
```

Beide Tabellen wurden über `restore_splits` mit den gespeicherten
`configs/hgb_baseline/splits/`-IDs und ursprünglichen Labels aufgeteilt.
Nur die geordnete Validierungsmenge wurde ausgewertet; kein Fit und keine
Testmengenauswertung wurden durchgeführt.

| Prüfung | Ergebnis |
| --- | --- |
| Validierungsfälle | 8.511 |
| Geänderte Klassenentscheidungen | 0 |
| Referenzmodell auf Referenztabelle gegen historische Vorhersagen | exakt identisch |
| Referenzmodell auf neuer Tabelle gegen historische Vorhersagen | exakt identisch |
| Klassenwahrscheinlichkeiten | exakt identisch, Maximaldifferenz 0 |
| Accuracy | beide 0.8883797438608859 |
| Balanced Accuracy | beide 0.8876300273012376 |
| Macro-F1 | beide 0.8395618142421619 |
| Weighted-F1 | beide 0.8975994687642329 |
| Konfusionsmatrix | exakt identisch |
| Sämtliche four_label_v2-Sicherheitsmetriken | exakt identisch |

Dieses Ergebnis sichert die Inferenz des vorhandenen HGB auf der eingefrorenen
Validierung ab. Es garantiert nicht bitgleiches erneutes Training auf der
abweichenden Tabelle oder identische Vorhersagen beliebiger anderer Modelle.

## Ursache und verbleibende Grenzen

Der API-Wechsel von `np.trapz` auf `np.trapezoid` erklärt die Abweichungen nicht:
Im vollständigen Neuaufbau stimmten alle 106.384 Integrationsaufrufe bitweise
mit der nachgebildeten früheren eindimensionalen Trapezformel überein.
Die übrigen sehr kleinen Unterschiede passen zu Rundung in historischen
Einlese-/CSV-Zwischenschritten oder damaligen Bibliotheksversionen. Eine
abschließende Zuordnung ist mangels historischer Extraktionsprovenienz nicht
möglich und wegen der nun geprüften Modellgleichheit kein Blocker für die
numerische Reproduzierbarkeitsaussage.

Die Katalog-Reproduktion ab ursprünglichen Annotationen wurde nachfolgend
byteidentisch bestätigt. Die Featurewerte bleiben unter der beschriebenen
Toleranz numerisch statt byteidentisch reproduzierbar. Universelle Aussagen
über andere Plattformen oder unabhängige neue Scans sind nicht belegt.

## Nachweise und Freigabestand

Neue Audit-Artefakte, ohne Überschreiben früherer Prüfergebnisse:

```text
results/reproducibility_checks/model_consistency_20261102T211838Z_510d2c35/
```

Darin: ausgeführtes Prüfskript, `comparison.json`,
`per_feature_differences.csv`, `per_feature_differences.md` sowie
`validation_prediction_comparison.csv`. Referenztabellen, Modell und MLflow-
Datenbank wurden vor/nach der Prüfung auf unveränderte Hashes geprüft.
Die bestehende Testsuite desselben Produktionscode-Stands bestand zuletzt in
der frisch installierten Umgebung vollständig: **74 Tests**, acht bekannte
sklearn-Einzelklassenwarnungen. Diese Analyse ergänzt nur Dokumentation und
separate Audit-Artefakte; der Produktionscode wurde nicht geändert.

Der technische Baseline-/Audit-Stand wird getrennt von einer wissenschaftlichen
Freigabe für unabhängige neue Scans veröffentlicht. Der abschließende Tag und
die Auswahl versionierter Nachweise stehen im
[Sessionabschluss](session_closure_2026-11-02.md). Große generierte Prüfungsdaten,
Modelle und lokale MLflow-Dateien bleiben lokal. Die numerische Einzelprüfung
selbst hat keine Git-Veröffentlichung ausgeführt.

## Vollständige Statistik je Feature

| Feature | Abweichende Zellen | Max. absolut | Max. relativ (Referenz) |
| --- | ---: | ---: | ---: |
| y_max | 7443 | 4.4408921e-16 | 2.3545045e-16 |
| y_min | 3655 | 1.7763568e-15 | 2.106044e-16 |
| y_peak_to_peak | 8842 | 3.5527137e-15 | 2.5106146e-15 |
| y_mean | 7609 | 1.3322676e-15 | 3.6033184e-16 |
| y_median | 22262 | 8.8817842e-16 | 2.2200232e-16 |
| y_std | 15474 | 1.110223e-16 | 4.393347e-16 |
| y_abs_mean | 7610 | 1.3322676e-15 | 3.6033184e-16 |
| y_energy | 11562 | 1.4551915e-11 | 3.1444465e-16 |
| y_abs_area | 6300 | 2.2737368e-13 | 3.3866885e-16 |
| x_at_y_max | 0 | 0 | 0 |
| x_at_y_min | 0 | 0 | 0 |
| first_third_energy | 8940 | 7.2759576e-12 | 4.3387821e-16 |
| middle_third_energy | 9382 | 7.2759576e-12 | 4.3763114e-16 |
| last_third_energy | 8739 | 7.2759576e-12 | 4.3285961e-16 |
| first_third_std | 18780 | 2.220446e-16 | 4.383091e-16 |
| middle_third_std | 22491 | 5.5511151e-17 | 5.7840555e-16 |
| last_third_std | 27859 | 5.5511151e-17 | 4.7119505e-16 |
| second_peak_max | 21630 | 1.7763568e-15 | 6.4210884e-14 |
| second_peak_min | 13466 | 1.7763568e-15 | 1.4415412e-14 |
| second_peak_max_prominence | 27340 | 1.3322676e-15 | 1.1104165e-10 |
| second_peak_max_width | 38618 | 2.7966962e-11 | 5.0832503e-12 |
| second_peak_min_prominence | 21754 | 1.3322676e-15 | 3.8462385e-11 |
| second_peak_min_width | 25258 | 5.286438e-12 | 2.5449551e-12 |
| second_peak_distance | 0 | 0 | 0 |
| second_peak_max_to_min_ratio | 27402 | 6.2172489e-14 | 6.7259687e-14 |
| second_peak_amp | 18754 | 1.7763568e-15 | 1.1306608e-14 |
| second_peak_snr | 50427 | 8.8107299e-13 | 1.7184535e-14 |
| second_peak_to_surface_ratio | 22285 | 1.3544721e-14 | 1.6679186e-14 |
| first_fifth_std | 39340 | 6.9388939e-17 | 1.4005814e-15 |
| second_fifth_std | 43551 | 6.2450045e-17 | 2.4802278e-15 |
| third_fifth_std | 42224 | 7.6327833e-17 | 3.8573749e-15 |
| fourth_fifth_std | 41652 | 8.3266727e-17 | 2.4241557e-15 |
| fifth_fifth_std | 48227 | 7.4593109e-17 | 5.4517747e-15 |
| max_window_std | 37367 | 8.3266727e-17 | 1.2549671e-15 |
| x_of_max_window_std | 0 | 0 | 0 |
| first_fifth_energie | 50419 | 3.2684966e-13 | 3.1803376e-14 |
| second_fifth_energie | 52775 | 1.7763568e-13 | 3.6210678e-14 |
| third_fifth_energie | 48275 | 2.4868996e-13 | 2.7749782e-14 |
| fourth_fifth_energie | 52537 | 2.2382096e-13 | 2.9378007e-14 |
| fifth_fifth_energie | 52196 | 3.2862602e-14 | 1.1431373e-13 |
| max_window_energy | 50726 | 3.2684966e-13 | 2.9378007e-14 |
| x_of_max_window_energy | 0 | 0 | 0 |
| post_peak_energy | 49364 | 9.9475983e-13 | 1.5134509e-14 |
| post_peak_std | 23476 | 5.5511151e-17 | 5.7167974e-16 |
| post_peak_abs_area | 41765 | 9.5923269e-14 | 7.1710379e-15 |
| post_peak_max_pos_slope | 23967 | 1.7763568e-14 | 1.7704561e-13 |
| post_peak_max_neg_slope | 25111 | 1.7763568e-14 | 2.085581e-13 |
| mean_abs_derivative | 47871 | 3.6082248e-16 | 1.2520857e-14 |
| std_derivative | 44798 | 3.0531133e-16 | 8.4537608e-15 |
| post_peak_max_sec_pos_slope | 29817 | 3.5527137e-13 | 4.2553118e-13 |
| post_peak_max_sec_neg_slope | 31887 | 2.6645353e-13 | 3.7683201e-13 |
| second_peak_surface_corr | 50672 | 1.4391266e-14 | 2.4468644e-11 |
