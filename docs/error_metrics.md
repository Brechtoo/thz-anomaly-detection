# Fehlerbewertung: four_label_v2

Labels: `0` schlechte Messung, `1` Hohlraum, `2` möglicher Hohlraum,
`3` kein Hohlraum. Die zentrale Funktion benötigt ursprüngliche, gleich lange,
eindimensionale Labelvektoren. Alle Raten sind **Prozentwerte (0–100)**.

| Anzahl / Rate im Rückgabedictionary | Ereignis | Nenner |
| --- | --- | --- |
| `critical_errors` / `critical_error_rate` | `1/2 → 3` | tatsächliche `1/2` |
| `unsafe_clearances` / `unsafe_clearance_rate` | `0 → 3` | tatsächliche `0` |
| `false_alarms` / `false_alarm_rate` | `3 → 1/2` | tatsächliche `3` |
| `unnecessary_remeasurements` / `unnecessary_remeasurement_rate` | `3 → 0` | tatsächliche `3` |
| `overall_unsafe_clearances` / `overall_unsafe_clearance_rate` | `0/1/2 → 3` | tatsächliche `0/1/2` |

Fehlt eine Nennerklasse, wird ihre Rate als `0.0` zurückgegeben. Dies ist eine
technische Konvention, kein Leistungsnachweis auf dieser Klasse. Leere Eingaben
ergeben ausschließlich Nullwerte. Ungleiche Längen oder mehrdimensionale
Eingaben werden abgelehnt, damit NumPy-Broadcasting keine Fehlerzahlen verfälscht.

Die Gesamtanzahl ist die Summe aus Critical Errors und Unsafe Clearances.
Die Gesamtrate verwendet deren gemeinsamen Support; sie ist weder die Summe
noch das ungewichtete Mittel der beiden Einzelraten.

## Verhältnis zur bisherigen Rot/Gelb/Grün-Logik

Die bestehenden Zusatzmetriken bleiben erhalten:

- `red_to_yellow`: `1 → 0/2`, Rate ausschließlich bezogen auf tatsächliche `1`.
- `green_to_yellow`: `3 → 0/2`, Rate bezogen auf tatsächliche `3`.

`3 → 2` ist sowohl False Alarm als auch grün-gelber Übergang.
`3 → 0` ist sowohl unnötige Wiederholungsmessung als auch grün-gelber Übergang.
Die Gesamtfreigaben enthalten bereits die kritischen Fehler und die Freigaben
schlechter Messungen. Diese überlappenden Metriken dürfen nicht addiert werden.

Die aktuelle Schwellenwertfunktion gibt nur `1`, `2`, `3` zurück. Deshalb sind
Unnecessary Remeasurements dort konstruktionsbedingt null. Unsafe Clearances
bleiben möglich, weil das wahre Label weiterhin `0` sein kann. Modellparameter,
Schwellenwerte und Vorhersagelogik wurden nicht geändert.

## Originale Labels und Aufrufstellen

| Datei | Originale Fehlerlabels | Verwendung |
| --- | --- | --- |
| `src/experiments/run_experiment.py` | `y_val` | Validierungsübersicht und künftige MLflow-Logs |
| `src/experiments/transformer_experiments.py` | `y_true` aus `predict` | Ergebnisdictionary |
| `src/evaluation/evaluate_optimized.py` | `y_val` | optimierte Validierungsübersicht |
| `src/experiments/optimize_thresholds.py` | `y_val` | Validierungsvergleich der Schwellenwerte |
| `src/evaluation/final_evaluation.py` | `y_test` | finale HGB-Auswertung |
| `src/evaluation/final_evaluation_transformer.py` | `y_test` | finale Transformer-Auswertung |

Die letzten drei Stellen verwenden für allgemeine Metriken separat eine Kopie
mit `0 → 2`. Diese Kopie darf nicht an `calculate_error_counts` übergeben werden:
Sonst werden Freigaben schlechter Messungen zu kritischen Fehlern umgezählt.
Bereits umcodierte Labels kann die zentrale Funktion nicht erkennen oder
rekonstruieren. Die Fehlerberechnung verändert ihre Eingaben nicht.

Künftige Ergebnisdictionaries/CSV-Übersichten tragen
`error_metrics_version=four_label_v2`. Künftige Runs aus `run_experiment.py`
erhalten denselben MLflow-Tag und alle Fehlerwerte mit dem Präfix `val_`.
Das numerische Rückgabedictionary der Fehlerfunktion enthält keine Metadaten.

## Historischer Bestand: rgy_v1 (unverändert)

**Die vorhandenen Fehlerauswertungen und MLflow-Runs im untersuchten Bestand
wurden nach alter Definition berechnet.** `rgy_v1` ist eine dokumentarische
Kennzeichnung; alte Dateien und Runs wurden nicht verändert oder umgetaggt.

Die alte Definition zählte nur `1 → 3` als kritisch (Nenner: `1`) und
`3 → 1` als False Alarm (Nenner: `3`). Die drei neuen Sicherheitsmetriken fehlten.
Das war historisch beabsichtigt: Commit `b406ddb` führte die gesonderte Bewertung
von `1/3` bei unsicheren Labels `0/2` ein; `6acea4c` übernahm sie in `metrics.py`.

Neu zu bewerten sind Fehlerwerte und darauf gestützte Entscheidungen in:

- `results/experiments/{baseline_features,extended_features,hgb_baseline_reproducibility}/validation_results.csv`.
- `results/experiments/extended_features/optimization/optimized_validation_results.csv`.
- Unter `results/experiments/extended_features/`: `threshold_results.csv`,
  `threshold_comparison.csv`, `threshold_comparison_fine.csv`,
  `threshold_comparison_fine_093_095.csv`.
- Im Unterordner `optimization/`: `optimized_threshold_comparison.csv` und
  `optimized_rgy_threshold_comparison.csv`.
- `results/experiments/extended_features/final_evaluation/results.csv`.
- Fehlervergleichen aus gespeicherten Transformer-Vorhersagen und den alten
  `val_critical_*`/`val_false_alarm_*`-Werten in MLflow.
  `transformer_weight_comparison.csv` enthält selbst keine Fehlerzählungen.

Accuracy, Balanced Accuracy und F1 ändern sich bei gleichen Labels und
Vorhersagen nicht. Rein F1-basierte Optimierungswerte bleiben deshalb gleich;
die Sicherheitsbewertung ihrer Ergebnisse muss trotzdem ergänzt werden.
Neue Schwellenwertentscheidungen gehören auf die Validierungsmenge, nicht auf
die finale Testmenge.

Vorhandene Originalvorhersagen der Baseline-, Extended-, optimierten und
Transformer-Validierungen erlauben grundsätzlich eine neue Fehlerauswertung
ohne Training. Schwellenwertübersichten allein enthalten nicht alle benötigten
Übergänge. Dafür werden passende Originalvorhersagen oder Wahrscheinlichkeiten
und Labels benötigt.

**Einschränkung finaler HGB-Berichte:** Die dortigen `predictions.csv` und
Konfusionsmatrizen enthalten bereits `0 → 2`. Sie reichen allein nicht aus,
um Critical Errors, Unsafe Clearances und deren Nenner separat zu bestimmen.
Dafür sind zuverlässig zugeordnete ursprüngliche Testlabels erforderlich.
Auch künftig sind diese RGY-Berichte von der Fehlerübersicht auf Originaldaten
zu unterscheiden.

Historische Auswertungen dürfen nicht stillschweigend mit v2 verglichen oder
überschrieben werden. Spätere Neubewertungen sollten separat gespeichert werden,
einschließlich Metrikversion, Split- und Vorhersageherkunft. In dieser Änderung
wurden keine Experimente ausgeführt und keine Modelle, Ergebnisse oder Runs verändert.
