# Wiederverwendbare historische Prüfskripte

Aus der Projektwurzel mit `.venv/bin/python scripts/reproducibility/<Skript> --help`
aufrufen. Alle Eingaben sind explizit; relative Eingaben beziehen sich auf das
Arbeitsverzeichnis. `--output-dir` muss neu sein. Es erfolgt kein Training.
Die ursprünglichen Fassungen bleiben vollständig im externen Archiv; dessen
Zuordnung steht im [zentralen Überblick](../../docs/reproducibility_checks.md).

| Skript | Erforderliche Argumente zusätzlich zu `--output-dir` |
| --- | --- |
| `feature_probe.py` | `--catalog` kombinierter Katalog; `--reference` Referenz-Feature-CSV |
| `verify_features.py` | `--catalog`; `--reference` |
| `verify_model_consistency.py` | `--reference`; `--rebuilt` neu extrahierte Feature-CSV; `--model` vertrauenswürdiges historisches `model.pkl`; `--predictions` historische Validierungsvorhersagen; `--reference-run-id` |
| `audit_catalog_leakage.py` | `--raw-root` Rohdatenwurzel; `--prior-check-dir` vollständiger ursprünglicher Feature-Prüfordner mit `initial_state.json` und `feature_rebuild/` |

Beispiel für eine kleine Vorprüfung (liest auch Dateigrößen des gesamten Katalogs):

```sh
.venv/bin/python scripts/reproducibility/feature_probe.py \
  --catalog results/measurements/combined_dataset.csv \
  --reference results/feature_tables/feature_table.csv \
  --output-dir results/reproducibility_checks/new_probe
```

Der vollständige Feature- und Katalogneuaufbau liest alle Rohmessungen und kann
mehrere Minuten dauern. Für wiederholte Aufrufe jeweils neue Ausgabeziele wählen.
Der Featurevergleich bewahrt das ursprüngliche strenge Kriterium `1e-12`;
der separate Modellvergleich bewahrt `rtol=1e-11`, `atol=1e-13`.
Die Skripte dokumentieren Befunde in JSON/CSV; ein erfolgreicher Prozessabschluss
allein bestätigt nicht, dass sämtliche numerischen Vergleichskriterien erfüllt sind.

Split-Manifeste, Lockdatei und Referenztabellen werden weiterhin aus dem Projekt
verwendet. Der Modellvergleich schützt auch die vorhandene `mlflow.db` per Hash.
Die Katalogprüfung verwendet historische Eingabe- und Quellcodehashes: bei einem
abweichenden Daten-/Code-Stand bricht sie absichtlich ab. Das ist keine portable
Rekonstruktion der ursprünglichen absoluten Messpfade. Die Probe verwendet wie
historisch 64 gleichmäßig verteilte Katalogzeilen. Keine neue Zufallsaufteilung.
