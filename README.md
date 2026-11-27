# HeRAI – Hohlräume in THz-Reflexionsmessungen

Ein Messpunkt hat eines von vier Labels: schlechte Messqualität (0), Hohlraum (1), unsicher (2) oder kein Hohlraum (3). Beim Modellvergleich hat eine falsche Freigabe (1/2 → 3) Vorrang vor Macro-F1. Auf einzelnen Prüfkörpern verfehlen selbst konservative Regeln sämtliche annotierten Hohlräume. [Modellexperimente und Ergebnisse](docs/modellversuche.md)

## Einstieg in den Code

1. **Messung und Prüfkörper.** `src/paths.py` legt Eingabepfade fest. `src/data/labeling/rebuild_catalog.py` verbindet XY-Annotationen und Rohsignale. `src/data/reference_measurements.py` schließt Metallflächen aus. `specimen_splits.py` und die Manifeste in `configs/` beschreiben die festen Gruppen.
2. **Merkmale.** In `src/features/` zeigen `loading.py` und `extract_all_features.py` den Weg vom Signal zu den 52 historischen Merkmalen. `specimen_inputs.py` richtet Tabellen anhand der Messungs-ID aus. `specimen_grid.py` bereitet die Merkmale je Trainingsfold auf.
3. **Bewertung.** `src/evaluation/metrics.py` zählt die Klassenübergänge. `svm_nested_selection.py` wählt Modell und Freigaberegel auf inneren Folds. [Nenner und historische Definitionen](docs/error_metrics.md)
4. **Experimente.** `src/experiments/model_registry.py` erzeugt die Modelle. `svm_nested_hpo.py` und `tabular_nested_hpo.py` trennen Parametersuche und äußere Bewertung nach Prüfkörpern. `src/analysis/nested_release_pareto.py` und `specimen_ood_review.py` untersuchen, welche Freigaben eine Zusatzprüfung abfängt.
5. **Historische Verfahren.** `src/transformer/`, `src/unsupervised/` und `src/generalization/` enthalten die Signalmodelle, Anomalieverfahren und externen Karten. Die älteren Einstiegsskripte verwenden feste Ergebnispfade.

## Lokal arbeiten

Python 3.12 verwenden. Eine neue Umgebung benötigt `requirements-dev.txt` und `requirements-comparison.txt`, die TSFEL-Extraktion zusätzlich `requirements-tsfel.txt`. Der historische HGB-Lock gilt für Python 3.12.5 auf macOS arm64.

```sh
.venv/bin/python -m ruff check .
MPLBACKEND=Agg .venv/bin/python -m pytest -q
```

Rohmessungen liegen unter `THZ_RAW_DATA_DIR`, externe Scans unter `THZ_EXTERNAL_SCAN_DIR`. Lokale Kataloge, Feature-Caches, Checkpoints und MLflow-Datenbanken sind für vollständige Wiederholungen erforderlich. Im Repository liegen die eingefrorenen Zuordnungen und kompakte Nachweise. [Datenstände und Wiederholung](docs/reproducibility_checks.md)
