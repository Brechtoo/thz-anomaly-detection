# Fehlerraten

`calculate_error_counts` erwartet zwei gleich lange eindimensionale Vektoren mit den ursprünglichen Labels. Alle Raten sind Prozentwerte.

| Zähler | Vorhersagefehler | Nenner |
| --- | --- | --- |
| `critical_errors` | 1/2 → 3 | wahre 1/2 |
| `false_alarms` | 3 → 1/2 | wahre 3 |
| `unsafe_clearances` | 0 → 3 | wahre 0 |
| `unnecessary_remeasurements` | 3 → 0 | wahre 3 |
| `overall_unsafe_clearances` | 0/1/2 → 3 | wahre 0/1/2 |

Bei leerem Nenner liefert diese Funktion 0, die klassenweise Aufschlüsselung in `critical_breakdown` liefert NaN. Die zusätzlichen Ampelzähler 1 → 0/2 und 3 → 0/2 überlappen mit anderen Fehlergruppen. Wer 0 vor der Zählung auf 2 umcodiert, zählt eine freigegebene schlechte Messung als kritischen Hohlraumfehler.

Historische HGB-, Transformer- und Schwellenvergleiche nach `rgy_v1` zählten ausschließlich 1 → 3 bzw. 3 → 1. Bei `four_label_v2` zählt Klasse 2 zu den sicherheitsrelevanten Fällen. Beim Vergleich alter Tabellen die Metrikversion und den Nenner prüfen. Eine explizite Prüfung zurückgestellter Freigaben hat in den späteren Versuchen eine eigene Prüfquote.
