"""Render the audited nested-CV tables and paired specimen sensitivity in German."""

import json

import pandas as pd

from src.experiments.tabular_nested_hpo import OUTPUT
from src.paths import PROJECT_ROOT

FAMILIES = {
    'rbf_svm': 'RBF-SVM', 'mlp': 'MLP', 'logistic_regression': 'LR', 'hgb': 'HGB',
    'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
}
FEATURES = {'manual': 'M', 'manual+tsfel': 'M+T', 'manual+tsfel+physical_v2': 'M+T+P'}


def fmt(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def table(headers, rows):
    return '\n'.join('| ' + ' | '.join(parts) + ' |' for parts in [headers, ['---'] * len(headers), *rows])


def write_summary(output=OUTPUT):
    report, sensitivity = output / 'report', output / 'robustness'
    status = json.loads((report / 'status.json').read_text())
    verified = (output / 'verification.json').exists()
    complete = status['complete'] and verified
    metrics = pd.read_csv(report / 'decision_metrics.csv')
    pooled = metrics.query("scope == 'pooled' and complete")
    pooled = pooled.assign(family=pooled.model.map(FAMILIES), inputs=pooled.features.map(FEATURES))
    intervals = pd.read_csv(sensitivity / 'paired_intervals.csv')
    loo = pd.read_csv(sensitivity / 'leave_one_specimen_out.csv')
    modes = {'adapted': 'Adaptive Freigabe', 'regular': 'Reguläre Entscheidung'}
    checked = pooled.query("mode == 'adapted' and model != 'rbf_svm'")
    if complete and (len(checked) != 15 or len(intervals) != 120 or len(loo) != 690):
        raise ValueError('Incomplete final report')
    counts = pooled[['n', 'critical_support', 'class3_support']].drop_duplicates()
    if len(counts) != 1:
        raise ValueError('Inconsistent evaluation cohort')
    total, critical, class3 = (f'{n:,}'.replace(',', '.') for n in counts.iloc[0])
    text = [
        '# Nicht räumliche Modelle im Vergleich zur optimierten RBF-SVM',
        f'{"Abschluss" if complete else "Zwischenstand"} mit {len(checked)} von 15 '
        'vollständig über drei äußere Gruppen-Folds bewerteten Modell-/Feature-Kombinationen. '
        f'Jede Tabellenzeile umfasst dieselben {total} Messungen der {loo.specimen_id.nunique()} Trainingsprüfkörper. '
        f'{critical} Messungen tragen das Label 1 oder 2, {class3} das Label 3. '
        'M steht für manuelle Features, T für TSFEL und P für Physik V2.',
        'Die Spalten 1→3 und 2→3 zählen kritische Fehler. Kritisch % bezieht sich auf alle Messungen '
        'der Klassen 1 und 2. Fehlalarm % zählt Entscheidungen 1 oder 2 unter allen wahren 3. '
        'BA bezeichnet Balanced Accuracy. Die gepaarte Differenz Δ vergleicht Modell minus SVM '
        'bei gleichem Feature-Set und gleicher Entscheidungsart. Negative Δ bei Fehlerquoten sind günstig.',
    ]
    for mode, title in modes.items():
        rows = pooled.loc[pooled['mode'].eq(mode)].sort_values(['inputs', 'family'])
        bounds = intervals.loc[(intervals['mode'] == mode) & intervals.metric.eq('critical_error_rate')]
        rows = rows.merge(bounds[['model', 'features', 'delta', 'delta_lower_95', 'delta_upper_95']],
                          how='left', on=['model', 'features'], validate='one_to_one')
        values = []
        for r in rows.itertuples():
            interval = 'Referenz' if r.model == 'rbf_svm' else (
                f'{fmt(r.delta)} [{fmt(r.delta_lower_95)}; {fmt(r.delta_upper_95)}]'
            )
            values.append([r.family, r.inputs, str(r.critical_1_errors), str(r.critical_2_errors),
                           fmt(r.critical_error_rate), fmt(r.false_alarm_rate), fmt(r.macro_f1, 3),
                           fmt(r.balanced_accuracy, 3), interval])
        text.extend([f'## {title}', table(
            ['Modell', 'Features', '1→3', '2→3', 'Kritisch %', 'Fehlalarm %', 'Macro-F1', 'BA',
             'Δ kritisch in Prozentpunkten [95-%-Intervall]'], values)])
    best = checked.sort_values(['critical_error_rate', 'macro_f1'], ascending=[True, False]).drop_duplicates('model')
    rows = []
    for r in best.itertuples():
        bodies = loo.loc[loo.model.eq(r.model) & loo.features.eq(r.features) & loo['mode'].eq('adapted')]
        eligible = bodies.loc[bodies.critical_support.gt(0)]
        gain = eligible.loc[eligible.delta_errors_on_specimen.idxmin()]
        loss = eligible.loc[eligible.delta_errors_on_specimen.idxmax()]
        d = eligible.delta_errors_on_specimen
        rows.append([r.family, r.inputs, f'{sum(d < 0)}/{sum(d == 0)}/{sum(d > 0)}',
                     f'{gain.specimen_id} ({int(gain.delta_errors_on_specimen):+d})',
                     f'{loss.specimen_id} ({int(loss.delta_errors_on_specimen):+d})',
                     f'{fmt(bodies.delta_without_critical_error_rate.min())} bis '
                     f'{fmt(bodies.delta_without_critical_error_rate.max())}'])
    text.extend([
        '## Einfluss einzelner Prüfkörper',
        'Je Familie zeigt die Tabelle das Feature-Set mit der niedrigsten beobachteten adaptiven kritischen Rate. '
        'Diese Auswahl dient der Beschreibung. Die erste Zählspalte enthält ausschließlich Körper mit '
        'kritischen Sollklassen. Die letzte Spalte gibt die verbleibende gepoolte Ratendifferenz an, wenn je '
        'ein Körper mit allen seinen Messungen aus der Auswertung entfällt. Ein Vorzeichenwechsel zeigt, '
        'dass schon ein einzelner Körper die Richtung des Vergleichs ändern kann.',
        table(['Modell', 'Features', 'Weniger/gleich/mehr Fehler', 'Kleinste Fehlerdifferenz',
               'Größte Fehlerdifferenz', 'Δ kritisch ohne je einen Körper (Prozentpunkte)'], rows),
        '## Klassenweise Güte der beschriebenen Betriebspunkte',
        table(['Modell', 'Features', 'F1 Klasse 0', 'F1 Klasse 1', 'F1 Klasse 2', 'F1 Klasse 3', 'Freigaben %'],
              [[r.family, r.inputs, *[fmt(getattr(r, f'f1_{i}'), 3) for i in range(4)], fmt(r.release_rate)]
               for r in pd.concat([best, pooled.query("model == 'rbf_svm' and mode == 'adapted'")]).itertuples()]),
        '## Unsicherheit und Interpretation',
        'Die 95-%-Perzentilintervalle verwenden 2.000 gepaarte Bootstrap-Ziehungen ganzer Prüfkörper '
        'innerhalb der äußeren Folds bei festen OOF-Vorhersagen. Der gepoolte Schätzer gewichtet '
        'Messungen nach ihrem Vorkommen. Prüfkörper mit vielen kritischen Messungen haben entsprechend '
        'größeren Einfluss. Die Intervalle beschreiben die Variation der untersuchten Körper. '
        'Überlappende Trainingsmengen, Modelltraining auf anderen Kohorten, Vorauswahl der Familien '
        'auf historischen OOF-Ergebnissen derselben Körper und Mehrfachvergleiche begrenzen die Inferenz.',
    ])
    if complete:
        top = best.iloc[0]
        target = checked.loc[checked.critical_error_rate.lt(2)]
        selections = []
        for model in checked.model.drop_duplicates():
            paths = [output / 'fits' / model / features / f'outer_{fold}/selection.json'
                     for features in FEATURES for fold in range(3)]
            choices = [json.loads(path.read_text())['adapted'] for path in paths]
            outer = metrics.loc[metrics.model.eq(model) & metrics['mode'].eq('adapted') & metrics.scope.eq('fold')]
            selections.append([FAMILIES[model], str(sum(c['feasible'] for c in choices)),
                               str(sum(c['below_target'] for c in choices)), str(sum(outer.critical_error_rate.lt(2))),
                               str(sum(c['gate'] != 'predict' or c['threshold'] > 0 for c in choices))])
        text.extend(['## Innere Auswahl und äußere Überprüfung',
                     'Jede Familie umfasst neun Entscheidungen (drei Feature-Sets × drei äußere Folds). '
                     'Die innere Zielerfüllung bezieht sich auf die zur Auswahl herangezogenen OOF-Entscheidungen. '
                     'Der äußere Fold prüft den gewählten Betriebspunkt an anderen Körpern. Die neun '
                     'Bewertungen einer Familie verwenden dieselben drei äußeren Gruppen wiederholt.',
                     table(['Modell', 'Innere Qualitätsschranken erfüllt', 'Kritisch < 2 % innen',
                            'Kritisch < 2 % außen', 'Adaptive Regel weicht von predict ab'], selections)])
        text.extend(['## Folgerung', f'{len(target)} von 15 adaptiven Varianten erreichen gepoolt eine kritische '
                     f'Fehlerrate unter 2 %. Den niedrigsten beobachteten Wert hat {top.family} ({top.inputs}) '
                     f'mit {int(top.critical_errors)} kritischen Fehlern ({fmt(top.critical_error_rate)} %), '
                     f'{fmt(top.false_alarm_rate)} % Fehlalarmen und einem Macro-F1 von {fmt(top.macro_f1, 3)}. '
                     'Vor einer Freigabeentscheidung sollten Modellfamilie, Feature-Set, Auswahlregel und '
                     'Akzeptanzgrenzen für beide kritischen Klassen festgelegt und an unabhängigen '
                     'Prüfkörpern überprüft werden.'])
    text.append('Alle gruppierten Kennzahlen, Precision, Recall und F1 je Klasse sowie die harten Entscheidungen '
                'der adaptiv gewählten Fits stehen in [decision_metrics.csv](../report/decision_metrics.csv). '
                'Die Unsicherheit von Fehlalarmrate, Macro-F1 und BA sowie die absoluten Intervalle stehen in '
                '[paired_intervals.csv](paired_intervals.csv). Alle einzelnen Ausschlüsse und Fehlerbeiträge '
                'stehen in [leave_one_specimen_out.csv](leave_one_specimen_out.csv).')
    path = sensitivity / 'vergleich.md'
    temp = path.with_suffix('.tmp')
    markdown = '\n\n'.join(text) + '\n'
    temp.write_text(markdown, encoding='utf-8')
    temp.replace(path)
    if complete and output == OUTPUT:
        result_dir = f'../{output.relative_to(PROJECT_ROOT)}'
        document = PROJECT_ROOT / 'docs/tabular_nested_ergebnisse.md'
        markdown = markdown.replace('](../report/', f']({result_dir}/report/')
        for name in ('paired_intervals.csv', 'leave_one_specimen_out.csv'):
            markdown = markdown.replace(f']({name})', f']({result_dir}/robustness/{name})')
        temp = document.with_suffix('.tmp')
        temp.write_text(markdown, encoding='utf-8')
        temp.replace(document)


if __name__ == '__main__':
    write_summary()
