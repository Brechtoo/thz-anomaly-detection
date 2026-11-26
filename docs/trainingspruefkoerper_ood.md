# Verbleibende kritische Freigaben und zusätzliche Prüfung

**Auf `messreihe_2_pk_08` lassen beide untersuchten Betriebspunkte alle elf Hohlraummessungen durch. Keiner der vier Detektoren fängt bei den untersuchten Prüfbudgets eine davon ab.** Auf anderen Körpern erzielt zusätzliche Prüfung einen messbaren Gewinn.

## Was PK08 von zuverlässigen Körpern unterscheidet

Die elf Klasse-1-Punkte bilden eine zusammenhängende Insel bei X = 65–67 und Y = 12–16. Der Körper enthält außerdem 1.313 Messungen der Klasse 3, 241 der Klasse 0 und keine der Klasse 2. Die elf Labels sowie die 59 umliegenden Klasse-3-Labels im Radius von drei Rastereinheiten stimmen mit der ursprünglichen XY-Annotation überein. Die physische Lage und Tiefe des Hohlraums wurden nicht unabhängig vermessen.

| Trainingsmessungen aus Reihe 2 | Anzahl | Verhältnis zweiter Peak / früher Oberflächenmarker, Median [Q1–Q3] | Spitze-Spitze-Amplitude, Median |
|---|---:|---:|---:|
| PK08, alle wahren 1 | 11 | 0,126 [0,114–0,136] | 5,77 |
| PK08, räumlich benachbarte wahre 3 | 59 | 0,082 [0,079–0,086] | 7,49 |
| PK04, korrekt als 1 klassifiziert, gleicher äußerer Fold | 630 | 1,384 [1,096–1,752] | 0,64 |
| PK07 und PK09, korrekt als 1 klassifiziert | 1.225 | 1,483 [1,130–1,851] | 0,65 |
| Alle wahren 1 aus dem Fit-Anteil für Außenfold 2 | 1.353 | 1,412 [0,997–1,817] | 0,68 |
| Alle wahren 3 aus demselben Fit-Anteil | 4.362 | 0,147 [0,087–0,383] | 3,52 |

Auch die Rohsignale zeigen die Verschiebung. Nach Ausrichtung am frühen Extremum beträgt das mediane maximale Nachlaufsignal im Abstand 6–40 X-Einheiten relativ zu diesem Extremum **0,105** auf den elf Fehlerpunkten, **0,046** bei lokalen wahren 3 und **1,000** bei korrekt erkannten Hohlräumen auf PK04. Für die beiden größeren Vergleichsgruppen wurden je 32 Signale deterministisch gezogen. Die elf Fehler liegen innerhalb des typischen Bereichs der Klasse 3 der Fit-Körper für das Peak-Verhältnis. Das spricht für eine Verschiebung der Signalausprägung innerhalb der Sollklasse 1. Hohlraumgeometrie, Material und mögliche Registrierungsfehler lassen sich aus diesen Daten als Ursachen nicht trennen. [Signalverläufe mit Median und Interquartilsband](../results/analysis/specimen_ood_review_v2/signale.png)

Beim 2-%-Betriebspunkt liegt selbst der kleinere Klasse-3-Score aus MLP und logistischer Regression auf allen elf Fehlern zwischen **0,984 und 0,996**. Unter den je 15 nächsten Messungen anderer Fit-Körper findet sich für keinen Fehler ein kritisches Label. Auf `messreihe_3_pk_06` haben die 47 fehlerhaft freigegebenen wahren 1 ebenfalls ein schwächeres Peak-Verhältnis als die 17 korrekt als 1 erkannten (Median 0,114 gegenüber 0,357). Ihr medianer Freigabescore liegt bei 0,922. Auf `messreihe_1_pk_03` liegen die 42 Klasse-2-Fehler mit 0,897 näher an der Freigabeschwelle 0,879. Ihr normierter Rohsignal-Nachlauf überlappt mit korrekt erkannten 2 (mediane Maxima 0,091 und 0,089). Eine einheitliche Signalursache für alle drei Körper ist damit nicht belegt.

## Ertrag zusätzlicher Prüfungen

Die eingefrorenen 0,5-%- und 2-%-Betriebspunkte mit Freigabepriorität wurden auf allen 34.235 Trainingsmessungen reproduziert. Je äußerem Fold standen Isolation Forest, Distanz zum 15. Nachbarn, Anteil kritischer Labels unter diesen Nachbarn, geringe Modellkonfidenz und die unveränderte Ausgangsregel zur Wahl. Jeder Detektor sieht beim Fit ausschließlich zulässige Körper. Imputation und Standardisierung der 52 manuellen Features lernen aus bis zu 256 Messungen je Fit-Körper (Seed 42). Innere gruppengetrennte OOF-Scores bestimmen die Schwelle und die Detektorwahl nach abgefangenen kritischen Fehlern, bei Gleichstand nach geringstem Prüfbedarf. Nur bisherige Freigaben gehen in Zusatzprüfung. Das innere Budget bezieht sich auf alle Messungen und kann außen überschritten werden.

| Ausgangspunkt | Inneres Zusatzbudget | Abgefangene 1→3 / 2→3 | Zusätzliche Prüfungen außen | Kritisch vorher → nachher | Freigabe nachher |
|---|---:|---:|---:|---:|---:|
| 2 % | 0,5 % | 2 / 8 | 242 (0,71 %) | 1,645 → 1,513 % | 24,96 % |
| 2 % | 1 % | 6 / 12 | 472 (1,38 %) | 1,645 → 1,408 % | 24,29 % |
| 2 % | 2 % | 13 / 22 | 937 (2,74 %) | 1,645 → 1,184 % | 22,93 % |
| 0,5 % | 0,5 / 1 % | je 0 / 0 | 113 / 320 | jeweils 0,368 → 0,368 % | 14,55 / 13,94 % |
| 0,5 % | 2 % | 2 / 2 | 1.121 (3,27 %) | 0,368 → 0,316 % | 11,60 % |

Am 2-%-Betriebspunkt fängt die Konfidenzregel allein bei 1 % innerem Budget **20 Fehler mit 672 Prüfungen** ab. Die feste Nachbarschaftsrisikoregel fängt **drei mit 65** ab, Isolation Forest **keinen mit 225**. Die reine Distanz löst außen bei diesem Budget keine Prüfung aus. Die innere Auswahl fängt **18 Fehler mit 472 Prüfungen** ab. Davon entfallen elf auf Klasse 2 auf `messreihe_1_pk_03` und sechs auf Klasse 1 auf `messreihe_3_pk_06`. Die Gesamtprüfquote steigt von 35,65 % auf 37,03 %, die formale Fehlalarmrate bleibt bei 8,30 %. Die Zahlen zählen zurückgestellte Freigaben vor menschlicher Nachprüfung.

Für geringen Zusatzaufwand ist das kleinste untersuchte Budget am 2-%-Punkt der beobachtete Kompromiss. Am strengeren Ausgangspunkt fehlt bei kleinen Budgets ein Sicherheitsgewinn. Für PK08 bleibt die annotierte Insel der gezielte Fall für eine unabhängige Prüfung von Geometrie und Registrierung. Die explorative Diagnose derselben 23 Körper und die kleine räumlich gekoppelte Fehlergruppe begrenzen Aussagen zur Generalisierung. Validation und Test blieben geschlossen.

[Prüfkörperergebnisse](../results/analysis/specimen_ood_review_v2/specimens.csv) · [Schwellen und innere Auswahl](../results/analysis/specimen_ood_review_v2/selection.csv) · [Feature-Verteilungen](../results/analysis/specimen_ood_review_v2/distributions.csv) · [Evaluationscode](../src/analysis/specimen_ood_review.py) · [Signaldiagnose](../src/analysis/specimen_ood_diagnosis.py)
