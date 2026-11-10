# Räumliche Prüfkörper-Kandidaten

**Aktueller Stand:** Die drei oberen Flächen sind als Metallreferenzen bestätigt.
Für neue Experimente ausschließlich `data/metadata_reference_excluded/` verwenden;
Gruppen-Holdout und Folgeexperimente stehen im Experimentkatalog (`docs/experiments.md`, folgt separat).
Der folgende erste Abschnitt dokumentiert den unveränderten historischen Kandidatenstand.

Aus dem unveränderten kombinierten Katalog (53.192 Punkte) wurden pro Messreihe
separat 15, 15 und 8 räumliche Flächen erkannt. Die beschrifteten Plots unter
`data/metadata/messreihe_*.png` zeigen plausible rechteckige Abgrenzungen.
`specimen_summary.csv` enthält für jede ID die Punktzahl und Koordinatengrenzen.

Verfahren: DBSCAN mit euklidischem Radius 1,5 in aufgezeichneten
Koordinateneinheiten und mindestens vier Nachbarn einschließlich Eigenpunkt.
Das Einheitsraster wird einschließlich Diagonalen verbunden, Zwischenräume ab
zwei Einheiten werden nicht überbrückt. Keine Normierung beider Achsen, keine
Labels oder Features zur Gruppierung. Interne Messlücken werden über andere
Nachbarschaftspfade umgangen. Sehr dünn besetzte Flächen können unzugeordnet
bleiben; große vollständige Lücken können einen Körper zerlegen. Die Methode
ist deshalb für diesen visuell geprüften Katalog vorgesehen, nicht als
ungeprüfte universelle Körpererkennung.

Sensitivitätsprüfung am realen Katalog: Radien 1,42, 1,6 und 1,75 liefern in
allen Reihen exakt dieselben Partitionen wie 1,5 (Adjusted Rand Index 1).
Bei Radius 2 verschmelzen in Reihe 2 mehrere benachbarte Flächen. Die Wahl
von 1,5 respektiert die sichtbaren Trennstreifen.

Alle 53.192 Punkte sind räumlich zugeordnet, ohne Rauschpunkte oder mehrdeutige
Randpunkte. Im historischen Kandidatenstand war die obere Fläche fachlich unsicher:

| ID | Punkte | Status |
| --- | ---: | --- |
| messreihe_1_pk_15 | 388 | manuell prüfen |
| messreihe_2_pk_15 | 378 | manuell prüfen |
| messreihe_3_pk_08 | 206 | manuell prüfen |

Im ersten Stand war anhand des Versuchsaufbaus noch zu bestätigen, ob diese Flächen
Prüfkörper oder Referenzen sind. Die Bestätigung als Referenzen ist unten dokumentiert.
Im historischen Kandidatenstand erhielten sie den Status `review_small_region`; dies ist keine Bestätigung
physischer Unabhängigkeit. Räumlich getrennte Flächen können außerdem zum selben
physischen Körper gehören. Ob Körper zwischen Messreihen wiederverwendet wurden,
lässt sich aus lokalen Koordinaten nicht bestimmen. Die IDs unterscheiden
räumliche Kandidaten über alle Messreihen eindeutig, identifizieren aber keine
messreihenübergreifend wiederverwendeten physischen Objekte.

Die allgemeine Regel markiert Flächen unter der halben medianen Clustergröße
zur Prüfung. Cluster mit weniger als 100 Punkten sowie Rauschpunkte bleiben
ohne ID. Randpunkte in Reichweite mehrerer Cluster-Kerne bleiben ebenfalls ohne
ID; der Status nennt den Grund. Alle Eingangs-IDs bleiben in der Mapping-Tabelle.

Reproduktion als expliziter Pipeline-Schritt nach `rebuild_catalog`, vor einer
späteren Gruppenauswertung (Ausgabeordner muss neu sein):

```sh
.venv/bin/python -m src.data.specimen_mapping \
  --input data/measurements/combined_dataset.csv \
  --output-dir data/metadata_neu
```

Bestehende Katalog-, Feature- und Split-Manifeste werden nicht ergänzt oder
überschrieben. Das neue Manifest hält Eingangs-, Code-, Lock- und Ausgabehashes,
Paketversionen, Git-Zustand und Parameter fest. Kein Training und kein
MLflow-Experiment wird gestartet. IDs entstehen deterministisch nach minimalem
Y, dann minimalem X je Cluster; die Tabelle wird nach `instance_id` sortiert.
Sortierung einer zufällig permutierten realen Eingabe lieferte dieselbe Tabelle.
Die IDs hängen vom Katalogumfang ab; neue Daten erfordern neue versionierte
Metadaten und erneute Sichtprüfung. Alle IDs stimmen mit der vorhandenen
Feature-Tabelle überein (bijektiver Abgleich).

Für spätere gruppenbasierte Splits ausschließlich `instance_id,specimen_id`
mit `validate='one_to_one'` an die Feature-Metadaten anbinden, vollständige
Abdeckung prüfen und `specimen_id` separat als `groups` verwenden. Prüfstatus
und Quelleninformationen gehören ebenfalls ausschließlich zu Metadaten.
Die zentrale `prepare_dataset`-Funktion entfernt die optional angefügte
`specimen_id`; die bestehende Feature-Erzeugung übernimmt sie nicht.
Die ursprüngliche Forderung nach separat geprüften Gruppensplits ist inzwischen
im Prüfkörper-Holdout (`docs/specimen_holdout.md`, folgt separat) umgesetzt. Die physische Wiederverwendung
zwischen Reihen bleibt weiterhin ungeklärt. Historische Splits bleiben unverändert.

## Aktualisierung: bestätigte Referenzmessungen

Der Dateneigentümer hat die drei oberen Flächen als Referenzmessungen bestätigt.
Der aktuelle, separat erzeugte Stand liegt unter **`data/metadata_reference_excluded/`**;
`data/metadata/` dokumentiert den vorherigen Kandidatenstand und ist nicht als
Trainingsfreigabe zu verwenden. Die alten räumlichen IDs bleiben zur Nachverfolgung
erhalten; `measurement_role=reference`, `training_eligible=False` und
`assignment_status=confirmed_reference` unterscheiden die Referenzen eindeutig.
Es gibt 14, 14 und 7 Prüfkörperflächen sowie je eine Referenzfläche.
Ausgeschlossen werden 388 + 378 + 206 = **972** Punkte; es verbleiben
20.916 + 21.044 + 10.260 = **52.220** Punkte.

`specimen_dataset.csv` enthält ausschließlich die verbleibenden Katalogzeilen,
mit unveränderten IDs, Labels und Datenpfaden. Bestehende Feature-Tabellen und
historische Split-Manifeste bleiben unverändert. Die Regel ist explizit auf die
bestätigten Koordinatengrenzen der drei Messreihen beschränkt, nicht auf allgemein
kleine Cluster. `exclude_references(df)` filtert auch bereits extrahierte Features
über ihre bestehenden IDs ohne Veränderung der Feature-Werte.

Die zentralen überwachten/unüberwachten Datenladewege, Transformer-Signalaufbereitung
und tsfresh-Tabellenerstellung verweigern Eingaben mit Referenzen. Das vermeidet
stilles Weitertrainieren auf alten Tabellen und inkonsistente ID-Zuordnungen.
Daher brechen entsprechende alte Trainingsaufrufe jetzt mit einer erklärenden
Meldung ab. Für zukünftige Läufe referenzfreie Eingaben und separate neue
Experiment-/Split-Konfigurationen verwenden. Bei diesem ursprünglichen Referenz-
ausschluss wurden noch keine neuen Splits erzeugt oder Modelle trainiert;
die späteren Experimente stehen im Experimentkatalog.
Die historische tsfresh-Selektion/Imputation wurde noch mit Referenzen bestimmt;
für eine neue Evaluation muss sie auf dem neuen Trainingsbestand neu gefittet werden.
