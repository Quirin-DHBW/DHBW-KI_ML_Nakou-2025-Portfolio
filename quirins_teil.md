# VibeluX
Tim Schacht, Quirin Barth

## Idee/Umfang

Das Ziel ist es, zwei Convolutional Neural Networks zu trainieren, die jeweils ihren gegebenen Input in eine von sieben Emotionen klassifizieren:
Wut, Ekel, Angst, Freude, Neutral, Traurigkeit und Überraschung.
Eines der Modelle wird darauf trainiert, Gesichter zu erkennen, während das andere darauf trainiert wird, diese Emotionen in Musik zu erkennen. Die Vorhersagen beider Modelle können dann verwendet werden, um einem Nutzer Musik basierend auf seinem aktuellen emotionalen Zustand zu empfehlen. Andere Anwendungen umfassen einige versteckte interne Features, die eine Musik-App nutzen könnte, um ihren Shuffle-Algorithmus subtil an die Stimmung des Nutzers anzupassen.

Zur Erleichterung des Trainings wurde ein Datensatz von Gesichtern von Kaggle bezogen: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

Die Musik für das Training des anderen Modells wurde von einer lizenzfreien Musikplattform bezogen: https://pixabay.com/music/

## Theorie

### Was sind CNNs?
Mache Dinge in Stücke, schaue auf Stücke, mache kleinere Stücke, wiederhole, es kann Dinge erkennen, indem es sie zerlegt!
Mathematisch? Du "faltest" die Daten über sich selbst nach bestimmten Regeln, was zu einer kleineren Ausgabe führt, die dichter an Informationen ist, z. B. (8 x 8 x 1) -> (4 x 4 x 4)

Ähnlich wie mehrschichtige Perzeptrons basieren CNNs lose auf der menschlichen Wahrnehmung, jedoch in direkterer Weise, indem sie "Filter" auf ein Bild anwenden. Diese Filter sind im Wesentlichen ein gleitendes Fenster, das nach bestimmten Mustern sucht und ausgibt, wie stark dieses Muster vorhanden ist. Die resultierende Musterkarte kann dann an die nächste CNN-Schicht weitergegeben werden, um größere zusammenhängende Muster zu erkennen! Das bedeutet, dass man sich vorstellen könnte, durch Betrachtung der Filter zu sehen, was das CNN "denkt"! Leider ist dies nicht wirklich der Fall, da es viele mögliche stabile und funktionale lokale Minima im Raum der Verlustfunktion gibt, sodass die meisten CNN-Filter genauso schwer verständlich sind wie jedes andere neuronale Netzwerk.

### Warum Maxpooling?
Quetsche die Daten, mache sie kleiner, jetzt weniger Daten mit hoffentlich noch relevanter Information -> Weniger Berechnung, yay :D

### Warum Dropout?
Dropout deaktiviert während des Trainings zufällig Neuronen oder ihre Entsprechungen in einem Netzwerk und skaliert die Ausgaben so, dass sie immer noch korrekt zurückpropagiert werden können. Dies ist eine Regularisierungsmethode, die Überanpassung reduziert und das Modell dazu zwingt, relevantere Merkmale zu lernen, die auf mehr Neuronen verteilt sind. Dadurch wird das Netzwerk dazu gebracht, tatsächlich mehr Neuronen zu nutzen, anstatt später im Training bestimmte Neuronen einfach zu ignorieren (siehe ReLU, das bei 0 stecken bleibt und "stirbt").

### Gesichts-Klassifikations-Datensatz
48x48 Graustufen-Gesichter!

### Musik-Klassifikations-Datensatz
Lieder, die mit ihrer Emotion oder einem Synonym dieser Emotion getaggt wurden, wurden ausgewählt, alle auf 1:30 Länge zugeschnitten/gepolstert und dann in Spektrogramme umgewandelt.

### Was sind Spektrogramme?
Musik besteht aus Frequenzen? Fourier-Transformation zur Frequenzgewinnung! Frequenzen (y) über Zeit (x) mit ihrer Lautstärke/Amplitude als Farbe (Graustufen) plotten, und du erhältst ein Spektrogramm, das das Audio vollständig beschreibt!!! Diese können sogar zurück in Audio umgewandelt werden! Für dieses Projekt bedeutet das aber, dass wir Musik in Bilder umwandeln können, die sie vollständig beschreiben. Indem wir die Musik auf genau 1:30 Länge zuschneiden oder puffern, können wir sicherstellen, dass die Bilder immer dasselbe Format haben, was sie als CNN-Input problemlos nutzbar macht.

Warum das Ganze? Weil frühere Forschungen gezeigt haben, dass CNNs nicht unbedingt menschenlesbare Bilder brauchen, sondern nur Details, die zur Klassifikation genutzt werden können. Es wurde bereits bewiesen, dass man ein CNN auf Musik-Spektrogramme trainieren kann und damit gute Ergebnisse erzielt! [TODO](QUIRIN GEH UND FINDE QUELLEN)

### Modellarchitektur
Mehrere CNN->Maxpool->Dropout-Schichten (unterscheiden sich je nach Gesichts- oder Musikklassifikator)
2x Dense mit abnehmender Neuronenzahl und Dropout
1x Dense-Schicht mit 7 Neuronen und Softmax-Aktivierungsfunktion

Dropout wurde als Regularisierungsmethode gewählt.

Wir verwenden die SparsecategoricalCrossentropy Loss für das Training, zusammen mit AdamW. (AdamW wurde gegenüber Adam aufgrund seiner Gewichtsnormierung gewählt.)
Genauigkeit wurde als zusätzliche Metrik verfolgt, um einen einfacheren Vergleich zu ermöglichen.

### Emotionale Einbettungssuche
Beide Modelle erzeugen einen emotionalen "Vektor", der das Gesicht oder die Musik einer Emotion zuordnet. Musik hat viele Facetten, und anstatt nur die primäre Emotion zu erkennen und passende Lieder auszuwählen, haben wir stattdessen die Kosinus-Ähnlichkeit verwendet, um das Lied oder die Lieder zu finden, die am besten zu allen 7 erkannten Emotionen im Gesicht passen.

## Ergebnis
Gesichtserkennung? Funktioniert! Ein einfaches Netzwerk wurde in TensorFlow erstellt und in weniger als 10 Minuten trainiert und lieferte akzeptable Ergebnisse! Genauigkeit von 0,545 (was niedrig klingt, aber mit 7 Klassen, von denen einige ähnlich aussehen können, z. B. wütend und traurig, ist dies für unsere Zwecke gut genug, zudem ergibt sich daraus die Idee von Emotionseinbettungen mit Mischungen der 7 Klassen).

Musikerkennung? Das Training nähert sich einem Verlust von 1,94 / Genauigkeit von 0,14 und verbessert sich danach nicht weiter. Bemerkenswert ist, dass das Modell niemals besser als zufällige Auswahl wird und tatsächlich oft schlechter abschneidet. Dies liegt wahrscheinlich an einem Mangel an Trainingsdaten, da derzeit nur 80 Lieder pro Emotion (insgesamt 560 Lieder) vorhanden sind und ein Lied-Spektrogramm ein ziemlich großes Bild (~1k) ist, was ein größeres Modell erfordert als die 48x48 der Gesichtserkennung.

Einbettungssuche mittels Kosinus-Ähnlichkeit? Funktioniert in Tests, aber der Musikklassifikator funktioniert nicht richtig, sodass wir derzeit keine **echten** Einbettungen für Lieder erzeugen können.

