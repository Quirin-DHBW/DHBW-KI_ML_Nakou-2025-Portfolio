# VibeluX
Tim Schacht, Quirin Barth


## Einleitung

Das Ziel ist es, zwei Convolutional Neural Networks zu trainieren, die jeweils ihren gegebenen Input in eine von sieben Emotionen klassifizieren:
Wut, Ekel, Angst, Freude, Neutral, Traurigkeit und Überraschung.
Eines der Modelle wird darauf trainiert, Gesichter zu erkennen, während das andere darauf trainiert wird, diese Emotionen in Musik zu erkennen. Die Vorhersagen beider Modelle können dann verwendet werden, um einem Nutzer Musik basierend auf seinem aktuellen emotionalen Zustand zu empfehlen. Andere Anwendungen umfassen einige versteckte interne Features, die eine Musik-App nutzen könnte, um ihren Shuffle-Algorithmus subtil an die Stimmung des Nutzers anzupassen.


## Theorie

### Convolutional Neural Network
Ähnlich wie Multilayer Perceptrons basieren Convolutional Neural Networks, im folgenden CNN genannt, lose auf der menschlichen Wahrnehmung, indem sie mithilfe von "Filtern" auf einem Bild zuerst kleinere, grobe merkmale erkennen, und dann darauf basieren größere Merkmale erkennen können. Diese Filter sind im Wesentlichen ein gleitendes Fenster, das nach bestimmten Mustern sucht und ausgibt, wie stark dieses Muster vorhanden ist oder nicht. Die resultierende Karte mit Musterstärken kann dann an die nächste CNN-Schicht weitergegeben werden, um größere zusammenhängende Muster zu erkennen! Oft werden dabei viele Filter auf einmal verwendet, da jeder Filter jeweils nur ein Merkmal suchen kann.
Mathematisch entspricht dieser Prozess der Funktionsfaltung, da die Daten nach bestimmten Regeln über sich selbst "gefaltet" werden, was zu einer kleineren Ausgabe führt, die "dichter" an Informationen ist bzw. tiefer, z.B. (8 x 8 x 1) -> (4 x 4 x 4) bei 4 Filtern.

### Maxpooling
Da die Anzahl an Datenpunkten in einem Bild sehr schnell (quadratisch mit der Bildgröße) anwächst, und durch die Faltungsoperationen mit mehreren Filtern diese Informationen oftmals sogar vervielfacht werden, lohnt es sich durch bestimmte Methoden die Datenmengen in check zu behalten. Maxpooling teilt ein Bild in Blöcke bestimmter größe, (oft 2x2) und weist diesen dann jeweils den Maximalwert unter den im Block beinhalteten Zellen zu. Dadurch wird die Datenmenge effektiv geviertelt, ohne die am wahrscheinlich wichtigsten Merkmale zu verlieren. Es gibt natürlich auch andere Pooling-Methoden, jedoch ist Maxpooling sehr weit verbreitet. Normalerweise findet man solche Maxpooling layer nach jedem Convolutional Layer in einem CNN, before der output and den nächsten CNN Layer weitergegeben wird.

### Dropout
Dropout deaktiviert während des Trainings zufällig Neuronen bzw. Outputs in einem Netzwerk und skaliert die Ausgaben so, dass sie immer noch korrekt zurückpropagiert werden können. Das Modell muss dadurch lernen sein Wissen breiter im Netzwerk zu verteilen, und sich nicht auf nur eine kleine Anzahl von Neuronen zu verlassen. Die Trainingsdauer wird dadurch zwar oft erhöht, da das Modell jetzt einem weiteren faktor entgegenwirken muss, jedoch sorgt dies auch für ein insgesammt besser angepasstes Modell. Dies ist eine Regularisierungsmethode, die Überanpassung reduziert und das Modell dazu zwingt, relevantere Merkmale zu lernen, die auf mehr Neuronen verteilt werden.

### Spektrogramm
Bei einem Spektrogramm handelt es sich um eine visuelle Darstellung der zeitabhängigen Frequenzanteile eines Signals sowie ihrer jeweiligen Intensität (Halliday et al., 2003). In der Audioverarbeitung können mit ihnen akustische Merkmale sichtbar gemacht werden, die für das menschliche Ohr nicht direkt erkennbar sind. Typischerweise werden Spektrogramme durch die Short-Time Fourier Transformation erzeugt. In diesem Prozess wird das Signal in überlappende Zeitintervalle aufgeteilt und auf jedes anschließend eine Fourier-Transformation angewandt. Die Fourier-Transformation überführt dabei eine zeitabhängige Signalfunktion in eine komplexwertige Funktion, die direkt von der Zeit und der Frequenz des Ursprungssignals abhängt.
Dadurch wird es möglich, die Frequenzanteile in einem Diagramm gegen die zeitliche Signalentwicklung aufzutragen.
Um die Darstellung einheitlicher zu gestalten, wird statt der Fourier-Transformierten, in der die Signalamplitude periodisch das Vorzeichen wechselt, in der Regel das Betragsquadrats der Fourier-Transformierten Funktion aufgetragen, welche auf diese Weise die frequenz- und zeitabhängige, immer positive Signalintensität in einem dreidimensionalen Diagramm darstellt.
Dieses kann zum Beispiel in Form einer Heat-Map als Bild gespeichert werden, wie es auch von den meisten Spektrogramm-erzeugenden Algorithmen umgesetzt wird.


## Umsetzung

### Datenstruktur des Repositories
Die Verzeichnisstruktur von VibeluX setzt sich folgendermaßen zusammen:

**Data**: Enthält die Verzeichnisse `archive`, für Bild-, und `audio`, für Audio-Dateien.
  - **archive**: Besteht aus den Unterordnern `test` und `train`, die jeweils PNG-Dateien für die sieben verschiedene "Haupt-Emotionen" (angry, disgusted, fearful, happy, neutral, sad, surprised) enthalten.
  - **audio**: Beinhaltet die Ordner `Processed`, für die Spektrogramm-Daten transformierter Songs, und `RAW`, für die MP3-Dateien derselben Songs.

**img**: Enhält erklärende Beispielbilder.

**Model_Training**: Enthält die folgenden Dateien und Skripte:
  - `Music_preprocessor.py`: Ein Skript zur Umwandlung von MP3-Dateien in normierte Spektrogramme.
  - `Train_face_emotion_classifier.py`: Ein Skript für das Training eines neuronalen Netzwerks zur Gesichtsemotionserkennung.
  - `Train_music_emotion_classifier.py`: Ein Skript für das Training eines neuronalen Netzwerks zur Musik-Emotions-Zuordnung.

**Songs**: Enthält die MP3-Dateien der im Rahmen des Projekts verwendeten Songs. ???????????????????????????????????

**generate_song_embeddings.py**: Erstellt eine JSON-Datei, die den Songdateinamen den durch ein CNN generierten Emotionen zuordnet.

**webcam_face_recognition.py**: Ein Skript für die Einbindung der Kamera des VibluX ausführenden Gerätes inklusive Gesichtserkennung und Bildformat-Normierung mittels OpenCV und der dazugehörenden Python anbindung durch das `cv2`-Moduls.

**main.py**: Das Hauptskript zur Durchführung des gesamten Prozesses der Gesichtsklassifikation, und anschließenden Embeddingsuche nach einem passenden Musikstück.

ALLE WEITEREN MÜSSEN ENTWEDER NOCH GENANNT ODER GELÖSCHT/SORTIERT WERDEN

### Datensammlung
Im ersten Schritt wurde eruiert, welche Daten für die Umsetzung einer Emotionszuordung zu Gesichtern und Musikstücken nötig sind.
Dabei ergab sich, dass für die Gesichtsdaten ein Datensatz benötigt wird, in dem frontal aufgenommene Nahaufnahmen von Gesichtern enthalten sind, vorklassifiziert nach den jeweilig dargestellten Gesichtsemotionen.
Ein solcher Datensatz konnte in der Online-Community Kaggle gefunden werden, in der er zur freien Weiterverarbeitung zur Verfügung gestellt wurde (Emotion Detection, 2020).
Durch die Wahl dieses Datensatzes waren aufgrund seiner Struktur somit die Anzahl an erkennbaren Emotionen auf 7 festgelegt.
Daraufhin wurden die Musikdateien für das Training des Musik-Emotions-zuordnenden Modells von der lizenzfreien Musikplattform Pixabay bezogen (Pixabay, a Canva Germany GmbH brand, o. D.). Auf dieser Seite können Musikstücke mit Tags versehen werden, nach denen getaggte Musikstücke anschließend über eine Suchfunktion identifiziert werden können. Für die Erstellung des Musikdatensatzes wurde daher nach den 7 erkennbaren Emotionen und ihren Synonymen gesucht und dabei erhaltene Suchergebnisse in den Datensatz `Songs` aufgenommen und mit der jeweils gesuchten Emotion vorklassifiziert.


### Musikdaten-Aufbereitung
Die vorklassifizierten MP3-Dateien aus `Songs` wurden in `Music_preprocessor.py` jeweils als 90 Sekunden lange Ausschnitte eingelesen und mittels des Python-Moduls `librosa` in 8k Bitrate geladen. Per Short-Time Fourier-Transformation wurden sie anschließend in Spektrogramme umgewandelt und in normierter Form gespeichert.
Dies basiert auf einer schon in der Vergangenheit erfolgreich angewandten Methode aus unterschiedlichen Papern, unter anderem Costa et al. (2016).

### Gesichts-Klassifikations-Datensatz
48x48 Graustufen-Gesichter, welche nah and das zu erkennende Gesicht zugeschnitten sind. In dem Datensatz befinden sich insgesamt 28709 Bilder in dem Trainingsdatensatz, und 7178 Bildern im Validierungsdatensatz. Die Bilder sind nicht gleichmäßig under den sieben Emotionsklassen verteilt, also haben manche Emotionsklassen weitaus mehr Trainingsdaten als andere. In der untenstehenden Tabelle sind die prozentualen Anteile der einzelnen Emotionsklassen im Datensatz angegeben:

|Emotionsklasse|Prozent v. Desamtdatensatz|
|-:|:-|
|angry|13.9%|
|disgusted|1.5%|
|fearful|14.3%|
|happy|25.1%|
|neutral|17.3%|
|sad|16.8%|
|surprised|11%|

Wie hier zu sehen ist hat die Emotionsklasse "happy" mit über 25% bei weitem die meisten Trainingsdaten. "disgusted" nimmt wiederum nur 1.5% des Trainningsdatensatzes ein, wodurch es wahrscheinlich weitaus schlechter erkannt werden wird.

### Musik-Klassifikations-Datensatz
Musikstücke, die mit ihrer Emotion oder einem Synonym dieser Emotion auf Pixabay getaggt waren wurden ausgewählt, alle auf 1:30 Länge zugeschnitten/gepolstert und dann bei einer Bitrate von 8k in Spektrogramme umgewandelt.
Es wurden für jede Emotionsklasse 80 Musikstücke herausgesucht, was zu insgesamt 560 Musikstücken in diesem Datensatz führt.

### Main-Programmausführung
Anschließend wurde `Train_face_emotion_classifier.py` verwendet, um ein CNN-Modell für die Gesichtsemotionserkennung zu trainieren.
`Train_music_emotion_classifier.py` trainierte ein weiteres Modell zur Zuordnung von Spektrogrammen zu Emotionen.

Über `webcam_face_recognition.py` wird auf die Gerätekamera zugegriffen, um ein Gesicht zu erfassen, zu verarbeiten und als Graustufenbild abzuspeichern.
Das CNN-Modell `face_emotion_classifier.h5` ermittelt die Emotion des Gesichts.
Anhand der JSON-Datei `song_embeddings.json` werden die erkannten Emotionen mit den gespeicherten Song-Embeddings verglichen, um den am besten passenden Song zu identifizieren.

### Emotionale Embedding Suche
Beide Modelle erzeugen einen emotionalen "Vektor", der das Gesicht oder die Musik einer Emotion zuordnet. Musik hat viele Facetten, und anstatt nur die primäre Emotion zu erkennen und passende Musikstücke auszuwählen, haben wir stattdessen die Kosinus-Ähnlichkeit verwendet, um das Musikstück oder die Musikstücke zu finden, die am besten zu allen 7 erkannten Emotionen im Gesicht passen.


## Ergebnis
Die Gesichtserkennung funktioniert gut genug für unsere Verwendungszwecke, trotz eines ziemlich niedrigen Accuracy Score von 0.545. Der niedrige Score lässt sich zum Teil durch die Verteilung der Emotionsklassifikation auch mehrere verschiedene Klassen auf einmal erklären, da die Zieldaten lediglich one-hot encoded sind, ein Gesicht jedoch oft mehr als nur exklusiv einer einzigen Emotionsklasse zugeordnet werden kann. (z.B.: Traurigkeit und Wut teilen sich ein paar Gesichtsmerkmale)
![Curves showing loss and accuracy over the course of training.](img/Training_Curves.png)
![Confusion matrix for the Face Classifier.](img/Confusion_Matrix.png)

Die Musikemotionsklassifikation ist jedoch nicht für die gedachte Anwendung geeignet. Der loss nähert sich bei Training an 1.94 an, und der accuracy score 0.14. Eine accuracy von grob 0.14 entspricht jedoch effektiv dem Zufall. Die schlechte performance des Modells lässt sich sehr einfach auf den Mangel an Trainingsdaten zurückführen, da ein Spektrogramm sehr viele kleine details enthalten, und Musik ein sehr vielseitiges Medium in der ausdrückung von Emotionen darstellt. Es gibt zum beispiel viel Musikstücke welche man als "happy" klassifizieren könnte, welche aber vom Klang und Verlauf des Musikstücks komplett unvergleichlich sind.

Die Embedding Suche von Musikstücken durch Cosine-Similarity funktioniert in tests wie erwartet. Jedoch konnten noch keine Echtdatentests mit Embeddings durchgeführt werden, da wie vorher schon erwähnt das Musikemotionsklassifikationsmodell keinen gebrauchbaren Trainingsstandt erreicht hat.


## Reflexion


## Literaturverzeichnis (APA7)
Halliday, D., Resnick, R. & Walker, J. (2003). Physik. Wiley-VCH.

Emotion detection. (2020, 11. Dezember). Kaggle. https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

Pixabay, a Canva Germany GmbH brand. (o. D.). Royalty-free music. Pixabay. Abgerufen am 30. März 2025, von https://pixabay.com/music/

Costa, Y. M., Oliveira, L. S. & Silla, C. N. (2016). An evaluation of Convolutional Neural Networks for music classification using spectrograms. Applied Soft Computing, 52, 28–38. https://doi.org/10.1016/j.asoc.2016.12.024
