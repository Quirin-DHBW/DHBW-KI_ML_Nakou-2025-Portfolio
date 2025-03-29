# VibeluX
Tim Schacht, Quirin Barth

## Idee/Umfang

Das Ziel ist es, zwei Convolutional Neural Networks zu trainieren, die jeweils ihren gegebenen Input in eine von sieben Emotionen klassifizieren:
Wut, Ekel, Angst, Freude, Neutral, Traurigkeit und Überraschung.
Eines der Modelle wird darauf trainiert, Gesichter zu erkennen, während das andere darauf trainiert wird, diese Emotionen in Musik zu erkennen. Die Vorhersagen beider Modelle können dann verwendet werden, um einem Nutzer Musik basierend auf seinem aktuellen emotionalen Zustand zu empfehlen. Andere Anwendungen umfassen einige versteckte interne Features, die eine Musik-App nutzen könnte, um ihren Shuffle-Algorithmus subtil an die Stimmung des Nutzers anzupassen.

Zur Erleichterung des Trainings wurde ein Datensatz von Gesichtern von Kaggle bezogen: https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

Die Musik für das Training des anderen Modells wurde von einer lizenzfreien Musikplattform bezogen: https://pixabay.com/music/

## Projektumfang

### Datenstruktur
Die Verzeichnisstruktur von VibeluX setzt sich folgendermaßen zusammen:

- **Data**: Enthält die Verzeichnisse `archive`, für Bild-, und `audio`, für Audio-Dateien.
  - **archive**: Besteht aus den Unterordnern `test` und `train`, die jeweils PNG-Dateien für die sieben verschiedene "Haupt-Emotionen" (angry, disgusted, fearful, happy, neutral, sad, surprised) enthalten.
  - **audio**: Beinhaltet die Ordner `Processed`, für die Spektrogramm-Daten transformierter Songs, und `RAW`, für die MP3-Dateien derselben Songs.

- **Model_Training**: Enthält die folgenden Dateien und Skripte:
  - `ausarbeitung.md`: die Dokumentation des Projekts. # TODO: Was macht die hier drin?
  - `Music_preprocessor.py`: Ein Skript zur Umwandlung von MP3-Dateien in normierte Spektrogramme.
  - `Train_face_emotion_classifier.py`: Ein Skript für das Training eines neuronalen Netzwerks zur Gesichtsemotionserkennung.
  - `Train_music_emotion_classifier.py`: Ein Skript für das Training eines neuronalen Netzwerks zur Musik-Emotion-Zuordnung.

- **generate_song_embeddings.py**: Erstellt eine JSON-Datei, die Songdateinamen den durch ein CNN generierten Emotionen zuordnet.

- **webcam_face_recognition.py**: Ein Skript für die Einbindung der Kamera des VibluX ausführenden Gerätes inklusive Gesichtserkennung und Bildformat-Normierung mittels `cv2`-Moduls.

- **main.py**: Das Hauptskript zur Durchführung des gesamten Prozesses.

## Theorie

### Was sind CNNs?
Mache Dinge in Stücke, schaue auf Stücke, mache kleinere Stücke, wiederhole, es kann Dinge erkennen, indem es sie zerlegt!
Mathematisch? Du "faltest" die Daten über sich selbst nach bestimmten Regeln, was zu einer kleineren Ausgabe führt, die dichter an Informationen ist, z. B. (8 x 8 x 1) -> (4 x 4 x 4)

Ähnlich wie mehrschichtige Perzeptrons basieren CNNs lose auf der menschlichen Wahrnehmung, jedoch in direkterer Weise, indem sie "Filter" auf ein Bild anwenden. Diese Filter sind im Wesentlichen ein gleitendes Fenster, das nach bestimmten Mustern sucht und ausgibt, wie stark dieses Muster vorhanden ist. Die resultierende Musterkarte kann dann an die nächste CNN-Schicht weitergegeben werden, um größere zusammenhängende Muster zu erkennen!

### Warum Maxpooling?
Quetsche die Daten, mache sie kleiner, jetzt weniger Daten mit hoffentlich noch relevanter Information -> Weniger Berechnung, yay :D

### Warum Dropout?
Dropout deaktiviert während des Trainings zufällig Neuronen oder ihre Entsprechungen in einem Netzwerk und skaliert die Ausgaben so, dass sie immer noch korrekt zurückpropagiert werden können. Dies ist eine Regularisierungsmethode, die Überanpassung reduziert und das Modell dazu zwingt, relevantere Merkmale zu lernen, die auf mehr Neuronen verteilt sind.

### Datenvorbereitung
Zu Beginn wurden jeweils 90-Sekunden-Abschnitte der vorklassifizierten MP3-Dateien in `Music_preprocessor.py` mittels des Python-Moduls `librosa` per Fourier-Transformation in Spektrogramme umgewandelt und in normierter Form gespeichert.
Anschließend wurde `Train_face_emotion_classifier.py` verwendet, um ein CNN-Modell für die Gesichtsemotionserkennung zu trainieren.
`Train_music_emotion_classifier.py` trainierte ein weiteres Modell zur Zuordnung von Spektrogrammen zu Emotionen.

### Gesichts-Klassifikations-Datensatz
48x48 Graustufen-Gesichter!

### Musik-Klassifikations-Datensatz
Lieder, die mit ihrer Emotion oder einem Synonym dieser Emotion getaggt wurden, wurden ausgewählt, alle auf 1:30 Länge zugeschnitten/gepolstert und dann in Spektrogramme umgewandelt.

### Main-Programmausführung
Über `webcam_face_recognition.py` wird auf die Gerätekamera zugegriffen, um ein Gesicht zu erfassen, zu verarbeiten und als Graustufenbild abzuspeichern.
Das CNN-Modell `face_emotion_classifier.h5` ermittelt die Emotion des Gesichts.
Anhand der JSON-Datei `song_embeddings.json` werden die erkannten Emotionen mit den gespeicherten Song-Embeddings verglichen, um den am besten passenden Song zu identifizieren.

### Emotionale Einbettungssuche
Beide Modelle erzeugen einen emotionalen "Vektor", der das Gesicht oder die Musik einer Emotion zuordnet. Musik hat viele Facetten, und anstatt nur die primäre Emotion zu erkennen und passende Lieder auszuwählen, haben wir stattdessen die Kosinus-Ähnlichkeit verwendet, um das Lied oder die Lieder zu finden, die am besten zu allen 7 erkannten Emotionen im Gesicht passen.

## Ergebnis
Gesichtserkennung? Funktioniert! Ein einfaches Netzwerk wurde in TensorFlow erstellt und in weniger als 10 Minuten trainiert und lieferte akzeptable Ergebnisse! Genauigkeit von 0,545.

Musikerkennung? Das Training nähert sich einem Verlust von 1,94 / Genauigkeit von 0,14 und verbessert sich danach nicht weiter. Dies liegt wahrscheinlich an einem Mangel an Trainingsdaten.

Einbettungssuche mittels Kosinus-Ähnlichkeit? Funktioniert in Tests, aber der Musikklassifikator funktioniert nicht richtig, sodass wir derzeit keine **echten** Einbettungen für Lieder erzeugen können.

TODO: auf Basis der Theorie genauer ausführen :3
TODO: Einleitung - ganz zum Schluss
