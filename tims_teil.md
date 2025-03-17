## Projektumfang

### Datenstruktur
Die Verzeichnisstruktur von VibeluX setzt sich folgendermaßen zusammen:

- **Data**: Enthält die Verzeichnisse `archive`, für Bild-, und `audio`, für Audio-Datein.
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

### Vorgehensweise

#### Datenvorbereitung
Zu Beginn wurden jeweils 90-Sekunden-Abschnitte der vorklassifizierten MP3-Dateien in `Music_preprocessor.py` mittels des Python-Moduls `librosa` per Fourier-Transformation in Spektrogramme umgewandelt und in normierter Form gespeichert.
Anschließend wurde `Train_face_emotion_classifier.py` verwendet, um ein CNN-Modell für die Gesichtsemotionserkennung zu trainieren.
`Train_music_emotion_classifier.py` trainierte ein weiteres Modell zur Zuordnung von Spektrogrammen zu Emotionen.

#### Main-Programmausführung
Über `webcam_face_recognition.py` wird auf die Gerätekamera zugegriffen, um ein Gesicht zu erfassen, zu verarbeiten und als Graustufenbild abzuspeichern.
Das CNN-Modell `face_emotion_classifier.h5` ermittelt die Emotion des Gesichts.
Anhand der JSON-Datei `song_embeddings.json` werden die erkannten Emotionen mit den gespeicherten Song-Embeddings verglichen, um den am besten passenden Song zu identifizieren.

TODO: auf basis der theorie genauer ausführen :3
TODO: Einleitung - ganz zum Schluss


