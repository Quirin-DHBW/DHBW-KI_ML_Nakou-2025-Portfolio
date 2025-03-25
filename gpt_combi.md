# VibeluX
Tim Schacht, Quirin Barth

## Projektumfang

### Zielsetzung
Das Ziel des Projekts VibeluX ist es, zwei Convolutional Neural Networks (CNNs) zu trainieren, die jeweils Eingaben einer von sieben Emotionen zuordnen können: Wut, Ekel, Angst, Freude, Neutral, Trauer und Überraschung.

Ein Modell wird darauf trainiert, Emotionen in Gesichtern zu erkennen, während das andere Modell Musik nach emotionalen Merkmalen klassifiziert. Basierend auf den Vorhersagen beider Modelle soll dem Benutzer eine passende Musikauswahl vorgeschlagen werden. Eine potenzielle Anwendung wäre eine verborgene Funktion innerhalb eines Musik-Streaming-Dienstes, die die Zufallswiedergabe unauffällig an die Stimmung des Nutzers anpasst.

### Datenstruktur
Die Verzeichnisstruktur von VibeluX setzt sich folgendermaßen zusammen:

- **Data**: Enthält die Verzeichnisse `archive` für Bilder und `audio` für Audiodateien.
  - **archive**: Beinhaltet die Unterordner `test` und `train`, die PNG-Dateien der sieben Hauptemotionen enthalten.
  - **audio**: Enthält `Processed` für transformierte Spektrogramm-Daten und `RAW` für MP3-Dateien.

- **Model_Training**: Beinhaltet Skripte zur Datenverarbeitung und zum Training der Modelle:
  - `Music_preprocessor.py`: Wandelt MP3-Dateien in normierte Spektrogramme um.
  - `Train_face_emotion_classifier.py`: Trainiert ein CNN zur Gesichtsemotionserkennung.
  - `Train_music_emotion_classifier.py`: Trainiert ein CNN zur Musik-Emotion-Klassifikation.

- **Weitere Skripte**:
  - `generate_song_embeddings.py`: Erstellt eine JSON-Datei mit den zugeordneten Emotionen der Songs.
  - `webcam_face_recognition.py`: Erfasst und verarbeitet Gesichter über die Gerätekamera.
  - `main.py`: Das Hauptskript zur Durchführung des gesamten Prozesses.

### Vorgehensweise
Die MP3-Dateien wurden in `Music_preprocessor.py` mit `librosa` in Spektrogramme umgewandelt. Danach wurde `Train_face_emotion_classifier.py` für das Training des Gesichtserkennungsmodells verwendet, während `Train_music_emotion_classifier.py` für die Musikklassifikation zuständig war.

Die Kamerafunktion `webcam_face_recognition.py` erfasst das Gesicht des Nutzers, das `face_emotion_classifier.h5`-Modell bestimmt die Emotion. Die JSON-Datei `song_embeddings.json` wird genutzt, um die ermittelte Emotion mit Songs abzugleichen.

## Theorie

### Convolutional Neural Networks (CNNs)
CNNs zerlegen Eingaben in kleinere Abschnitte, analysieren diese und extrahieren relevante Merkmale. Mathematisch betrachtet, „faltet“ ein CNN die Daten über sich selbst und reduziert dabei die Größe, während es relevante Informationen verdichtet.

CNNs verwenden sogenannte „Filter“, die als gleitende Fenster bestimmte Muster in Bildern erkennen. Mehrere Schichten von Filtern können zunehmend komplexere Muster extrahieren. Dies ist besonders nützlich für Bild- und Audiodaten, da Strukturen und Frequenzen so effizient analysiert werden können.

### Maxpooling und Dropout
- **Maxpooling**: Reduziert die Datenmenge, indem nur die relevantesten Merkmale beibehalten werden. Dies spart Rechenzeit und verhindert unnötige Redundanz.
- **Dropout**: Schaltet während des Trainings zufällig Neuronen ab, um Überanpassung zu verhindern und das Modell robuster zu machen.

### Datensätze
- **Gesichtsdaten**: 48x48-Graustufenbilder aus einem Kaggle-Datensatz.
- **Musikdaten**: Songs mit emotionalen Labels, die auf 90 Sekunden gekürzt und in Spektrogramme umgewandelt wurden.

Ein Spektrogramm stellt die Frequenzen eines Songs über die Zeit hinweg dar. Dies ermöglicht es, Musik als Bilddaten zu behandeln und CNNs zur Analyse zu nutzen.

### Modellarchitektur
Beide Modelle bestehen aus mehreren CNN-, Maxpool- und Dropout-Schichten, gefolgt von voll verbundenen (dense) Schichten mit abnehmender Neuronenanzahl. Die finale Schicht hat sieben Neuronen mit einer Softmax-Aktivierungsfunktion, um die Emotionen vorherzusagen.

Für das Training wurde die **SparseCategoricalCrossentropy**-Loss-Funktion verwendet, zusammen mit dem **AdamW**-Optimizer. Die Genauigkeit diente als zusätzliche Metrik zur Leistungsbewertung.

### Emotionale Embedding-Suche
Anstatt nur die stärkste Emotion zur Musikauswahl zu nutzen, verwendet das System eine **Cosine-Similarity**-Methode. Dabei wird der emotionale Vektor des erkannten Gesichts mit den gespeicherten Emotionen der Songs verglichen, um eine möglichst präzise Empfehlung zu generieren.

## Ergebnisse

### Gesichtserkennung
Das Modell zur Gesichtserkennung wurde mit TensorFlow trainiert und erreichte nach weniger als 10 Minuten eine Genauigkeit von **54,5%**. Da es sieben Klassen gibt, von denen einige schwer zu unterscheiden sind (z. B. Wut und Trauer), ist dieses Ergebnis für unseren Zweck ausreichend. Zudem unterstützt es die Idee der **Emotion-Embeddings**, bei denen Emotionen als Mischverhältnisse der sieben Klassen dargestellt werden.

### Musikklassifikation
Das Musikmodell erreichte eine Loss von **1.94** und eine Genauigkeit von **14%**, ohne sich weiter zu verbessern. Dies liegt wahrscheinlich an der geringen Datenmenge (nur 80 Songs pro Emotion). Zudem sind Musik-Spektrogramme mit etwa **1000x1000 Pixeln** wesentlich komplexer als Gesichtsbilder mit **48x48 Pixeln**. Da Musik emotionale Vielfalt besitzt, fällt es dem Modell schwer, eindeutige Muster zu identifizieren.

### Embedding-Suche
Die Cosine-Similarity-Methode funktioniert gut, aber aufgrund der schlechten Leistung des Musikmodells können keine realistischen Emotionsembeddings für Songs generiert werden. Tests mit synthetischen Daten zeigen jedoch, dass das Matching-System an sich einwandfrei arbeitet.

## Fazit
Die Gesichtserkennung liefert brauchbare Ergebnisse, während die Musikklassifikation unter zu wenigen Trainingsdaten leidet. Eine größere Datenbasis könnte die Leistung deutlich verbessern. Die Idee der Emotion-Embeddings bleibt dennoch vielversprechend und könnte mit einem besseren Musikmodell erfolgreich eingesetzt werden.

