# VibeluX
Tim Schacht, Quirin Barth

## Projektumfang

### Zielsetzung

Das Ziel dieses Projekts ist die Entwicklung und das Training von zwei Convolutional Neural Networks (CNNs), die jeweils Eingaben in eine von sieben Emotionen klassifizieren: 
- Wut (angry)
- Ekel (disgusted)
- Angst (fearful)
- Freude (happy)
- Neutral (neutral)
- Trauer (sad)
- Überraschung (surprised)

Eines der Modelle wurde darauf trainiert, Emotionen in Gesichtern zu erkennen, das andere darauf, Emotionen in Musikstücken zu klassifizieren. Durch die Kombination der Vorhersagen beider Modelle soll es möglich sein, einem Nutzer basierend auf seinem aktuellen emotionalen Zustand Musik zu empfehlen. Eine weitere potenzielle Anwendung wäre die Integration in eine Musik-App, um deren Shuffle-Algorithmus subtil an die Stimmung des Nutzers anzupassen.

Das zur Gesichtserkennung verwendete Datenset stammt von Kaggle:
[https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data)

Die Musikstücke zur Klassifikation wurden von einer Plattform für lizenzfreie Musik bezogen:
[https://pixabay.com/music/](https://pixabay.com/music/)

### Datenstruktur

Die Verzeichnisstruktur von VibeluX ist folgendermaßen aufgebaut:

- **Data**: Enthält die Verzeichnisse `archive` für Bild- und `audio` für Audiodateien.
  - **archive**: Beinhaltet die Unterordner `test` und `train`, welche PNG-Dateien für die sieben Hauptemotionen enthalten.
  - **audio**: Besteht aus den Ordnern `Processed` für Spektrogramm-Daten der Musikstücke und `RAW` für die ursprünglichen MP3-Dateien.

- **Model_Training**: Enthält verschiedene Dateien und Skripte:
  - `ausarbeitung.md`: Dokumentation des Projekts. # TODO: Warum ist das hier?
  - `Music_preprocessor.py`: Skript zur Umwandlung von MP3-Dateien in normalisierte Spektrogramme.
  - `Train_face_emotion_classifier.py`: Skript zum Training eines neuronalen Netzwerks zur Gesichtsemotionserkennung.
  - `Train_music_emotion_classifier.py`: Skript zum Training eines neuronalen Netzwerks zur Klassifikation von Musik-Emotionen.

- **generate_song_embeddings.py**: Erstellt eine JSON-Datei, die Songtitel mit den durch das CNN generierten Emotionen verknüpft.

- **webcam_face_recognition.py**: Skript zur Gesichtserkennung über die Gerätekamera mit Bildformat-Normalisierung mittels `cv2`-Modul.

- **main.py**: Hauptskript zur Ausführung des gesamten Prozesses.

### Theorie

#### Convolutional Neural Networks (CNNs)

Ein CNN verarbeitet Daten durch schrittweises Extrahieren relevanter Merkmale. Dabei werden Eingabedaten (z. B. Bilder oder Spektrogramme) durch Faltungsoperationen in immer kleinere, aber informationsreichere Repräsentationen überführt. Ähnlich wie bei menschlicher Wahrnehmung analysieren CNNs Muster und Strukturen durch Anwendung von Filtern. Diese Filter erkennen zunächst einfache Merkmale (z. B. Kanten) und später komplexere Muster.

#### Warum Maxpooling?

Maxpooling reduziert die Dimensionalität der Daten, indem es nur die relevantesten Informationen einer Region beibehält. Dadurch wird der Berechnungsaufwand gesenkt und Overfitting reduziert.

#### Warum Dropout?

Dropout deaktiviert während des Trainings zufällig einige Neuronen und skaliert die restlichen Aktivierungen entsprechend. Dies verbessert die Generalisierungsfähigkeit des Netzwerks und verhindert, dass sich das Modell zu stark an bestimmte Trainingsdaten anpasst.

#### Gesichtsklassifikations-Datensatz

Die Gesichtsbilder sind 48x48 Pixel große Graustufenbilder.

#### Musikkategorisierungs-Datensatz

Die Musikstücke wurden nach Emotionen kategorisiert, auf 1:30 Minuten getrimmt und in Spektrogramme umgewandelt.

#### Was sind Spektrogramme?

Ein Spektrogramm visualisiert Frequenzinformationen über die Zeit. Mittels Fourier-Transformation werden die Frequenzen eines Audiosignals extrahiert und als Bild dargestellt. Da CNNs mit Bilddaten gut umgehen können, lassen sich auf diese Weise Musikstücke klassifizieren.

#### Modellarchitektur

Das Modell besteht aus mehreren CNN-Layern, die durch Maxpooling- und Dropout-Schichten ergänzt werden. Danach folgen zwei dichte Fully-Connected-Layer und eine abschließende Softmax-Ausgabe mit sieben Neuronen (für die sieben Emotionen).

Als Loss-Funktion wurde `SparseCategoricalCrossentropy` gewählt, während `AdamW` als Optimierungsverfahren genutzt wurde. `AdamW` wurde aufgrund seiner besseren Gewichtsnormierung gegenüber `Adam` bevorzugt. Zusätzlich wurde die Genauigkeit (`accuracy`) als Metrik verfolgt.

#### Emotionale Embedding-Suche

Beide Modelle geben eine emotionale Vektorrepräsentation aus. Statt einfach nur eine einzelne dominante Emotion zu verwenden, wird die Ähnlichkeit zwischen der erkannten Emotion im Gesicht und den gespeicherten Musik-Embeddings mittels Kosinus-Ähnlichkeit berechnet. Dadurch kann ein passender Song empfohlen werden.

### Ergebnisse

#### Gesichtserkennung

Das Gesichtserkennungsmodell wurde in TensorFlow implementiert und in weniger als 10 Minuten trainiert. Es erreichte eine Genauigkeit von 54,5%. Dies mag auf den ersten Blick gering erscheinen, ist aber angesichts der sieben Klassen und ihrer Ähnlichkeit (z. B. Wut vs. Trauer) akzeptabel. Dies legt außerdem nahe, dass Emotionen als Mischungen mehrerer Klassen betrachtet werden können.

#### Musikklassifikation

Das Musikklassifikationsmodell erreichte eine Loss von 1,94 und eine Genauigkeit von nur 14%, was schlechter als Zufallstreffer ist. Dies liegt wahrscheinlich an der geringen Anzahl an Trainingsdaten (nur 80 Songs pro Emotion, insgesamt 560 Songs) sowie der hohen Variabilität von Musikstücken im Vergleich zu Gesichtsausdrücken. Die aktuelle Architektur scheint nicht ausreichend, um relevante Muster aus den Spektrogrammen zu extrahieren.

#### Emotionale Embedding-Suche

Die Embedding-Suche funktioniert technisch, jedoch scheitert sie in der Praxis, da das Musikklassifikationsmodell keine sinnvollen Emotionsembeddings generieren kann. Mit künstlichen Beispiel-Embeddings funktioniert der Ansatz jedoch einwandfrei.

#### Fazit

Die Gesichtserkennung funktioniert gut, während die Musikklassifikation deutliche Schwächen aufweist. Es ist klar, dass das Modell für Musik entweder mehr Trainingsdaten oder eine tiefere Architektur benötigt. Die emotionale Embedding-Suche ist ein vielversprechender Ansatz, der mit einem funktionierenden Musikmodell effektiv eingesetzt werden könnte.

# TODO
- Theoretischen Teil ausbauen und mit Quellen versehen.
- Fehlende Literatur zu CNN-Spektrogramm-Klassifikation ergänzen.
- Einleitung am Ende schreiben.
- Lösung für das Musikmodell finden: Mehr Trainingsdaten? Alternative Architektur?

