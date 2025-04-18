# VibeluX
Tim Schacht, Quirin Barth


## Einleitung

Wie aufwendig ist es, ein Maschine-Learning Modell zu trainieren, das einem Musikstück die beim Anhören ausgelösten Emotionen zuweisen kann?
Aus dieser Grundfrage heraus wurde ein Projektkonzept entwickelt, um mehr über diese Frage herauszufinden.
Als Ziel des Projekts wurde festgelegt, zwei Convolutional Neural Networks zu trainieren, die jeweils ihren gegebenen Input in eine von sieben Emotionen klassifizieren:
Wut, Ekel, Angst, Freude, Neutralität, Trauer und Überraschung.
Eines der Modelle wird dabei darauf trainiert, Gesichter zu erkennen, während das andere darauf trainiert wird, dieselben Emotionen in Musik zu erkennen. Die Vorhersagen beider Modelle können dann verwendet werden, um beispielsweise in einer App einem Nutzer Musik basierend auf seinem durch die Kamera erkannten aktuellen emotionalen Zustand zu empfehlen. Eine weitere Anwendungen wären versteckte interne Features, die eine Musik-App nutzen könnte, um ihren Shuffle-Algorithmus subtil an die Stimmung des Nutzers anzupassen.
Die Entwicklung einer solchen Endnutzer-Anwendung ist jedoch nicht Teil des Projektrahmens. Dieser umfasst lediglich das Aufstellen und Training beider KI-Modelle, die dazu notwendige Datenerfassung sowie die Entwicklung einer Proof-of-Concept-Lösung, mit der der Klassifikationserfolg der Modelle erkennbar gemacht werden kann, indem durch Kamerazugriff die Echtzeit-Emotionen einer Person erkannt und daraus ein Songvorschlag generiert werden soll. 


## Theorie

### Convolutional Neural Network
Ähnlich wie Multilayer Perceptrons basieren Convolutional Neural Networks, im folgenden CNN genannt, lose auf der menschlichen Wahrnehmung, indem sie mithilfe von "Filtern" auf einem Bild zunächst kleinere, grobe Merkmale erkennen und anschließend auf diesen basierende, größere Merkmale erfassen können. Diese Filter sind im Wesentlichen ein über das Bild gleitendes Fenster, das nach bestimmten Mustern sucht und ausgibt, wie stark dieses Muster vorhanden ist oder nicht. Die resultierende Karte mit Musterstärken kann dann an die folgende CNN-Schicht weitergegeben werden, um größere zusammenhängende Muster zu erkennen. Oft werden dabei viele Filter auf einmal verwendet, da jeder Filter jeweils nur ein Merkmal suchen kann. Die Größe der Filter kann auch angepasst werden, und dies wird oft der "Kernel" genannt.
Mathematisch entspricht dieser Prozess der Funktionsfaltung, da die Daten nach bestimmten Regeln über sich selbst "gefaltet" werden, was zu einer kleineren Ausgabe führt. Die Informationsdichte im Bild ändert sich jedoch basierend auf der Anzahl der Filter. Genauso wie in einem RGB-Bild die Merkmale "Rot", "Grün", "Blau" in drei Schichten erfasst werden, werden die resultierenden Merkmale, die von den Filtern erkannt werden, ebenso in Schichten erfasst. Dadurch kann sich die Datenmenge oft stark erhöhen, wobei das Bild von der Kernelgröße abhängig gequetscht wird.  Ein Beispiel ist ein 8x8 Pixel Graustufen-Bild, welches die Dimension (8 x 8 x 1) hat. Jetzt werden 4 Filter mit jeweils Kernelgröße (2x2) auf dieses Bild angewandt. Das resultierende Bild hat die Dimension (7 x 7 x 4), also 4 Merkmalkanäle, aber nur noch 7x7 Pixel tatsächliche Bildbreite. (3Blue1Brown, 2022)

![Beispieldarstellung eines Convolution Durchlaufs](../img/Convolution_Beispiel.png)
##### Abbildung 1: Beispieldarstellung eines Convolution Durchlaufs, Eigendarstellung

### Maxpooling
Da die Anzahl an Datenpunkten in einem Bild sehr schnell (quadratisch mit der Bildgröße) anwächst und durch die Faltungsoperationen mit mehreren Filtern diese Informationen oftmals sogar vervielfältigt werden, lohnt es sich, die Datenmengen durch bestimmte Methoden begrenzt zu behalten. Maxpooling teilt ein Bild in nicht-überlappende Blöcke bestimmter Größe, oft 2x2 Pixel große Abschnitte, und weist diesen dann jeweils den Maximalwert unter den im Block beinhalteten Zellen zu. Dadurch wird die Datenmenge in diesem Beispiel effektiv gevierteilt, ohne die am wahrscheinlich wichtigsten Merkmale zu verlieren. Es gibt natürlich auch andere Pooling-Methoden wie zum Beispiel Averagepooling, jedoch ist Maxpooling sehr weit verbreitet. Normalerweise findet man solche Maxpooling-Layer nach jedem Convolutional-Layer in einem CNN, bevor der Output an den nächsten CNN Layer weitergegeben wird (Zafar et al., 2022).

![Beispieldarstellung eines Maxpooling Durchlaufs](../img/Beispiel_Maxpooling.png)
##### Abbildung 2: Beispieldarstellung eines Maxpooling Durchlaufs, Eigendarstellung

### Dropout
Dropout deaktiviert während des Trainings zufällig Neuronen, beziehungsweise Outputs, in einem Netzwerk und skaliert die Ausgaben so, dass sie immer noch korrekt zurückpropagiert werden können. Das Modell muss dadurch lernen, sein Wissen breiter im Netzwerk zu verteilen und sich nicht auf nur eine kleine Anzahl von Neuronen zu verlassen. Die Trainingsdauer wird dadurch zwar oft erhöht, da das Modell jetzt einem weiteren Faktor entgegenwirken muss, jedoch sorgt dies auch für ein insgesamt besser angepasstes Modell. Dies ist eine Regularisierungsmethode, die Überanpassung (englisch: Overfitting) reduziert und das Modell dazu zwingt, relevantere Merkmale zu lernen, die auf mehr Neuronen verteilt werden.

### Spektrogramm
Bei einem Spektrogramm handelt es sich um eine visuelle Darstellung der zeitabhängigen Frequenzanteile eines Signals sowie ihrer jeweiligen Intensität (Halliday et al., 2003). In der Audioverarbeitung können mit ihnen akustische Merkmale sichtbar gemacht werden, die für das menschliche Ohr nicht direkt erkennbar sind. Typischerweise werden Spektrogramme durch die Short-Time Fourier Transformation erzeugt. In diesem Prozess wird das Signal in überlappende Zeitintervalle aufgeteilt und auf jedes anschließend eine Fourier-Transformation angewandt. Die Fourier-Transformation überführt dabei eine zeitabhängige Signalfunktion in eine komplexwertige Funktion, die direkt von der Zeit und der Frequenz des Ursprungssignals abhängt.
Dadurch wird es möglich, die Frequenzanteile in einem Diagramm gegen die zeitliche Signalentwicklung aufzutragen.
Um die Darstellung einheitlicher zu gestalten, wird statt der Fourier-Transformierten, in der die Signalamplitude periodisch das Vorzeichen wechselt, in der Regel das Betragsquadrat der Fourier-Transformierten Funktion aufgetragen, welche auf diese Weise die frequenz- und zeitabhängige, immer positive Signalintensität in einem dreidimensionalen Diagramm darstellt.
Dieses kann zum Beispiel in Form einer Heat-Map als Bild gespeichert werden, wie es auch von den meisten Spektrogramm-erzeugenden Algorithmen umgesetzt wird.

### Embeddings
Ein Embedding bezeichnet einen Vektor von Größe n, welcher Merkmale festhält. Diese sind oft sehr groß und die Merkmale, die von ihnen erfasst werden, werden von einem dazugehörigen Modell normalerweise selbst gelernt. Solche Embeddings sind sehr gut darin, Komplexe Dinge numerisch und vor allem für Maschinelles Lernen einfach verwendbar darzustellen, weshalb Embeddings in vielen Anwendungsbereichen zu finden sind, wie zum Beispiel Large Language Modelle.

### Cosine-Similarity
Cosine-Similarity ist eine Methode zum Vergleichen der Ähnlichkeit zwischen zwei Vektoren. Diese bestimmte Vektorenvergleichs-Methode hat sich besonders im Feld der Embeddingsuche etabliert, und ist in vielen Vektordatenbanken als eine Standardeinstellung verfügbar wie zum Beispiel ChromaDB und Pinecone (Configure - Chroma Docs, o. D.).

## Umsetzung

### Datenstruktur des Repositories
Die Verzeichnisstruktur von VibeluX setzt sich folgendermaßen zusammen:

**Data**: Enthält die Verzeichnisse für die Traingsdaten.
  - **archive**: Besteht aus den Unterordnern `test` und `train`, die jeweils PNG-Dateien für die sieben verschiedene "Haupt-Emotionen" (angry, disgusted, fearful, happy, neutral, sad, surprised) enthalten.
  - **audio**: Beinhaltet die Ordner `Processed`, für die Spektrogramm-Daten transformierter Songs, und `RAW`, für die MP3-Dateien derselben Songs.

**Model_Training**: Enthält die folgenden Dateien und Skripte:
  - `Music_preprocessor.py`: Ein Skript zur Umwandlung von MP3-Dateien in normierte Spektrogramme.
  - `Train_face_emotion_classifier.py`: Ein Skript für das Training eines neuronalen Netzwerks zur Gesichtsemotionserkennung.
  - `Train_music_emotion_classifier.py`: Ein Skript für das Training eines neuronalen Netzwerks zur Musik-Emotions-Zuordnung.

**generate_song_embeddings.py**: Erstellt eine JSON-Datei, die den Songdateinamen den durch ein CNN generierten Emotionen zuordnet.

**Songs**: Die Musikstücke welche von `generate_song_embeddings.py` Embedded werden, und daraufhin in `main.py` als findbare Musikstücke auftauchen.

**webcam_face_recognition.py**: Ein Skript für die Einbindung der Kamera des VibluX ausführenden Gerätes inklusive Gesichtserkennung und Bildformat-Normierung mittels OpenCV und der dazugehörenden Python-Anbindung durch das `cv2`-Moduls.

**main.py**: Das Hauptskript zur Durchführung des gesamten Prozesses der Gesichtsklassifikation, und anschließenden Embeddingsuche nach einem passenden Musikstück.

### Datensammlung
Im ersten Schritt wurde eruiert, welche Daten für die Umsetzung einer Emotionszuordung zu Gesichtern und Musikstücken nötig sind.
Dabei ergab sich, dass für die Gesichtsdaten ein Datensatz benötigt wird, in dem frontal aufgenommene Nahaufnahmen von Gesichtern enthalten sind, vorklassifiziert nach den jeweils dargestellten Gesichtsemotionen.
Ein solcher Datensatz konnte in der Online-Community Kaggle gefunden werden, in der er zur freien Weiterverarbeitung zur Verfügung gestellt wurde (Emotion Detection, 2020).
Durch die Wahl dieses Datensatzes waren aufgrund seiner Struktur somit die Anzahl an erkennbaren Emotionen auf 7 festgelegt.

### Gesichts-Klassifikations-Datensatz
Der Kaggle-Datensatz enthält 48x48 Pixel große Graustufen-Bilder, welche auf das zu erkennende Gesicht in Nahaufnahme zugeschnitten sind. Der Datensatz ist aufgeteilt in einen Trainingsdatensatz mit insgesamt 28709 Bildern und einen Validierungsdatensatz mit 7178 Bildern. Die Bilder sind nicht gleichmäßig über die sieben Emotionsklassen verteilt, also haben manche Emotionsklassen mehr Trainingsdaten als andere. In der folgenden Tabelle sind die prozentualen Anteile der einzelnen Emotionsklassen im Datensatz angegeben:

|Emotionsklasse|Prozent v. Trainingsdatensatz|
|-:|:-|
|angry|13.9%|
|disgusted|1.5%|
|fearful|14.3%|
|happy|25.1%|
|neutral|17.3%|
|sad|16.8%|
|surprised|11%|

##### Tabelle 1: Prozentualen Anteilen der einzelnen Emotionsklassen in Trainingsdatensatz.

Wie hier zu sehen ist, hat die Emotionsklasse "happy" mit über 25% bei weitem die meisten Trainingsdaten. "disgusted" nimmt wiederum nur 1.5% des Trainingsdatensatzes ein, wodurch es wahrscheinlich weitaus schlechter erkannt werden wird.

### Musik-Klassifikations-Datensatz
Die Musikdateien für das Training des Musik-Emotions-Klasifikations-Modells wurden von der lizenzfreien Musikplattform Pixabay bezogen (Pixabay, a Canva Germany GmbH brand, o. D.). Auf dieser Seite können Musikstücke mit Tags versehen werden, nach denen getaggte Musikstücke anschließend über eine Suchfunktion identifiziert werden können. Für die Erstellung des Musikdatensatzes wurde daher nach den 7 erkennbaren Emotionen und Synonymen dieser (happy, upbeat, joy, usw.) gesucht und dabei erhaltene Suchergebnisse in den Datensatz aufgenommen und in die Ordernstruktur in `audio/RAW` eingeordnet. Es wurden für jede Emotionsklasse 80 Musikstücke herausgesucht, woraus sich eine Gesamtheit von 560 Musikstücken in dem erstellten Datensatz ergab. Zusätzlich zu bemerken ist das der Musikdatensatz keine Validierungsdaten enthält, da die Datensatzgröße zu klein ist.

### Modelltraining

#### Gesichts-Klassifikations-Training
Für das Aufstellen und Trainieren eines CNN-Modells zur Emotionszuordnung zu Gesichtern wurde `Train_face_emotion_classifier.py` erstellt.
Dort werden zunächst die erforderlichen Bibliotheken importiert, darunter `TensorFlow` und `Keras` für das neuronale Netzwerk, `NumPy` für numerische Berechnungen sowie `Matplotlib` für die spätere Visualisierung der Trainingsergebnisse.
Anschließend werden die zwei Grundfunktionen des Programms, `create_dataset` und `create_model` definiert.

```python
def create_dataset(directory, batch_size=64, image_size=(48, 48)):
    return k.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
        label_mode="int"
    )


def create_model(conv_layers, dropout=0.25, input_size=(48, 48, 1)):
    """
    Create a Convolutional network :)
    """
    model = k.Sequential()
    model.add(k.Input(input_size))
    # Normalize pixels
    model.add(k.layers.Rescaling(1./255))
    
    for filters, kernel_size in conv_layers:
        model.add(k.layers.Conv2D(filters, kernel_size, activation='relu', padding='same'))
        model.add(k.layers.MaxPooling2D((2, 2), padding="same"))
        model.add(k.layers.Dropout(dropout))
    
    model.add(k.layers.Flatten())

    model.add(k.layers.Dense(128, activation='leaky_relu'))
    model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense(64, activation='leaky_relu'))
    model.add(k.layers.Dropout(dropout))

    model.add(k.layers.Dense(7, activation='softmax'))

    model.compile(optimizer='adamw',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
```

Die erste lädt Bilddaten aus einem angegebenen Verzeichnis, konvertiert sie in Graustufen-Bilder mit einer Größe von 48x48 Pixeln und gibt ein in Batches unterteiltes Tensorflow-Dataset zurück, das für die Bearbeitung von großen Bildmengen optimiert ist.
Die zweite Funktion definiert das verwendete CNN für das Training.
Dabei werden zunächst die Pixelwerte umdimensioniert und anschließend durch mehrere Convolutional- und Pooling-Schichten gegeben, ergänzt um Dropout-Layers für die Verringerung der Overfitting-Wahrscheinlichkeit.
In diesen Schichten werden die räumlichen Merkmale des Bildes extrahiert.
Um den Output dieser Schichten für die darauffolgende Klassifikation vorzubereiten, wird daran ein Flatten-Layer angebunden.
Diesem folgen drei vollständig verbundene Dense-Layer mit abnehmender Neuronenzahl für die Klassifikation der zuvor extrahierten Merkmale, was in einer Softmax-Ausgabe mit den sieben klassifizierten Emotionsklassen resultiert.
Diesen Funktionen folgt die Durchführung des Trainings.
Dazu werden zunächst Trainings- und Test-Datensatz aus den angegebenen Verzeichnissen geladen.
Anschließend wird das neuronale Netzwerk mit vier Convolutional-Layers unterschiedlicher Filtergröße initialisiert, um mehr verschiedene Merkmale aus den Bildern zu extrahieren.
Für den Fall, dass sich der Validations-Loss-Wert durch das Training nicht genügend verbessert, wird ein frühzeitiger Abbruchsmechanismus definiert und anschließend wird das Model für 100 Epochen trainiert.
Nach erfolgreichem Training wird die Modelleistung durch die zwei Visualisierungsfunktionen `plot_history` und `plot_confusion_matrix` analysiert. Die daraus resultierenden Visualisierungen finden sich in [Abbildung 3](#abbildung-3-accuracy--und-loss-verlauf-über-die-trainierten-epochen) und [Abbildung 4](#abbildung-4-confusion-matrix-des-gesichtsklassifikations-modells-die-label-0-6-entsprechen-angry-disgusted-fearful-happy-neutral-sad-surprised) wieder, auf die im Kapitel Ergebnisse und Diskussion genauer eingegangen wird.
Das trainierte Modell wird schlussendlich als `face_emotion_classifier.h5` gespeichert, um es später in `main.py` verwenden zu können.

#### Musikdaten-Aufbereitung
In `Music_preprocessor.py` werden die vorklassifizierten MP3-Dateien aus dem Ordner `RAW` in der Funktion `audio_to_spectrogram` jeweils als 90 Sekunden lange Ausschnitte eingelesen und mittels des Python-Moduls `librosa` in 8k Bitrate geladen. Per Short-Time Fourier-Transformation werden sie anschließend in Spektrogramme umgewandelt und in normierter Form im Ordner `Processed` gespeichert.

```python
def audio_to_spectrogram(audio_path:str, output_path:str, sample_rate:int=8000, sample_len_sec:int=90) -> None:
    y, sr = librosa.load(audio_path, sr=sample_rate)
    y = librosa.util.fix_length(y, size=sample_rate * sample_len_sec)

    # This bit makes the spectrogram
    D = librosa.stft(y) # Fourier Magic
    img = librosa.amplitude_to_db(np.abs(D), ref=np.max) # Spectrogram.

    skimage.io.imsave(output_path, img.astype(np.uint8))


def process_directory(directory_path:str, output_directory:str) -> None:
    for audio_file in pathlib.Path(directory_path).glob('*.mp3'):
        audio_to_spectrogram(audio_file, f"{output_directory}/{pathlib.Path(audio_file).stem}.png")
```

Dies basiert auf einer schon in der Vergangenheit erfolgreich angewandten Methode aus unterschiedlichen Papern, unter anderem Costa et al. (2016).

#### Musik-Klassifikations-Training
`Train_music_emotion_classifier.py` trainiert ein weiteres Modell zur Zuordnung von Spektrogrammen zu Emotionen. Im Vergleich zu dem Gesichts-Emotions-Klassifikations-Modell ist das Musik-Emotions-Klassifikations-Modell beinahe identisch, denn nur die Anzahl an CNN Layers sowie die Filteranzahl und Kernelgröße wurden erhöht, um der Bildgröße der Spektrogramme nachzukommen. Dazu wurden auch die Inputgrößen von `create_dataset()` und `create_model()` an die der Spektrogramme angepasst.
Das trainierte Modell wird zu Programmende als `music_emotion_classifier.h5` gespeichert.

### Prozessverlauf des Prototyps

#### Gesichtserkennung
Über `webcam_face_recognition.py` wird auf die primäre Gerätekamera zugegriffen, um ein Gesicht zu erfassen. Diese wird zu einem Graustufen-Bild konvertiert, und anschließend auf die richtige Größe von 48x48 Pixel zugeschnitten, und in `zoomed_face.png` abgelegt. Dieser Prozess wird in der Funktion `caputure_and_save_face()` bereitgestellt, um in `main.py` verwendet zu werden.

```python
def capture_and_save_face(visualize:bool=False, verbose:bool=False):
    if verbose:
        print("CAPTURE WAS CALLED!")

    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("WEBCAM ERROR: NO FRAME WAS RETURNED!!!")
        return
    
    if verbose:
        print("Greyscaling image...")
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if verbose:
        print("Detecting face...")
    # Detect all faces in the frame
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Find largest detected face
    largest_face = None
    max_area = 0
    for (x, y, w, h) in faces:  # x,y = start coordinates; w,h = rectangle
        area = w * h
        if area > max_area:
            max_area = area
            largest_face = (x, y, w, h)

    # If face biggest, make even biggerer
    if largest_face:
        x, y, w, h = largest_face
        # Q: I set the margin to zero to more closely resemble the training data
        margin = int(0 * w)  # Add margin for good measure - should be enough for emotion detection like this
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(grey.shape[1], x + w + margin), min(grey.shape[0], y + h + margin)
        
        # Standalone face and pixel-inator
        zoomed_face = grey[y1:y2, x1:x2]
        zoomed_face = cv2.resize(zoomed_face, (48, 48))  # Resize for consistency

        if verbose:
            print("Saving face...")
        # Save the frame
        cv2.imwrite("zoomed_face.png", zoomed_face)

    # Show the original frame with rectangles around faces - for testing
    if visualize:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Webcam Feed", frame)
```

### Emotions-Embedding-Suche
Beide Modelle erzeugen einen emotionalen "Vektor", der das Gesicht oder die Musik einer Emotion zuordnet. Musik hat viele Facetten, und anstatt nur die primäre Emotion zu erkennen und passende Musikstücke auszuwählen, haben wir stattdessen die Cosine-Similarity verwendet, um das Musikstück oder die Musikstücke zu finden, die am besten zu allen 7 erkannten Emotionen im Gesicht passen.

```python
best_song = ""
best_score = 0

with open("song_embeddings.json", "r") as embed_json:
    song_embeddings = json.load(embed_json)
    for song, embed in song_embeddings.items():
        cos_sim = (np.dot(recognized_emotions[0], embed) / (np.linalg.norm(recognized_emotions[0]) * np.linalg.norm(embed)))
        #print(song, cos_sim)
        if cos_sim > best_score:
            best_song = song
            best_score = cos_sim
```

#### song_embeddings.json
Anhand der JSON-Datei `song_embeddings.json` werden die erkannten Emotionen mit den gespeicherten Song-Embeddings verglichen, um den am besten passenden Song später identifizieren zu können. Hierbei folgen die Einträge dem Schema eines Dictionarys, in welchem der Dateipfad der Key ist und der dazu gespeicherte Wert der jeweilige Embedding Vektor.

### Main-Programmausführung
Anschließend wurde das `main.py` Skript erstellt, in dem die endgültige Programmausführung stattfindet.
In diesem werden zunächst alle erforderlichen Bibliotheken importiert, sowie eine Hilfsfunktion aus `webcam_face_recognition.py` zum Erfassen eines Gesichts. Anschließend wird zuerst das Gesichts-Emotions-Klassifikations-Modell geladen, und anschließend ein Bild über die Webcam des Benutzers aufgenommen. Dieses Bild wird daraufhin mithilfe des Gesichts-Emotions-Klassifikations-Modells zu einem Emotionsvektor umgewandelt.
Mithilfe dieses Vektors wird durch Cosine-Similarity auf diese Weise in der `song_embeddings.json` Datei nach einem am besten passenden Musikstück gesucht.

## Ergebnisse und Diskussion
Die Gesichtserkennung funktioniert gut genug für unsere Verwendungszwecke, trotz eines niedrigen Accuracy Score von 0.545, wie in [Abbildung 3](#abbildung-3-accuracy--und-loss-verlauf-über-die-trainierten-epochen) zu erkennen. Der niedrige Score lässt sich zum Teil durch die Verteilung der Emotionsklassifikation auf mehrere verschiedene Klassen auf einmal erklären, da die Zieldaten lediglich One-Hot-Encoded sind, ein Gesicht jedoch oft mehr als nur exklusiv einer einzigen Emotionsklasse zugeordnet werden kann, wie in [Abbildung 4](#abbildung-4-confusion-matrix-des-gesichtsklassifikations-modells-die-label-0-6-entsprechen-angry-disgusted-fearful-happy-neutral-sad-surprised) zu sehen ist. (z.B.: Trauer und Wut teilen sich ein paar Gesichtsmerkmale)
![Curves showing loss and accuracy over the course of training.](../img/Training_Curves.png)
##### Abbildung 3: Accuracy- und Loss-Verlauf über die trainierten Epochen. 
![Confusion matrix for the Face Classifier.](../img/Confusion_Matrix.png)
##### Abbildung 4: Confusion Matrix des Gesichtsklassifikations-Modells, die Label 0-6 entsprechen: [angry, disgusted, fearful, happy, neutral, sad, surprised]

Die Musik-Emotions-Klassifikation in der bisher umgesetzten Form ist allerdings nicht für die gedachte Anwendung geeignet. Der Loss-Wert nähert sich beim Training an 1.94 an, und die Accuracy an 0.14. Eine Accuracy von grob 0.14 entspricht jedoch dem Erwartungswert eines gleichverteilten Zufallsexperiments mit 7 Ereignissen von 1/7 und damit effektiv dem Zufall. Die schlechte Performance des Modells lässt sich sehr einfach auf den Mangel an Trainingsdaten zurückführen, da die in einem Spektrogramm enthaltene Informationsdichte sehr hoch ist, wodurch die Größe des erhobenen Datensatzes für das Modell nicht ausreicht, um Kernmerkmale erlernen zu können. Außerdem ist Musik ein sehr vielseitiges und subjektives Medium in dem Ausdruck von Emotionen. So gibt es zum Beispiel viel Musikstücke, welche man als "happy" klassifizieren könnte, welche aber vom Klang und Verlauf des Musikstücks nicht ähnlich zueinander sind, was die Merkmalerkennung zusätzlich erschwert.

Die Embedding Suche von Musikstücken durch Cosine-Similarity funktioniert in Tests wie erwartet. Jedoch konnten noch keine Echtdaten-Tests mit Embeddings durchgeführt werden, da das Musik-Emotions-Klassifikations-Modell keinen verwendbaren Trainingsstand erreichte.

## Reflexion und Ausblick
Für eine Weiterführung des Projektes und die endgültige Beantwortung der anfangs gestellten Leitfrage muss in jedem Fall der Datensatz für die Musikklassifikation stark erweitert werden.
Dabei können leider auch keine anderen der Standard-Methoden für die Vervielfältigung von Datensätzen wie Rotation, Spiegelung, oder Mutation der schon vorhandenen Daten zu Rate gezogen werden, da die Spektrogramme durch ihre Informationsdichte extrem empfindlich auf jegliche Manipulation reagieren und somit essenzielle Merkmale verloren gehen würden.

Da alle sonstigen Grundfunktionen im Projekt umgesetzt werden konnten, könnte mit mehr Daten ein erfolgreiches Training des Musik-Emotions-Klassifikations-Modells durchgeführt werden, und von dem Proof-of-Concept direkt zu einer In-App-Lösung übergegangen werden.
Dieser Schritt wird jedoch vermutlich noch weit entfernt sein, da lizenzfreie und mit für diese Anwendung passenden Tags versehene Musikstücke in großer Menge schwer zu finden sind.

## Literaturverzeichnis

3Blue1Brown. (2022, 18. November). But what is a convolution? [Video]. YouTube. https://www.youtube.com/watch?v=KuXjwB4LzSA

Zafar, A., Aamir, M., Nawi, N. M., Arshad, A., Riaz, S., Alruban, A., Dutta, A. K. & Almotairi, S. (2022). A Comparison of Pooling Methods for Convolutional Neural Networks. Applied Sciences, 12(17), 8643. https://doi.org/10.3390/app12178643

Halliday, D., Resnick, R. & Walker, J. (2003). Physik. Wiley-VCH.

Configure - Chroma Docs. (o. D.). https://docs.trychroma.com/docs/collections/configure

Emotion detection. (2020, 11. Dezember). Kaggle. https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer/data

Pixabay, a Canva Germany GmbH brand. (o. D.). Royalty-free music. Pixabay. Abgerufen am 30. März 2025, von https://pixabay.com/music/

Costa, Y. M., Oliveira, L. S. & Silla, C. N. (2016). An evaluation of Convolutional Neural Networks for music classification using spectrograms. Applied Soft Computing, 52, 28–38. https://doi.org/10.1016/j.asoc.2016.12.024
