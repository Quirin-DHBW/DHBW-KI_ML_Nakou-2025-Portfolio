# VibeluX
*(Vibe + Flux)*

Detection of emotional states in human faces, to suggest fitting music.

Most recent scanned face:\
![Small greyscale image of the last face the main.py scanned before the last push.](zoomed_face.png)

### Zur Ausarbeitung hier klicken: [Ausarbeitung.md](Report/Ausarbeitung.md)

## Basic Idea

Conv Net uses images of a face to detect the emotion.
Some secondary system uses this emotional data to suggest a song.
![Picture showing the initial idea behind VibeluX, sketched on a smart-whiteboard in the DHBW Lörrach.](img/Tafelbild_0.png)


## Requirements

- Numpy
- Pandas
- Tensorflow
- matplotlib
- librosa
- scikit-image
- OpenCV


## Files & Directories

`/Data` - Contains Training Data\
`/Data/archive` - Contains Image training data in sorted subdirectories for the face emotion classifier\
`/Data/audio` - Contains sorted subdirectories for the Audio, and preprocessed audio used to train the music emotion classifier\
`/img` - Contains images for the README\
`/Model Training` - Contains the files used to prepare training data, and train the models

`main.py` - Run this to see the project in action! :O\
`README.md` - You are here.\
`Progress_Notes.md` - Notes that were taken during the Project's lifetime\
`generate_song_embeddings.py` - Take a bunch of mp3 files in a folder, and process them for use in main.py\
`song_embeddings.json` - Contains the embeddings and their associated songs generated by generate_song_embedings.py for use in main.py\
`webcam_face_recognition.py` - Provides functions for interfacing with webcams for use in main.py\
`zoomed_face.png` - The most recent face that was captured using webcam_face_recognition.py for use in main.py\
`face_emotion_classifier.h5` and `music_emotion_classifier.h5` - Trained models for use in main.py

`/Model Training/Music_preprocessor.py` - Prepares .mp3 files in the /Data/audio/RAW directories for use in training\
`Train_face_emotion_classifier.py` - Trains the face_emotion_classifier.h5 model using the data in /Data/archive\
`Train_music_emotion_classifier.py` - Trains the music_emotion_classifier.h5 model using the data in /Data/audio/Processed


## A!
```
        /\/\    _
       < o  \  /_|
        |  | \/ /
        |/\|<__/
```

