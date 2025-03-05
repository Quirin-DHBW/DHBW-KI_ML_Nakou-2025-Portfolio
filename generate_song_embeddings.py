"""
This file generates a .json containing the various songs and their respective
CNN generated emotion embedding. This can then be used in main.py to search for
a Song!
"""

################
## IMPORTS #####
################
print("Importing dependencies...", end="")

import os
import sys
os.chdir(sys.path[0])

import pathlib
from tempfile import TemporaryDirectory
import json

from Music_preprocessor import audio_to_spectrogram, process_directory

import tensorflow as tf
from tensorflow import keras as k

import numpy as np
print("Done!")


##############
## STUFF #####
##############

"""
1. Turn the songs to index into images
2. Feed them to the CNN
3. Record {<song_path> : <embedding>} in Json
"""

input_dir = "Data/audio/RAW/surprised"

print("Loading music embedding model...", end="")

music_model = k.models.load_model('face_emotion_classifier.h5')

#music_model.summary()

print("Done!")


with TemporaryDirectory() as tmp_dir:
    print("Converting Songs to images...", end="")

    process_directory(input_dir, tmp_dir)

    processed_songs = k.preprocessing.image_dataset_from_directory(
            tmp_dir,
            image_size=(48, 48),#(1407, 1025),
            batch_size=8,
            color_mode="grayscale",
            label_mode=None
        )
    
    print("Done!")
    print("Generating embeddings...", end="")

    embeddings = music_model.predict(processed_songs)

    print("Done!")
    print("Writing results to file...", end="")

    res = {}
    temp_images = pathlib.Path(tmp_dir).glob('*.png')

    for i, audio_image_file in enumerate(temp_images):
        res[f"{input_dir}/{pathlib.Path(audio_image_file).stem}.mp3"] = embeddings[i].tolist()
    
    with open("song_embeddings.json", "w") as file:
        json.dump(res, file)
    
    print("Done!")

