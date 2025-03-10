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

from tempfile import TemporaryDirectory
import json

from Model_Training.Music_preprocessor import process_directory

import tensorflow as tf
from tensorflow import keras as k

import pathlib
import numpy as np
import cv2
print("Done!")


##############
## STUFF #####
##############

"""
1. Turn the songs to index into images
2. Feed them to the CNN
3. Record {<song_path> : <embedding>} in Json
"""

input_dir = "Songs"

print("Loading music embedding model...", end="")

music_model = k.models.load_model('music_emotion_classifier.h5')

#music_model.summary()

print("Done!")

with TemporaryDirectory() as tmp_dir:
    print("Converting Songs to images...", end="")

    process_directory(input_dir, tmp_dir)
    
    print("Done!")

    print("Generating Embeddings...", end="")

    res = {}
    """
    temp_images = pathlib.Path(tmp_dir).glob('*.png')
    image_values = []

    for audio_image_file in temp_images:
        img = cv2.imread(audio_image_file)[..., :1]
        image_values.append(img)
    
    image_values = np.array(image_values)
    """
    image_values = k.preprocessing.image_dataset_from_directory(
        tmp_dir,
        image_size=(1407, 1025),
        batch_size=32,
        color_mode="grayscale",
        label_mode=None
    )

    recognized_emotions = music_model.predict(image_values)

    temp_images = pathlib.Path(tmp_dir).glob('*.png')
    for i, audio_image_file in enumerate(temp_images):
        res[f"{input_dir}/{pathlib.Path(audio_image_file).stem}.mp3"] = recognized_emotions[i].tolist()
    
    print("Done!")
    print("Saving results...", end="")
    
    with open("song_embeddings.json", "w") as file:
        json.dump(res, file, indent=True)
    
    print("Done!")

