"""
VibeluX

Eat image, return emotion, use emotion to find songs :3
"""

################
## IMPORTS #####
################

print("Importing dependencies...", end="")

import os
import sys
os.chdir(sys.path[0])

import cv2
from webcam_face_recognition import capture_and_save_face, cleanup

import tensorflow as tf
from tensorflow import keras as k

import numpy as np
import json

import matplotlib.pyplot as plt

emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]

print("Done!")


##############
## STUFF #####
##############

print("Loading face recognition model...", end="")

face_model = k.models.load_model('face_emotion_classifier.h5')

#face_model.summary()

print("Done!")

print("Recording face...", end="")

capture_and_save_face()

img_raw = cv2.imread("zoomed_face.png")[..., :1]
#print(img, img.shape)

cleanup()

print("Done!")
print("Detecting emotion...", end="")

img = np.reshape(img_raw, (-1, 48, 48, 1))
recognized_emotions = face_model.predict(img)

print("Done!\n")

for val, emot in zip(recognized_emotions[0], emotions):
    print(f"{emot.capitalize().ljust(10)}: {float(val):.5f}")

print(f"\nDominant Emotion: {emotions[recognized_emotions[0].argmax()]}")


print("Visualizing...", end="")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1.2]})

# Left: imshow plot
im = axes[0].imshow(img_raw, cmap="bone", aspect='auto')
axes[0].set_title("Dein schönes Gesicht :)")

# Right: horizontal bar chart
axes[1].barh(emotions, recognized_emotions[0], color='skyblue')
axes[1].set_xlabel("Percentage")
axes[1].set_ylabel("Emotionen")
axes[1].set_xlim(0, 1)
axes[1].set_title("Erkannte Emotionen")

plt.tight_layout()
plt.show()

print("Done!\n")


print("Finding songs...", end="")

# TODO: SONG FINDING STUFF HERE
# Generic Idea: Absolute difference between vectors
best_song = ""
best_score = 0

with open("FAKE_song_embeddings.json", "r") as embed_json: # TODO - Don't forget to replace this with the real embeddings :)
    song_embeddings = json.load(embed_json)
    for song, embed in song_embeddings.items():
        cos_sim = (np.dot(recognized_emotions[0], embed) / (np.linalg.norm(recognized_emotions[0]) * np.linalg.norm(embed)))
        #print(song, cos_sim)
        if cos_sim > best_score:
            best_song = song
            best_score = cos_sim

print("Done!\n")
print(f"Found song: {best_song}")

