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

img = cv2.imread("zoomed_face.png")[..., :1]
#print(img, img.shape)

print("Done!")
print("Detecting emotion...", end="")

img = np.reshape(img, (-1, 48, 48, 1))
recognized_emotions = face_model.predict(img)

print("Done!")

print()
for val, emot in zip(recognized_emotions[0], emotions):
    print(f"{emot.capitalize().ljust(10)}: {float(val)}")

print(f"\nDominant Emotion: {emotions[recognized_emotions[0].argmax()]}")

cleanup()

