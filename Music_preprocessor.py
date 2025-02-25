import os
import sys
os.chdir(sys.path[0])


################
## IMPORTS #####
################

import librosa
#import soundfile as sf # Used for the audio crustifyer!
import numpy as np
import skimage.io
import glob
import pathlib


####################
## DEFINITIONS #####
####################


def audio_to_spectrogram(audio_path:str, output_path:str, sample_rate:int=8000) -> None:
    y, sr = librosa.load(audio_path, sr=sample_rate)
    y = librosa.util.fix_length(y, size=sample_rate * 90)

    #sf.write("crusty.mp3", y, samplerate=512) # Instantly make any audio crusty.

    D = librosa.stft(y)  # STFT of y
    img = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    skimage.io.imsave(output_path, img.astype(np.uint8))


def process_directory(directory_path:str, output_directory:str) -> None:
    name = 0
    for audio_file in pathlib.Path(directory_path).glob('*.mp3'):
        audio_to_spectrogram(audio_file, f"{output_directory}/{name}.png")
        name += 1


##########################
## PROCESS ALL AUDIO #####
##########################

process_directory("Data/audio/RAW/angry", "Data/audio/Processed/angry")
process_directory("Data/audio/RAW/disgusted", "Data/audio/Processed/disgusted")
process_directory("Data/audio/RAW/fearful", "Data/audio/Processed/fearful")
process_directory("Data/audio/RAW/happy", "Data/audio/Processed/happy")
process_directory("Data/audio/RAW/neutral", "Data/audio/Processed/neutral")
process_directory("Data/audio/RAW/sad", "Data/audio/Processed/sad")
process_directory("Data/audio/RAW/surprised", "Data/audio/Processed/surprised")

