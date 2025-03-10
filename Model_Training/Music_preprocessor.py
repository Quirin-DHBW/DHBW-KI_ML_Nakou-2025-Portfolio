"""
Here we eat .mp3 files to turn them into constant length spectrograms!
"""


################
## IMPORTS #####
################
import pathlib

if __name__ == "__main__":
    import os
    import sys
    os.chdir(pathlib.Path(sys.path[0]).parent)

import librosa
import soundfile as sf # Used for the audio crustifyer!
import numpy as np
import skimage.io


####################
## DEFINITIONS #####
####################

def scrungle_audio(audio_path:str, output_path:str) -> None:
    y, sr = librosa.load(audio_path)
    sf.write(output_path, y, samplerate=512) # Instantly make any audio crusty.


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


##########################
## PROCESS ALL AUDIO #####
##########################

if __name__ == "__main__":
    print("Processing songs...")
    process_directory("Data/audio/RAW/angry", "Data/audio/Processed/angry")
    print("Angry completed... 1/7")
    process_directory("Data/audio/RAW/disgusted", "Data/audio/Processed/disgusted")
    print("Disgusted completed... 2/7")
    process_directory("Data/audio/RAW/fearful", "Data/audio/Processed/fearful")
    print("Fearful completed... 3/7")
    process_directory("Data/audio/RAW/happy", "Data/audio/Processed/happy")
    print("Happy completed... 4/7")
    process_directory("Data/audio/RAW/neutral", "Data/audio/Processed/neutral")
    print("Neutral completed... 5/7")
    process_directory("Data/audio/RAW/sad", "Data/audio/Processed/sad")
    print("Sad completed... 6/7")
    process_directory("Data/audio/RAW/surprised", "Data/audio/Processed/surprised")
    print("Surprised completed... 7/7")
    print("Conversion complete!")

