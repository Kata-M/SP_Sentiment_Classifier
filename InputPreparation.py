"""
Convert the .wav audio files to arrays that can be used for further processing and classification
"""

import os
import numpy as np
from scipy.io.wavfile import read

def import_audio(directory):
    """
    Method to import the .wav files of a directory
    :param directory: given directory
    :return: an array of .wav files
    """
    audio_collection = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            audio_collection.append(read(directory + filename)[1])
    return audio_collection


def convert_audio(directory):
    """
    Method to convert .wav files in a directory to float arrays
    :param directory: given directory path
    :return: list of float arrays, one for each wav.file
    """
    py_audio = []
    wav_audio = import_audio(directory)
    for audio in wav_audio:
        py_audio.append(np.array(audio, dtype=float))
    return py_audio
