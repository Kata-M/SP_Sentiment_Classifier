"""
Convert the .wav audio files to arrays that can be used for further processing and classification
"""

import os
import numpy as np
from scipy.io.wavfile import read
from pydub import AudioSegment
import math
import pandas as pd


def import_audio_names(path):
    """
    Method to import the .wav files of a directory
    :param path: given directory
    :return: an array of the sampling frequency and .wav files
    """
    audio_collection = []
    for filename in os.listdir(path):
        audio_collection.append(filename)
        # if filename.endswith(".wav"):
        #      a = read(directory + filename)
        #    audio_collection.append([a[0], a[1]])
    return audio_collection


def import_timestamps(path):
    """
    Method to import timestamps from tsv file specified in DATA folder
    :type path: Path to CSV file for timestamps
    :return: list of timestamps
    """
    df = pd.read_csv(path)
    timestamps = df.values.tolist()
    return timestamps


def cut_audio(audio, start, end, new_name):
    """
    Method to cut audio at a specific time
    :param new_name: participant number / seg / filename, e.g. 2/p2_seg0/p2_seg0_q1
    :param audio: audio to be cut
    :param start: start of audio snippet in minutes.second
    :param end: end of audio snippet in minutes.second
    :return: the cut audio
    """
    start = (math.floor(start) * 60 * 1000) + ((start - math.floor(start)) * 10000)  # Works in milliseconds
    end = (math.floor(end) * 60 * 1000) + ((end - math.floor(end)) * 10000)
    old_audio = AudioSegment.from_wav(audio)
    new_audio = old_audio[start:end]
    new_audio.export('Data/' + new_name + '.wav', format="wav")

cut_audio("Data/pPeter4.wav", 0, 0.1, "test")

def convert_audio(directory):
    """
    Method to convert .wav files in a directory to float arrays
    :param directory: given directory path
    :return: list of float arrays, one for each wav.file with the sample frequencies preceding
    """
    py_audio = []
    wav_audio = import_audio_names(directory)
    for audio in wav_audio:
        py_audio.append([audio[0], np.array(audio[1], dtype=float)])
    return py_audio


def prepare_input_data(audio, timestamps):
    """
    Method to be called from main to process input (cut audios for further processing)
    """
    filenames = import_audio_names(audio)
    ts = import_timestamps(timestamps)
    p_i = 1
    for filename in filenames:
        q_i = 1
        for timestamp in ts:
            seg_id = timestamp[0]
            name = 'p' + str(p_i) + '/p' + str(p_i) + '_' + str(seg_id) + '/' + 'p' + str(p_i) + '_' + str(seg_id) + '_q' + str(q_i)
            cut_audio(filename, timestamp[1], timestamp[2], name)
            q_i += 1
        p_i += 1

#prepare_input_data("Data/Original/", "Data/timestamps.csv")

