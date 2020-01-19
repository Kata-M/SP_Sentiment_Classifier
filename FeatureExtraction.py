"""
Feature extraction
"""
import scipy.io.wavfile as wav
import speechpy as sp
import numpy as np
import os
import pandas as pd
import glob
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from itertools import chain


def preemphasize_signal(audio):
    """
    Method to preemphasize the signal by 0.97
    :param audio: audio file location
    :return: preemphasized signal
    """
    sample_rate, signal = wav.read(audio)
    p_signal = sp.processing.preemphasis(signal, cof=0.97)
    return p_signal, sample_rate


def extract_mfcc(audio, rate):
    """
    Method to extract 39 mfcc features (mfcc, energy, deltas, delta deltas) per frame
    :param rate: sampling rate of audio
    :param audio: audio signal (preemphasized best)
    :return: 12 mfccs, 1 energy, 12 mfcc deltas, 12 mfcc delta-deltas, 1 delta energy, 1 delta-delta energy
    """
    mfcc_features = sp.feature.mfcc(audio, rate, num_cepstral=12)
    mfe_features = sp.feature.lmfe(audio, rate)  # get back 40 log energies for each filter for each frame

    energy_features = []
    for frame_energies in mfe_features:
        mean_energy = 0
        for energy in frame_energies:
            mean_energy += energy
        mean_energy = mean_energy / len(frame_energies)
        energy_features.append(mean_energy)

    derivatives_mfcc = sp.feature.extract_derivative_feature(mfcc_features)
    mfcc_delta_features = []
    mfcc_delta_delta_features = []
    for frame_derivatives in derivatives_mfcc:
        coefficient_delta = []
        coefficient_delta_delta = []
        for coefficient in frame_derivatives:
            coefficient_delta.append(coefficient[1])
            coefficient_delta_delta.append(coefficient[2])
        mfcc_delta_features.append(coefficient_delta)
        mfcc_delta_delta_features.append(coefficient_delta_delta)

    derivatives_mfe = sp.feature.extract_derivative_feature(mfe_features)
    energy_delta_features = []
    energy_delta_delta_features = []
    for frame_energy_derivatives in derivatives_mfe:
        mean_delta_energy = 0
        mean_delta_delta_energy = 0
        for energy in frame_energy_derivatives:
            mean_delta_energy += energy[1]
            mean_delta_delta_energy += energy[2]
        mean_delta_energy = mean_delta_energy / len(frame_energy_derivatives)
        mean_delta_delta_energy = mean_delta_delta_energy / len(frame_energy_derivatives)
        energy_delta_features.append(mean_delta_energy)
        energy_delta_delta_features.append(mean_delta_delta_energy)

    # calculate averages over all frames
    avg_energy = sum(energy_features) / len(energy_features)
    avg_energy_d = sum(energy_delta_features) / len(energy_delta_features)
    avg_energy_dd = sum(energy_delta_delta_features) / len(energy_delta_delta_features)
    avg_mfcc = sum(map(np.array, mfcc_features)) / len(mfcc_features)
    avg_mfcc_d = sum(map(np.array, mfcc_delta_features)) / len(mfcc_delta_features)
    avg_mfcc_dd = sum(map(np.array, mfcc_delta_delta_features)) / len(mfcc_delta_delta_features)

    return avg_mfcc, avg_energy, avg_mfcc_d, avg_mfcc_dd, avg_energy_d, avg_energy_dd


def extract_pitch(path):
    """
    Method to extract pitch values and energy
    :param path: path to the audio file
    :return: pitch values and pitch energy, averaged over number of frames
    """
    signal = basic.SignalObj(path)
    pitch = pYAAPT.yaapt(signal)
    avg_pitch = sum(map(np.array, pitch.samp_values)) / pitch.nframes
    avg_pitch_energy = sum(map(np.array, pitch.energy)) / pitch.nframes
    return avg_pitch, avg_pitch_energy


def extract_features(directory):
    """
    Method to extract the MFCC feature vectors (39 dimensions) for each frame of an audio file and write it to csv
    :param directory: Path to audio file
    """
    final_data = []
    for filename in glob.glob(os.path.join(directory, '*.wav')):
        signal, rate = preemphasize_signal(filename)
        mfcc, energy, mfcc_d, mfcc_dd, energy_d, energy_dd = extract_mfcc(signal, rate)
        pitch, pitch_energy = extract_pitch(filename)
        end_vector = list(chain(mfcc, [energy], mfcc_d, mfcc_dd, [energy_d], [energy_dd], [pitch], [pitch_energy]))
        final_data.append(end_vector)
    df = pd.DataFrame(final_data)
    return df

print(extract_features("Data/all_p_no_silence/"))
