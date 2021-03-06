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

import InputPreparation


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


def normalise_pitch(pitch, pitch_e, p_mu, p_sd, p_e_mu, p_e_sd):
    """
    Method: to normalise pitch of one participant
    Param: path_to_audio_files is path to files of only one participant
    """
    normalised_pitch = (pitch - p_mu) / p_sd
    normalised_pitch_energy = (pitch_e - p_e_mu) / p_e_sd

    return normalised_pitch, normalised_pitch_energy


def calculate_mu_sd(directory):
    """
    Method to calculate mu and sd for one participant given the folder
    :param directory: participant folder
    :return: mu, sd of that participant
    """
    audiofiles = InputPreparation.get_audiofilenames(directory)
    audiofiles = InputPreparation.get_audiofilenames(path_to_audio_file)
    pitch_allfiles = []
    pitchenergy_allfiles = []

    for audiofile in audiofiles:
        avg_pitch, avg_pitch_energy = extract_pitch(directory + audiofile)
        avg_pitch, avg_pitch_energy = extract_pitch(path_to_audio_file + audiofile)
        pitch_allfiles.append(avg_pitch)
        pitchenergy_allfiles.append(avg_pitch_energy)

    pitch_mu = np.mean(pitch_allfiles)
    pitch_mu_energy = np.mean(pitchenergy_allfiles)

    pitch_SD = np.std(pitch_allfiles)
    pitch_SD_energy = np.std(pitchenergy_allfiles)
    return pitch_mu, pitch_mu_energy, pitch_SD, pitch_SD_energy


def extract_features(directory):
    """
    Method to extract the MFCC feature vectors (39 dimensions) for each frame of an audio file and write it to csv
    :param directory: Path to audio file
    """
    p2_pitch_mu, p2_pitch_e_mu, p2_pitch_sd, p2_pitch_e_sd = calculate_mu_sd("Data/p2/")
    p3_pitch_mu, p3_pitch_e_mu, p3_pitch_sd, p3_pitch_e_sd = calculate_mu_sd("Data/p3/")
    p4_pitch_mu, p4_pitch_e_mu, p4_pitch_sd, p4_pitch_e_sd = calculate_mu_sd("Data/p4/")
    p5_pitch_mu, p5_pitch_e_mu, p5_pitch_sd, p5_pitch_e_sd = calculate_mu_sd("Data/p5/")
    p6_pitch_mu, p6_pitch_e_mu, p6_pitch_sd, p6_pitch_e_sd = calculate_mu_sd("Data/p6/")
    p7_pitch_mu, p7_pitch_e_mu, p7_pitch_sd, p7_pitch_e_sd = calculate_mu_sd("Data/p7/")

    final_data = []
    for filename in glob.glob(os.path.join(directory, '*.wav')):
        print(filename[25])
        signal, rate = preemphasize_signal(filename)
        mfcc, energy, mfcc_d, mfcc_dd, energy_d, energy_dd = extract_mfcc(signal, rate)
        pitch, pitch_energy = extract_pitch(filename)
        if str(filename[25]) == '2':
            this_pitch_mu, this_pitch_e_mu, this_pitch_sd, this_pitch_e_sd = p2_pitch_mu, p2_pitch_e_mu, p2_pitch_sd, p2_pitch_e_sd
        elif str(filename[25]) == '3':
            this_pitch_mu, this_pitch_e_mu, this_pitch_sd, this_pitch_e_sd = p3_pitch_mu, p3_pitch_e_mu, p3_pitch_sd, p3_pitch_e_sd
        elif str(filename[25]) == '4':
            this_pitch_mu, this_pitch_e_mu, this_pitch_sd, this_pitch_e_sd = p4_pitch_mu, p4_pitch_e_mu, p4_pitch_sd, p4_pitch_e_sd
        elif str(filename[25]) == '5':
            this_pitch_mu, this_pitch_e_mu, this_pitch_sd, this_pitch_e_sd = p5_pitch_mu, p5_pitch_e_mu, p5_pitch_sd, p5_pitch_e_sd
        elif str(filename[25]) == '6':
            this_pitch_mu, this_pitch_e_mu, this_pitch_sd, this_pitch_e_sd = p6_pitch_mu, p6_pitch_e_mu, p6_pitch_sd, p6_pitch_e_sd
        elif str(filename[25]) == '7':
            this_pitch_mu, this_pitch_e_mu, this_pitch_sd, this_pitch_e_sd = p7_pitch_mu, p7_pitch_e_mu, p7_pitch_sd, p7_pitch_e_sd
        pitch_n, pitch_energy_n = normalise_pitch(pitch, pitch_energy, this_pitch_mu, this_pitch_sd, this_pitch_e_mu,
                                                  this_pitch_e_sd)
        end_vector = list(chain(mfcc, [energy], mfcc_d, mfcc_dd, [energy_d], [energy_dd], [pitch_n], [pitch_energy_n]))
        final_data.append(end_vector)
    df = pd.DataFrame(final_data)
    return df


print(extract_features("Data/all_p_no_silence/"))
