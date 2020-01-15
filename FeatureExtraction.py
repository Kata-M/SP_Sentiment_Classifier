"""
Feature extraction
"""
import scipy.io.wavfile as wav
import speechpy as sp
import numpy as np
import os
import pandas as pd
import glob

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
    mfe_features = sp.feature.lmfe(audio, rate)#get back 40 log energies for each filter for each frame

    energy_features = []
    for frame_energies in mfe_features:
        mean_energy = 0
        for energy in frame_energies:
            mean_energy += energy
        mean_energy = mean_energy/len(frame_energies)
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
        mean_delta_energy = mean_delta_energy/len(frame_energy_derivatives)
        mean_delta_delta_energy = mean_delta_delta_energy/len(frame_energy_derivatives)
        energy_delta_features.append(mean_delta_energy)
        energy_delta_delta_features.append(mean_delta_delta_energy)
    return mfcc_features, energy_features, mfcc_delta_features, mfcc_delta_delta_features, energy_delta_features, energy_delta_delta_features


def extract_mfcc_vectors(directory):
    """
    Method to extract the MFCC feature vectors (39 dimensions) for each frame of an audio file and write it to csv
    :param audiofile: Path to audio file
    """
    max_no_frames = 0 #know this to fill up the rest with 0 to have comparable data
    for filename in glob.glob(os.path.join(directory, '*.wav')):
        signal, rate = preemphasize_signal(filename)
        file_mfcc = extract_mfcc(signal, rate)[0]
        max_no_frames = max(max_no_frames, len(file_mfcc))
    print(max_no_frames)
    final_data = []
    for filename in glob.glob(os.path.join(directory, '*.wav')):
        signal, rate = preemphasize_signal(filename)
        file_mfcc, file_energy, file_mfcc_d, file_mfcc_d_d, file_energy_d, file_energy_d_d = extract_mfcc(signal, rate)
        mfcc, energy, mfcc_d, mfcc_dd, e_d, e_dd = [], [], [], [], [], []
        for coefficients in file_mfcc:
            mfcc.extend(coefficients)
        energy = file_energy
        for delta_coefficients in file_mfcc_d:
            mfcc_d.extend(delta_coefficients)
        for delta_delta_coefficients in file_mfcc_d_d:
            mfcc_dd.extend(delta_delta_coefficients)
        e_d.extend(file_energy_d)
        e_dd.extend(file_energy_d_d)
        print(len(mfcc_dd))
        if len(file_mfcc) < max_no_frames:
            difference = max_no_frames - len(file_mfcc)
            print(difference)
            mfcc.extend(np.zeros(difference*12, dtype=int))
            energy.extend(np.zeros(difference, dtype=int))
            mfcc_d.extend(np.zeros(difference*12, dtype=int))
            mfcc_dd.extend(np.zeros(difference*12, dtype=int))
            e_d.extend(np.zeros(difference, dtype=int))
            e_dd.extend(np.zeros(difference, dtype=int))
        print(len(mfcc_dd))
        end_vector = mfcc + energy + mfcc_d + mfcc_dd + e_d + e_dd
        print(len(end_vector))
        final_data.append(end_vector)
    df = pd.DataFrame(final_data)
    df.to_csv("Data/mfcc_features.csv")

extract_mfcc_vectors("Data/all_p_no_silence/")


def extract_spectrum(audio, rate):
    """
    Method to extract spectrum via Fast fourier transform
    :param audio: audio
    :param rate: rate
    :return: Spectrum feature
    """
    frames = sp.processing.stack_frames(audio, rate)
    spectrum = sp.processing.fft_spectrum(frames)
    return spectrum


def extract_power_spectrum(audio, rate):
    """
    Method to extract power spectrum for each frame
    :param audio: audio
    :param rate: rate
    :return: Spectrum feature
    """
    frames = sp.processing.stack_frames(audio, rate)
    power_spectrum = sp.processing.power_spectrum(frames)
    return power_spectrum


s,r = preemphasize_signal("Data/p2_segmented/N_p2_0_1a.wav")
m, e, md, mdd, ed, edd = extract_mfcc(s, r)

print(len(mdd[0]))

