"""
Feature extraction
"""
import scipy.io.wavfile as wav
import speechpy as sp
import numpy as np


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
    :return: 39 mfcc features for each frame
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

print(len(ed), len(edd))

