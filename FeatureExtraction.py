"""
Feature extraction
"""
import scipy.io.wavfile as wav
import speechpy as sp


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
    Method to extract 39 mfcc features (mfcc, energy, deltas, delta deltas)
    :param rate: sampling rate of audio
    :param audio: audio signal (preemphasized best)
    :return: 39 mfcc features????
    """
    mfcc_features = sp.feature.mfcc(audio, rate, num_cepstral=12)
    mfe_features = sp.feature.lmfe(audio, rate)
    derivatives_mfcc = sp.feature.extract_derivative_feature(mfcc_features)
    derivatives_mfe = sp.feature.extract_derivative_feature(mfe_features)
    return mfcc_features, mfe_features, derivatives_mfcc, derivatives_mfe


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


s,r = preemphasize_signal("Data/p2_segmented/p2_1_q1.wav")
spec = extract_spectrum(s,r)
power = extract_power_spectrum(s,r)