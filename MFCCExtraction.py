"""
Extract the following features for classification:
MFCCs with deltas and delta deltas
"""
import numpy
import scipy.io.wavfile as wav
from scipy.fftpack import dct


def import_signal(file):
    """
    Method to import one wav file and convert it to a signal
    :param file: file name and path
    :return: signal (array format)
    """
    sample_rate, signal = wav.read(file)
    return sample_rate, signal

r, s = import_signal("Data/p2_segmented/p2_1_q1.wav")

def preemphasis(signal):
    """
    Method to pre-emphasize given audio signal by alpha = 0.97
    :param signal: given audio signal
    :return: pre-emphasized audio signal
    """
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
    return emphasized_signal
print(preemphasis(s))

def frame(signal, sample_rate):
    """
    Method to split audio signal to frames of 0.25ms and 0.10 ms of stride (0.15ms overlap)
    :param signal: audio signal
    :param sample_rate: sample rate of audio signal
    :return: frames from the given audio signal and their length
    """
    frame_size = 0.025
    frame_stride = 0.01
    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(
        float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    pad_signal = numpy.append(signal,
                              z)  # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    return frames, frame_length


def h_window(frames, frame_length):
    """
    Apply hamming window to frames
    :param frames: 0.025ms frames
    :param frame_length: length of one frame
    :return: windowed frames
    """
    frames *= numpy.hamming(frame_length)
    windowed_frames = frames
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    return windowed_frames


def power_spectrum(frames, nfft):
    """
    Method to calculate the power spectrum
    :param frames: frames (best windowed frames)
    :param nfft: 512- fft
    :return: power spectrum frames
    """
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, nfft))  # Magnitude of the FFT
    pow_frames = ((1.0 / nfft) * (mag_frames ** 2))  # Power Spectrum
    return pow_frames


def compute_filter_banks(pow_frames, nfft, sample_rate):
    """
    Method to compute the mel filter banks
    :param pow_frames: given frames to pass through the filter
    :param nfft: number of ffts
    :return: filter banks
    """
    nfilt = 40 #40 filters
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    b = numpy.floor((nfft + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(b[m - 1])  # left
        f_m = int(b[m])  # center
        f_m_plus = int(b[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - b[m - 1]) / (b[m] - b[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (b[m + 1] - k) / (b[m + 1] - b[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB
    return filter_banks


def compute_mfcc(file):
    """
    Method to compute 12 MFCCs given a file
    :param file: file in path
    :return: 12 mfccs per frame in the given file
    """
    num_ceps = 12 #number of cepstral coefficients
    nfft = 512
    sample_rate, signal = import_signal(file)
    signal = preemphasis(signal)
    frames, frame_length = frame(signal, sample_rate)
    windowed_frames = h_window(frames, frame_length)
    pow_frames = power_spectrum(windowed_frames, nfft)
    filter_banks = compute_filter_banks(pow_frames,  nfft, sample_rate)
    mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    return mfccs

x = compute_mfcc("Data/p2_segmented/p2_1_q1.wav")
print(x[0], len(x))
