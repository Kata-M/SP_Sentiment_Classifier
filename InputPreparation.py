"""
Preprocess audio recordings
Removing silences
cutting of audio
labeling
"""

import os
import math
import pandas as pd
import librosa
from pydub import AudioSegment

path_test = "Data/Original/"

def get_audiofilenames(path):
    """
    Method to get the audiofile names with extension .wav
    :param path: directory where the audio files are
    :return: array of audio file names
    """
    audio_collection = []
    for filename in os.listdir(path):
        root, ext = os.path.splitext(filename)
        if ext == '.wav':
            audio_collection.append(filename)

    print("get audio files result: ")
    print("---------------------------")
    for audio in audio_collection:
        print(audio)
    print("---------------------------")
    return audio_collection

# test the above function
# get_audiofilenames(path_test)


def get_timestamps(path):
    """
    Method to get timestamps from csv file specified in DATA folder
    :type path: Path to CSV file for timestamps
    :return: array of timestamps of the format
    [segment number, question number, time-start, time-end]
    for example:
    [1, 'Intro', 0.04, 1.01]
    [1, '1', 1.06, 1.58]
    """
    df = pd.read_csv(path)
    timestamps = df.values.tolist()
    print("get timestamps results: ")
    for timestamp in timestamps:
        print(timestamp)
    print("---------------------------")
    return timestamps

#test the above function
#get_timestamps("Data/timestamps.csv")


def cut_audio_segment(old_name, start_stamp, end_stamp, new_name):
    """
    Method to cut audio at a specific time
    :param new_name: participant number / seg / filename, e.g. 2/p2_seg0/p2_seg0_q1
    :param old name: old name of audio file
    :param start: start of audio snippet in minutes.second
    :param end: end of audio snippet in minutes.second
    :return: the cut audio
    """
    if math.isnan(end_stamp):
        print("NaN entry in time stamps")
    else:
        print("in cut audio segment else")
        start = (math.floor(start_stamp) * 60 * 1000) + (
                (start_stamp - math.floor(start_stamp)) * 100000)  # Works in milliseconds
        print("---------------------------")
        print(math.floor(start_stamp))
        print(start_stamp - math.floor(start_stamp))
        print(start)
        print("---------------------------")
        end = (math.floor(end_stamp) * 60 * 1000) + ((end_stamp - math.floor(end_stamp)) * 100000)
        print("---------------------------")
        print(math.floor(end_stamp))
        print(end_stamp - math.floor(end_stamp))
        print(end)
        print("---------------------------")
        old_audio = AudioSegment.from_wav(old_name)
        new_audio = old_audio[start:end]
        print("past audio split")
        new_audio.export("./Data/Segmented/" + new_name + '.wav', format="wav")
        print("past export of:  " + new_name + '.wav')

# test the above function
# cut_audio_segment("Data/Original/abc.wav", 0.00 , 0.165 , "test")


def cut_all_audio(filenames, timestamps):
    """
    Method to loop through all audio files and save the audio between start-end timestamps
    :param filenames: filenames for files which want to be cut
    :param timestamps: timestamps array with the same format from return of get_timestamps
    """
    for filename in filenames:
        print("---- filename --- ")
        print(filename)
        print("test librosa: ")
        # gets the duration of the audio in seconds
        audio_duration = librosa.get_duration(filename="Data/Original/" + filename)
        # convert audio duration to ms
        audio_duration_ms = audio_duration * 1000
        print(audio_duration_ms)
        for timestamp in timestamps:
            # reading the whole audio file from file takes a long time!
            # old_audio = AudioSegment.from_wav("Data/Original/"+filename)
            end_stamp = timestamp[3]
            if math.isnan(end_stamp):
                print("reached NaN timestamp")
            else:
                end = (math.floor(end_stamp) * 60 * 1000) + ((end_stamp - math.floor(end_stamp)) * 100000)
                if end <= audio_duration_ms:
                    seg_no = timestamp[0]
                    q_no = timestamp[1]
                    new_name = filename[:2] + '_' + str(seg_no) + '_q' + str(q_no)
                    print("start cut audio segment for : " + filename)
                    cut_audio_segment("Data/Original/" + filename, timestamp[2], timestamp[3], new_name)
                else:
                    print("end time stamp larger than audio duration")


def prepare_segments(path_audiofiles, path_timestamps):
    """
    Method to be called from main to process input (cut audios for further processing)
    """
    print("start of prepare_input_data()")
    filenames = get_audiofilenames(path_audiofiles)
    timestamps = get_timestamps(path_timestamps)
    print("PRINT PREPARE INPUT DATA filenames")
    print(filenames)
    cut_all_audio(filenames, timestamps)
    print("cut all audio called from prepare input data and done!")


# prepare_segments("Data/Original/", "Data/timestamps.csv")

def add_classification_labels(path, self_eval_rates):
    """"
    Method for labelling the audio files to stressed and non-stressed. File name Y-name if stressed, N-name if not stressed
    """
    audio_files = get_audiofilenames(path)
    df = pd.read_csv(self_eval_rates)
    print(df)
    eval_rates = df.values.tolist()
    print(eval_rates)
    i = 0
    for file in audio_files:
        rating = ""
        print("----File----")
        print(file)
        p_no_audio_file = file[1] #str
        print("person in audio file : "+p_no_audio_file)
        for rate in eval_rates:
            # check that the evaluation participant matches with participant audio
            p_no_rate = str(rate[0])  # cast int to str
            print("person in evaluation form : " + p_no_rate)
            if p_no_rate == p_no_audio_file:
                print("found a match in participant numbers between evaluation and audio files : "+p_no_audio_file+" - "+p_no_rate)
                file_segment = int(file[3]) #get the segment number from the audio file name
                rate_seg = rate[1] #get the segment number from the evaluation form
                rate_rating = rate[3] #get the rating of overwhelmedness from the evaluation form
                print("*************")
                print("file_segment : " + str(file_segment))
                print("rate_seg : "+str(rate_seg))
                print("rate_rating : "+str(rate_rating))
                print("*************")
                if file_segment == rate_seg and rate_rating < 3:
                    print("in rating N")
                    rating = "N_"
                    # remane the file name according to self eval rating
                    os.rename(path + file, path + rating + file)
                elif file_segment == rate_seg and rate_rating > 2:
                    print("in rating Y")
                    rating = "Y_"
                    # remane the file name according to self eval rating
                    os.rename(path + file, path + rating + file)
                else:
                    print("No match between audio and evaluation form found!")
        i += 1
    print("Labelling DONE!")


# test the function above
# change the path to audio files according to the participant: "Data/pX_segmented/". E.g. "Data/p2_segmented/"
# add_classification_labels("Data/p2_segmented/", "Data/Self_Eval.csv")

def remove_silences(path , silence_thres_dB ):
    # get audio file names
    filenames = get_audiofilenames(path)
    for filename in filenames:
        print("----file name  ----")
        print(filename)
        # Load some audio
        y, sr = librosa.load(path + filename)
        print("get audio done")
        # Trim the beginning and ending silence
        print("trim start")
        yt, index = librosa.effects.trim(y, silence_thres_dB)
        print("trim done")
        # Print the durations
        print(librosa.get_duration(y), librosa.get_duration(yt))
        print("Remove silence done")
        librosa.output.write_wav("Data/all_p_no_silence/"+filename, yt, sr, norm=True)
        print("write to file done")

# execute remove silences
#remove_silences("Data/all_p_segmented/", 27.0)
