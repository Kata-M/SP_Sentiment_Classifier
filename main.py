from FeatureExtraction import preemphasize_signal
from FeatureExtraction import extract_mfcc
from InputPreparation import get_audiofilenames
import pandas as pd
import csv


def main(path_to_audiofiles, feature_file_path, DT_pic_name):
    """
    Main function which runs through audio files in path_to_audiofiles.
    Steps:
    1. Pre-emphasis applied on the audio files
    2. Compute features (MFCC, MFCC delta etc)
    3. Save features to a file
    4. Run desicion tree code

    """
    audiofiles = get_audiofilenames(path_to_audiofiles)
    print("test df ")
    df = pd.DataFrame()
    mfcc_list = df.values.tolist()
    print(" number of rows in the dataframe ")
    print(len(df.index))
    df.to_csv(feature_file_path)

    i = 0
    for audiofile in audiofiles:
        print(path_to_audiofiles+audiofile)
        audio_signal, sample_rate = preemphasize_signal(path_to_audiofiles+audiofile)
        mfcc, energy, mfccD, energyD = extract_mfcc(audio_signal, sample_rate)
        #print(mfcc)
        fields = []
        #for mfcc_coef_row in mfcc:
            #for mfcc_coef in mfcc_coef_row:
                #fields.append(mfcc_coef)

        mfcc_coef_row = mfcc[0]
        for mfcc_coef in mfcc_coef_row:
            fields.append(mfcc_coef)
        print("length of fields ")
        print(len(fields))
        print(fields)
        i += 1
        with open(r'Data/test.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        #for coef_row in mfcc:
        #    mfcc_list.append(coef_row)
        #    print("test mfcc list ")

        #print(mfcc_list)
    print("DONE")
    print("number of iterations :" + str(i))

    df = pd.read_csv("Data/test.csv", error_bad_lines=False)
    print(df)
    print(len(df.index))
    print("DONE DONE")





main("Data/all_p_no_silence/","Data/test.csv", "DT_pic.png")



