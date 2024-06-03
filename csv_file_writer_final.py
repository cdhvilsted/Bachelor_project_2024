
#from examples.split_based_on_resting import LoadRawP
from examples.Feature_calculation import FeatureMatrix2

import os
import csv
import numpy as np
import pandas as pd
import mne

path ="EDF_test/"
folder = os.fsencode(path)

def allToCSV(folder, path):

    all_pn = set()

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        print(filename)
        fname = path+filename
        
        person_number = filename[:2]
        if person_number[0] == 0:
            person_number = person_number[1]
        
        pn = int(person_number)
        all_pn.add(pn)

    for patient in all_pn:
        print('Patient number: ', patient)

        r, m, f = LoadRawP(folder=folder, path=path, patient_number=patient)
        a, b, c = 3, 10, 10
        if patient == 64:
            a, b, c = 1, 4, 4
        if patient == 72:
            a, b, c = 3, 9, 10
        if patient == 26:
            a, b, c = 2, 7, 6        

        rest_features = FeatureMatrix2(r, num_events=a)
        med_features = FeatureMatrix2(m, num_events=b)
        fam_features = FeatureMatrix2(f, num_events=c)

        markers_list = ['PSD Delta', 'PSD Delta_N', 'PSD Theta', 'PSD Theta_N', 'PSD Alpha', 'PSD Alpha_N', 'PSD Beta', 'PSD Beta_N', 'PSD Gamma', 'PSD Gamma_N', 'PSD SE', 'PSD MSF', 'PSD Sef90', 'PSD Sef95', 'PE', 'wSMI', 'Kolmogorov', 'Mean RR', 'Std RR', 'Mean HR', 'Std HR', 'Min HR', 'Max HR', 'Freq_Slope mean','Freq_Slope std'] #, 'VLF', 'LF', 'HF', 'VHF', 'Total power', 'LF/HF' 

        df_r = pd.DataFrame(rest_features)
        df_r.columns = markers_list
        df_r.insert(0, "Event", 'R')

        df_m = pd.DataFrame(med_features)
        df_m.columns = markers_list
        df_m.insert(0, 'Event', 'M')

        df_f = pd.DataFrame(fam_features)
        df_f.columns = markers_list
        df_f.insert(0, 'Event', 'F')

        df_all = pd.concat([df_r, df_m, df_f])
        print('df_all shape: ', df_all.shape)

        df_all.to_csv(path_or_buf = 'examples/CSV_inspected_features/p' + str(patient) +'_features.csv')
        print('File created succesfully!')

def fifToCsvP(folder, path, filename):
    picks = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'ECG EKG']
    raw = mne.io.read_raw_fif(fname=path+filename)
    events_from_annot, event_dict = mne.events_from_annotations(raw=raw)
    epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks, baseline=None, preload=True, event_repeated='merge', on_missing='warn')
    pnumber = filter(str.isdigit, filename)
    pnumber = "".join(pnumber)
    pnumber = int(pnumber)
    print('Patient is ',pnumber)
    rest = epochs['Resting']
    if len(rest) > 3:
        rest = rest[0:3]

    med = epochs['Medical staff']
    if len(med) > 10:
        med = med[0:10]
    
    if pnumber == 55:
        med.drop(3) # Only for patient nr 55 which has a bad epoch

    fam = epochs['Familiar voice']
    if len(fam) > 10:
        fam = fam[0:10]
    
    rest_features = FeatureMatrix2(rest, num_events=len(rest))
    med_features = FeatureMatrix2(med, num_events=len(med))
    fam_features = FeatureMatrix2(fam, num_events=len(fam))

    markers_list = ['PSD Delta', 'PSD Delta_N', 'PSD Theta', 'PSD Theta_N', 'PSD Alpha', 'PSD Alpha_N', 'PSD Beta', 'PSD Beta_N', 'PSD Gamma', 'PSD Gamma_N', 'PSD SE', 'PSD MSF', 'PSD Sef90', 'PSD Sef95', 'PE', 'wSMI', 'Kolmogorov', 'Mean RR', 'Std RR', 'Mean HR', 'Std HR', 'Min HR', 'Max HR', 'Freq_Slope mean','Freq_Slope std'] #, 'VLF', 'LF', 'HF', 'VHF', 'Total power', 'LF/HF' 
    
    df_r = pd.DataFrame(rest_features)
    df_r.columns = markers_list
    df_r.insert(0, "Event", 'R')

    df_m = pd.DataFrame(med_features)
    df_m.columns = markers_list
    df_m.insert(0, 'Event', 'M')

    df_f = pd.DataFrame(fam_features)
    df_f.columns = markers_list
    df_f.insert(0, 'Event', 'F')

    df_all = pd.concat([df_r, df_m, df_f])
    print('df_all shape: ', df_all.shape)

    df_all.to_csv(path_or_buf = 'examples/CSV_features_NEW/p' + str(pnumber) +'_features.csv') # Remember to change patient number in filename
    print('File created succesfully! for patient ', pnumber)


path = 'examples/fif_to_csv_1/'
folder = os.fsencode(path)

for file in os.listdir(folder):
    filename = os.fsdecode(file)
#filename = os.fsdecode('patient55_raw.fif')
    fifToCsvP(folder=folder, path=path, filename=filename)

