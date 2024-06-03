import mne 
import os
import matplotlib.pyplot as plt
from mne_icalabel import label_components
import numpy as np
import pyprep
from autoreject import AutoReject

XY_keys = "M/P, P/M, M/P, M/P, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, P/M, P/M, M/P, M/P, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, M/P, M/P, M/P, P/M, M/P, M/P, M/P, M/P, P/M, P/M, M/P, M/P"
XY_keys = XY_keys.split(", ")

new_XY_keys = []
for i in range(len(XY_keys)):
    new_XY_keys.append((XY_keys[i][0], XY_keys[i][2]))

#removing nr 70 as the patient is missing
del new_XY_keys[69]

def reject_bad_segs(raw, annot_to_reject = ''):
    """ This function rejects all time spans annotated as annot_to_reject and concatenates the rest"""
    # this implementation seemed buggy, modified it here
    raw_segs = []
    tmin = 0
    for jsegment in range(1, len(raw.annotations)):
        #print(raw.annotations.description[jsegment], annot_to_reject)
        
        if raw.annotations.description[jsegment] == annot_to_reject:  # Append all other than 'bad_ITI'
             # start at ending of last bad annot
            tmax = raw.annotations.onset[jsegment] - 0.01 # end at onset of current bad annot
            raw_segs.append(
                raw.copy().crop( # this retains raw between tmin and tmax
                    tmin=tmin, 
                    tmax=tmax,
                    include_tmax=False, # this is onset of bad annot
                )
            )
            tmin = raw.annotations.onset[jsegment] + raw.annotations.duration[jsegment] + 0.01
            #print(tmin, tmax, len(raw_segs))
    
    print('tmin: ', tmin)
    raw_segs.append(raw.copy().crop(tmin=tmin))
    print(len(raw_segs))
    return mne.concatenate_raws(raw_segs)



# Function for preprocessing
def PreprocesEEG(raw, incl_ica = True):
    # Bad channels interpolation
    prep_params = {
    'ref_chs': 'eeg',
    'reref_chs': 'eeg',
    'line_freqs': [50] # notch filter
    }

    prep = pyprep.PrepPipeline(raw, prep_params, montage='standard_1020', random_state=97)
    prep.fit()
    raw = prep.raw

    print('Len: ', len(raw.ch_names))

    raw = raw.copy().filter(l_freq=0.1, h_freq=45, verbose=False) # 1-45 Hz filter
    #raw = raw.copy().interpolate_bads(reset_bads=True, verbose=True, mode='accurate') #interpolation of bad channels, if any
    raw = raw.copy().resample(sfreq = 250) # downsampling as some files have 500 Hz sample freq.

    raw.info['bads'] = []

    # ICA: set up and fit the ICA
    if incl_ica == True:
        # Break raw data into 1 s epochs
        tstep = 1.0
        events_ica = mne.make_fixed_length_events(raw, duration=tstep)
        epochs_ica = mne.Epochs(raw, events_ica,
                                tmin=0.0, tmax=tstep,
                                baseline=None,
                                preload=True)


        ar = AutoReject(n_interpolate=[1, 2, 4],
                        random_state=42,
                        picks=mne.pick_types(epochs_ica.info, 
                                            eeg=True,
                                            eog=False
                                            ),
                        n_jobs=-1, 
                        verbose=False
                        )

        ar.fit(epochs_ica)

        reject_log = ar.get_reject_log(epochs_ica)

                # ICA parameters
        random_state = 97   # ensures ICA is reproducible each time it's run
        ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=ica_n_components, random_state=random_state, method='infomax', fit_params=dict(extended=True))
        ica.fit(epochs_ica[~reject_log.bad_epochs], decim=3)


        #picks = raw.ch_names
        #ica = mne.preprocessing.ICA(n_components=len(picks)-1, random_state=97, max_iter=800, method='infomax') # ICA parameters - willl need to be adjusted probably
        #ica.fit(raw)

        #ica.exclude = [1, 2]  # details on how we picked these are omitted here
        #ica.plot_properties(raw, picks=ica.exclude)
        #plt.show()

        component_dict = label_components(raw, ica, method='iclabel')
        #c = collections.Counter(component_dict['labels'])

        prob = list(component_dict['y_pred_proba'])
        labels = list(component_dict['labels'])

        bad_comp = []

        for i in range(len(labels)):
            if labels[i] != 'brain':
                if prob[i] >= 0.8: # threshold is 80%
                    bad_comp.append(i)

        ica.exclude.extend(bad_comp)
        ica.apply(raw)

    return raw



# Selecting channels
picks = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'ECG EKG']

def LoadRaw(folder, path, patient_number, picks = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'ECG EKG']):
    raw_liste = []
    #finding all files for specific person
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        fname = path+filename
        person_number = filename[:2]
        if person_number[0] == 0:
            person_number = person_number[1]
        pn = int(person_number)

        if pn == patient_number:
            raw = mne.io.read_raw_edf(fname, verbose=False)
            annot = raw.annotations
            annot_dict = raw.annotations.count()

            # Changing annotations
            if 'Tiltale-X' in annot_dict:
                if new_XY_keys[int(filename[0:1])][0] == "P":
                    title = "Familiar voice"
                else:
                    title = "Medical staff"
                annot.rename({'Tiltale-X': title})
            if 'Tiltale-Y' in annot_dict:
                if new_XY_keys[int(filename[0:1])][1] == "P":
                    title = "Familiar voice"
                else:
                    title = "Medical staff"
                annot.rename({'Tiltale-Y': title})
            annot_dict = raw.annotations.count()

            #renaming channels
            channels = raw.info.ch_names
            channels_rename = [i.replace('EEG ','') for i in channels]
            channels_rename = [i.replace('-REF','') for i in channels_rename]
            chan_dict = dict(zip(channels,channels_rename))
            mne.rename_channels(raw.info, chan_dict)
            channels = raw.info.ch_names

            if picks != None:
                raw.pick(picks)

            #setting channel types
            channel_dict = {'ECG EKG':'ecg'}
            raw.set_channel_types(channel_dict, verbose=False)
            
            #setting montage type
            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage)

            if picks != None:
                raw.pick(picks)
            channels = raw.info.ch_names

            raw = raw.copy().resample(sfreq = 250)

            raw_liste.append(raw)

            #print(raw_liste)
        
        if pn > patient_number:
            break

    if len(raw_liste) != 1:
        raw_con = mne.concatenate_raws(raw_liste)
    else:
        raw_con = raw_liste[0]


    raw_con.plot()
    plt.show()

    if 'BAD_' in raw_con.annotations.count():
        raw_rej = reject_bad_segs(raw_con, 'BAD_')
    else:
        raw_rej = raw_con

    raw_rej = PreprocesEEG(raw_rej, incl_ica = True)


    raw_rej.save('examples/fif_to_csv_2/patient'+str(patient_number)+'_raw.fif') #saving raw as fif file for future usage. 

    return raw_rej, annot, annot_dict
path ="EDF filer/"
folder = os.fsencode(path)


for i in range(60,61):
    if i not in [64,1,26,55,76,70]:
        LoadRaw(folder, path, i)