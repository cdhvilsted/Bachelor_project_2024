import mne 
import os
import matplotlib.pyplot as plt
from mne_icalabel import label_components
import numpy as np
import pyprep
import collections

XY_keys = "M/P, P/M, M/P, M/P, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, P/M, P/M, M/P, M/P, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, M/P, M/P, M/P, P/M, M/P, M/P, M/P, M/P, P/M, P/M, M/P, M/P"
XY_keys = XY_keys.split(", ")

new_XY_keys = []
for i in range(len(XY_keys)):
    new_XY_keys.append((XY_keys[i][0], XY_keys[i][2]))

#removing nr 70 as the patient is missing
del new_XY_keys[69]


# Function for preprocessing
def PreprocesEEG(raw, incl_ica = False):
    # Bad channels interpolation
    prep_params = {
    'ref_chs': 'eeg',
    'reref_chs': 'eeg',
    'line_freqs': [50] # notch filter
    }

    prep = pyprep.PrepPipeline(raw, prep_params, montage='standard_1020', random_state=97)
    prep.fit()
    raw = prep.raw

    raw = raw.copy().filter(l_freq=0.1, h_freq=45, verbose=False) # 1-45 Hz filter
    #raw = raw.copy().interpolate_bads(reset_bads=True, verbose=True, mode='accurate') #interpolation of bad channels, if any
    raw = raw.copy().resample(sfreq = 250) # downsampling as some files have 500 Hz sample freq.
    
    # ICA: set up and fit the ICA
    if incl_ica == True:
        ica = mne.preprocessing.ICA(n_components=18, random_state=97, max_iter=800) # ICA parameters - willl need to be adjusted probably
        ica.fit(raw)

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
                if prob[i] >= 0.5: # threshold is 50%
                    bad_comp.append(i)

        ica.exclude.extend(bad_comp)
        ica.apply(raw)

    return raw



# Function for loading file
def LoadFile(fname, picks=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'ECG EKG']):
    """
    fname: path to file
    """
    filename = fname.split("/")[-1]
    raw = mne.io.read_raw_edf(fname, verbose=False, preload=filename[0:5]) #.filter(l_freq=0.1, h_freq=45, verbose=False).interpolate_bads(reset_bads=True, verbose=False, mode='accurate').resample(sfreq = 250, verbose= False)
    #print(raw.info) #uncomment if info is wanted
    #print(raw.annotations.description)  #uncomment annotation info if wanted

    annot = raw.annotations
    #renaming annotations
    annot_dict = raw.annotations.count()
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

    #events_from_annot, event_dict = mne.events_from_annotations(raw=raw)

    #renaming channels
    channels = raw.info.ch_names
    channels_rename = [i.replace('EEG ','') for i in channels]
    channels_rename = [i.replace('-REF','') for i in channels_rename]
    chan_dict = dict(zip(channels,channels_rename))
    mne.rename_channels(raw.info, chan_dict)
    channels = raw.info.ch_names
    #raw.drop_channels(['EOG AOG', 'ECG EKG', 'Photic', 'Pulse Rate','IBI', 'Bursts', 'Suppr'])
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
    
    #print(raw.info["sfreq"])


    raw = PreprocesEEG(raw=raw, incl_ica = False) #uncomment if preprocessing is wanted 


    return raw, annot, annot_dict





def LoadAllRaw(path,picks):
    folder = os.fsencode(path)

    filenames = []
    raw_files_list = []

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith( '.edf'):
            filenames.append(filename)
            
            fname = path + filename
            raw_files, annot, annot_dict = LoadFile(fname,picks)
            raw_files_list.append(raw_files)
            print('-----------------------')
            print(filename)
            '''
            print(raw_files.info.ch_names)
            print('-----------------------')'''

    #raw_fnames = eegbci.load_data(subject, runs)
    #raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
    raw = mne.io.concatenate_raws(raw_files_list, preload=True)
    #raw.filter(1, 50)
    #raw = PreprocesEEG(raw=raw)
    
    mne.set_eeg_reference(raw, copy=False)

    # strip channel names of "." characters
    #raw.rename_channels(lambda x: x.strip('.'))

    events_from_annot, event_dict = mne.events_from_annotations(raw=raw)
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, ecg=True, exclude='bads')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = mne.Epochs(
        raw, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks,
        baseline=None, preload=True, event_repeated='merge')
    
    return epochs, raw

#picks = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'ECG EKG', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']


def LoadEpochs(folder, path, nstart, nend):
    nfiles = nend-nstart+1
    all_resting = [ [] for _ in range(81)]
    all_medical = [ [] for _ in range(81)]
    all_familiar = [ [] for _ in range(81)]
    raw_liste = [ [] for _ in range(81)]
    pn_set = set()

    
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        print(filename)
        fname = path+filename
        person_number = filename[:2]
        if person_number[0] == 0:
            person_number = person_number[1]
        pn = int(person_number)
        if nstart <= pn and pn <=nend:
            raw, annot, annot_dict = LoadFile(fname)
            raw_liste[pn].append(raw)
            pn_set.add(pn)    
        if pn > nend:
            break
       

        

        '''
        event_id = {'Resting': 1, 'Medical staff': 2, 'Familiar voice': 3}
        events_from_annot, event_dict = mne.events_from_annotations(raw=raw)
        picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
        epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks, baseline=None, preload=True, event_repeated='merge')    
        if 'Resting' in event_dict:
            all_resting[pn].append(epochs['Resting'])
        if 'Medical staff' in event_dict:
            all_medical[pn].append(epochs['Medical staff'])
        if 'Familiar voice' in event_dict:
            all_familiar[pn].append(epochs['Familiar voice'])
        '''
    
    rest = []
    med = []
    fam = []

    for i in pn_set:
    
        if len(raw_liste[i]) != 1:
            raw_con = mne.concatenate_raws(raw_liste[i])
        else:
            raw_con = raw_liste[i][0]
        '''
        epoch_list_r = mne.concatenate_epochs(all_resting[i])
        epoch_list_m = mne.concatenate_epochs(all_medical[i])
        epoch_list_f = mne.concatenate_epochs(all_familiar[i])
        rest_average.append(epoch_list_r.average())
        med_average.append(epoch_list_m.average())
        fam_average.append(epoch_list_f.average())
        '''
        events_from_annot, event_dict = mne.events_from_annotations(raw=raw_con)
        picks = mne.pick_types(raw_con.info, meg=False, eeg=True, stim=False, eog=False, ecg=True, exclude='bads')
        epochs = mne.Epochs(raw_con, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks, baseline=None, preload=True, event_repeated='merge')
    
        epoch_list_r = epochs['Resting']
        epoch_list_m = epochs['Medical staff']
        epoch_list_f = epochs['Familiar voice']
        
        rest.append([epoch_list_r])
        med.append([epoch_list_m])
        fam.append([epoch_list_f])
                
    return rest, med, fam

'''   
path ="FuldEDF_test/"
folder = os.fsencode(path)

r, m, f = LoadEpochs(folder=folder, path=path, nstart=0, nend=14)
r1, m1, f1 = LoadEpochs(folder=folder, path=path, nstart=15, nend=29)
r2, m2, f2 = LoadEpochs(folder=folder, path=path, nstart=30, nend=44)
r3, m3, f3 = LoadEpochs(folder=folder, path=path, nstart=45, nend=59)
r4, m4, f4 = LoadEpochs(folder=folder, path=path, nstart=60, nend=74)
r5, m5, f5 = LoadEpochs(folder=folder, path=path, nstart=75, nend=80)
print(len(r),len(r1),len(r2),len(r3),len(r4),len(r5))
print(len(r))
'''



#raw, annot, annot_dict = LoadFile('EDF filer/13IS-EDF+1.edf', picks=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz', 'F10', 'F9', 'P9', 'P10', 'ECG EKG', 'Photic', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr'])
#print(raw.info.ch_names)
#epochs = LoadAllRaw(path=path)
#print(epochs['Resting'])

#raw = mne.io.read_raw_edf('EDF filer/13IS-EDF+1.edf', verbose=False, preload=True)

'''raw = mne.io.read_raw_edf('EDF filer/13IS-EDF+1.edf', verbose=False, preload=True)
print(len(raw.info.ch_names))
raw = mne.io.read_raw_edf('EDF filer/13IS-EDF+2.edf', verbose=False, preload=True)
print(len(raw.info.ch_names))
raw = mne.io.read_raw_edf('EDF filer/13IS-EDF+3.edf', verbose=False, preload=True)
print(len(raw.info.ch_names))'''
