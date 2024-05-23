import mne 
import os
import matplotlib.pyplot as plt
import numpy as np
from mne_icalabel import label_components
import numpy as np
import pyprep
from examples.manual_inspection import LoadRaw

XY_keys = "M/P, P/M, M/P, M/P, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, P/M, P/M, M/P, M/P, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, M/P, M/P, M/P, P/M, M/P, M/P, M/P, M/P, P/M, P/M, M/P, M/P"
XY_keys = XY_keys.split(", ")

new_XY_keys = []
for i in range(len(XY_keys)):
    new_XY_keys.append((XY_keys[i][0], XY_keys[i][2]))

#removing nr 70 as the patient is missing
del new_XY_keys[69]


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
        picks = raw.ch_names
        ica = mne.preprocessing.ICA(n_components=len(picks)-1, random_state=97, max_iter=800) # ICA parameters - willl need to be adjusted probably
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
                if prob[i] >= 0.8: # threshold is 80%
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

    raw = PreprocesEEG(raw=raw, incl_ica = True) #uncomment if preprocessing is wanted 


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
            #raw, annot, annot_dict = LoadFile(fname)
            raw, annot, annot_dict = LoadRaw(folder, path, patient_number=pn)
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
    
    rest1 = []
    med1 = []
    fam1 = []
    rest2 = []
    med2 = []
    fam2 = []

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

        #to plot and inspect the concatenated file
        #raw_con.plot()
        #plt.show()

        events_from_annot, event_dict = mne.events_from_annotations(raw=raw_con)
        #print(events_from_annot)
        #print(event_dict)
        r_number = event_dict['Resting']
        r_index =  np.where(events_from_annot==r_number)
        #print(r_index)
        picks = mne.pick_types(raw_con.info, meg=False, eeg=True, stim=False, eog=False, ecg=True, exclude='bads')
        epochs1 = mne.Epochs(raw_con, events_from_annot[:r_index[0][1]], event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks, baseline=None, preload=True, event_repeated='merge', on_missing='warn')
        epochs2 = mne.Epochs(raw_con, events_from_annot[r_index[0][1]:], event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks, baseline=None, preload=True, event_repeated='merge', on_missing='warn')
        epoch_list_r1 = epochs1['Resting']
        epoch_list_m1 = epochs1['Medical staff']
        epoch_list_f1 = epochs1['Familiar voice']
        epoch_list_r2 = epochs2['Resting']
        epoch_list_m2 = epochs2['Medical staff']
        epoch_list_f2 = epochs2['Familiar voice']
        rest1.append([epoch_list_r1])
        med1.append([epoch_list_m1])
        fam1.append([epoch_list_f1])
        rest2.append([epoch_list_r2])
        med2.append([epoch_list_m2])
        fam2.append([epoch_list_f2])
                
    return rest1, rest2, med1, med2, fam1, fam2


def LoadEpochsP(folder, path, patient_number):
    raw_liste = []
    #finding all files for specific person
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        print(filename)
        fname = path+filename
        person_number = filename[:2]
        if person_number[0] == 0:
            person_number = person_number[1]
        pn = int(person_number)
        if pn == patient_number:
            raw, annot, annot_dict = LoadFile(fname)
            raw_liste.append(raw)
        if pn > patient_number:
            break
    #checking if multiple files should be concatenated


    #to plot and inspect the concatenated file
    #raw_con.plot()
    #plt.show()

    events_from_annot, event_dict = mne.events_from_annotations(raw=raw_con)
    #print(events_from_annot)
    #print(event_dict)
    picks = mne.pick_types(raw_con.info, meg=False, eeg=True, stim=False, eog=False, ecg=True, exclude='bads')
    epochs = mne.Epochs(raw_con, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks, baseline=None, preload=True, event_repeated='merge', on_missing='warn')
    rest = epochs['Resting']
    med = epochs['Medical staff']
    fam = epochs['Familiar voice']
            
    return rest, med, fam






#New correct load epochs!!!!!!!!!!!!!!!!!
def LoadRawP(folder, path, patient_number):
  
    raw_con, annot, annot_dict = LoadRaw(folder, path, patient_number)

    events_from_annot, event_dict = mne.events_from_annotations(raw=raw_con)

    picks = mne.pick_types(raw_con.info, meg=False, eeg=True, stim=False, eog=False, ecg=True, exclude='bads')
    epochs = mne.Epochs(raw_con, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks, baseline=None, preload=True, event_repeated='merge', on_missing='warn')
    
    rest = epochs['Resting']
    if len(rest) > 3:
        rest = rest[0:3]
    
    med = epochs['Medical staff']
    if len(med) > 10:
        med = med[0:10]

    fam = epochs['Familiar voice']
    if len(fam) > 10:
        fam = fam[0:10]
            
    return rest, med, fam


