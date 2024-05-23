import mne 
import os
import matplotlib.pyplot as plt

XY_keys = "M/P, P/M, M/P, M/P, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, P/M, P/M, M/P, M/P, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, M/P, M/P, M/P, P/M, M/P, M/P, M/P, M/P, P/M, P/M, M/P, M/P"
XY_keys = XY_keys.split(", ")

new_XY_keys = []
for i in range(len(XY_keys)):
    new_XY_keys.append((XY_keys[i][0], XY_keys[i][2]))

#removing nr 70 as the patient is missing
del new_XY_keys[69]


# Function for preprocessing
def PreprocesEEG(raw, incl_ica = False):

    raw.filter(l_freq=0.1, h_freq=45, verbose=False) # 1-45 Hz filter

    raw = raw.copy().interpolate_bads(reset_bads=True, verbose=True, mode='accurate') #interpolation of bad channels, if any

    raw = raw.copy().resample(sfreq = 250) # downsampling as some files have 500 Hz sample freq.
    
    # ICA
    # set up and fit the ICA
    if incl_ica:
        ica = mne.preprocessing.ICA(n_components=10, random_state=97, max_iter=800) # ICA parameters - willl need to be adjusted probably
        ica.fit(raw)
        ica.exclude = [1, 2]  # details on how we picked these are omitted here
        ica.plot_properties(raw, picks=ica.exclude)
        plt.show()

    return raw



# Function for loading file
def LoadFile(fname, picks=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'ECG EKG', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']):
    """
    fname: path to file
    """
    filename = fname.split("/")[-1]
    raw = mne.io.read_raw_edf(fname, verbose=False, preload=True)
       
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

    events_from_annot, event_dict = mne.events_from_annotations(raw=raw)

    #renaming channels
    channels = raw.info.ch_names
    channels_rename = [i.replace('EEG ','') for i in channels]
    channels_rename = [i.replace('-REF','') for i in channels_rename]
    chan_dict = dict(zip(channels,channels_rename))
    mne.rename_channels(raw.info, chan_dict)
    channels = raw.info.ch_names
    #raw.drop_channels(['EOG AOG', 'ECG EKG', 'Photic', 'Pulse Rate','IBI', 'Bursts', 'Suppr'])

    #setting channel types
    channel_dict = {'ECG EKG':'ecg', 'Photic':'misc', 'Pulse Rate':'misc', 'IBI':'misc', 'Bursts':'misc', 'Suppr':'misc'}
    raw.set_channel_types(channel_dict, verbose=False)

    #setting montage type
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)


    if picks != None:
            raw.pick(picks)
    
    channels = raw.info.ch_names
    
    #print(raw.info["sfreq"])


    #raw = PreprocesEEG(raw=raw) #uncomment if preprocessing is wanted 


    return raw, annot, annot_dict




path ="EDF filer/"

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
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    epochs = mne.Epochs(
        raw, events_from_annot, event_id=event_dict, tmin=-0.2, tmax=15, proj=True, picks=picks,
        baseline=None, preload=True, event_repeated='merge')
    
    return epochs, raw

picks = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'ECG EKG', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']

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
