import mne 
import numpy as np
import matplotlib.pyplot as plt
import os
from read_files import LoadFile

"""
path ="EDF filer/"
folder = os.fsencode(path)

filenames = []

XY_keys = "M/P, P/M, M/P, M/P, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, P/M, P/M, M/P, M/P, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, M/P, M/P, P/M, M/P, P/M, M/P, M/P, P/M, P/M, M/P, P/M, M/P, P/M, M/P, M/P, P/M, M/P, P/M, M/P, M/P, M/P, P/M, M/P, M/P, M/P, M/P, P/M, P/M, M/P, M/P"
XY_keys = XY_keys.split(", ")

new_XY_keys = []
for i in range(len(XY_keys)):
    new_XY_keys.append((XY_keys[i][0], XY_keys[i][2]))


for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( '.edf'):
        filenames.append(filename)
        fname = path + filename
        #print(fname)
        raw = mne.io.read_raw_edf(fname, verbose=False)
        #print(raw.info) #uncomment if info is wanted
        #print(raw.annotations.description)  #uncomment if wanted
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



def LoadFile(fname):
    
    raw = mne.io.read_raw_edf(fname, verbose=False)
       
    #print(raw.info) #uncomment if info is wanted
    #print(raw.annotations.description)  #uncomment if wanted
        
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
    channels = raw.info.ch_names
    channels_rename = [i.replace('EEG ','') for i in channels]
    channels_rename = [i.replace('-REF','') for i in channels_rename]
    chan_dict = dict(zip(channels,channels_rename))
    mne.rename_channels(raw.info, chan_dict)
    channels = raw.info.ch_names
    raw.drop_channels(['EOG AOG', 'ECG EKG', 'Photic', 'Pulse Rate','IBI', 'Bursts', 'Suppr'])

    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    channels = raw.info.ch_names

    return raw, annot, annot_dict





"""




path ="EDF_test/"

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

    #mne.set_eeg_reference(raw, copy=False)

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
picks = ['Cz','Pz','O1','O2','C3','C4','P3','P4']
raw1,annot, annot_dict  = LoadFile('FuldEDF_test/24BC-EDF+.edf',picks=picks)
print(raw1.times[-1])
raw2,annot, annot_dict = LoadFile('FuldEDF_test/24BC-EDF+1.edf',picks=picks)
print(raw2.times[-1])
raw3,annot, annot_dict = LoadFile('FuldEDF_test/24BC-EDF+2.edf',picks=picks)
print(raw3.times[-1])
#raw4,annot, annot_dict = LoadFile('FuldEDF_test/04IW-EDF+3.edf',picks=picks)
#print(raw4.times[-1])
#raw5,annot, annot_dict = LoadFile('FuldEDF_test/04IW-EDF+4.edf',picks=picks)
#print(raw5.times[-1])
#raw6,annot, annot_dict = LoadFile('FuldEDF_test/04IW-EDF+5.edf',picks=picks)
#print(raw6.times[-1])
#raw7,annot, annot_dict = LoadFile('FuldEDF_test/04IW-EDF+6.edf',picks=picks)
raw = mne.io.concatenate_raws([raw1,raw2,raw3], preload=True)
#epochs, raw = LoadAllRaw(path=path, picks=picks)
raw1.plot()
raw2.plot()
raw3.plot()
#raw4.plot()
#raw5.plot()
#raw6.plot()
#raw7.plot()
#raw.plot()
plt.show()
