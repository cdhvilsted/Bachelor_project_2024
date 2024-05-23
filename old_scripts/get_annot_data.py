import mne 
import numpy as np
import matplotlib.pyplot as plt
import os
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
'''

fname = "EDF filer/01CX-EDF+.edf"
fname1 = "EDF filer/01CX-EDF+1.edf"
raw = mne.io.read_raw_edf(fname)
raw2 = mne.io.read_raw_edf(fname1)
print(raw.times[-1]/60)
print(raw2.times[-1]/60)

events_from_annot, event_dict = mne.events_from_annotations(raw=raw)
epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict, event_repeated=True)

print(epochs.times)

annot = mne.read_annotations(fname)
print(annot.count())

print("--------------------")
print(raw.info)
print(raw.annotations.description)
raw.set_annotations(annot)

fig = raw.plot()

plt.show()
'''


# Investigating whether annotations are 15 sec each
# Subtracting annotated data from all files and combining into 1 list

#rom read_files import LoadFile

path ="EDF filer/"
folder = os.fsencode(path)


def CountAnnotations(path, filename):
    fname = path + filename
    picks = ['Cz', 'C3', 'C4', 'Pz', 'P4', 'P3', 'O1', 'O2']
    raw, annot, annot_dict = LoadFile(fname=fname, picks=picks)
    count = 0
    count_impedance = 0

    all_resting = []
    all_medical = []
    all_familiar = []

    for i in annot.description:
        if i == 'Resting':
            begin_rest = annot.onset[count]
            duration_rest = annot.duration[count]
            end_rest = begin_rest + duration_rest
            all_resting.append(raw.get_data(start=int(begin_rest), stop=int(end_rest), verbose=False))

        if i == 'Medical staff':
            begin = annot.onset[count]
            duration = annot.duration[count]
            end = begin + duration
            all_medical.append(raw.get_data(start=int(begin), stop=int(end), verbose=False))

        if i == 'Familiar voice':
            begin = annot.onset[count]
            duration = annot.duration[count]
            end = begin + duration
            all_familiar.append(raw.get_data(start=int(begin), stop=int(end), verbose=False))

        count += 1

        if i == 'Impedance':
            count_impedance += 1

    return all_resting, all_medical, all_familiar    


def AnnotationPrPerson(folder):
    person_number = '01CX-EDF+'
    numbers = np.zeros((57,3))
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        if filename.endswith( '.edf') and filename[0:9] == person_number:
            all_resting, all_medical, all_familiar = CountAnnotations(path=path, filename=filename)

        else:
            person_number = filename[0:9]

    noRest = len(all_resting)
    noMed = len(all_medical)
    noFam = len(all_familiar)

    if noRest != 3 or noMed != 10 or noFam != 10:
        print('Person ', person_number, 'is splitted during annotation.')
        print('Rest: ', noRest,'. Medical: ', noMed,'. Familiar: ', noFam)


def AnnotationPrPerson2(folder):
    numbers = np.zeros((81,3))
    all_resting = [ [] for _ in range(81)]
    all_medical = [ [] for _ in range(81)]
    all_familiar = [ [] for _ in range(81)]


    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        person_number = filename[:2]
        if person_number[0] == 0:
            person_number = person_number[1]
        pn = int(person_number)
        
        resting, medical, familiar = CountAnnotations(path=path, filename=filename)
        all_resting[pn].append(resting)
        all_medical[pn].append(medical)
        all_familiar[pn].append(familiar)
        channels[pn].append(chan)
        numbers[pn,0] += len(resting)
        numbers[pn,1] += len(medical)
        numbers[pn,2] += len(familiar)    
    
    for i in range(1,80):
        if numbers[i,0] != 3 or numbers[i,1] != 10 or numbers[i,2] != 10:
            print('Person nr ', i, 'has wrong number of annotations: ', numbers[i]) 
    return numbers, all_resting, all_medical, all_familiar, channels


def CountChannels(folder):
    channels_len =  [ [] for _ in range(81)]
    channels =  [ [] for _ in range(81)]
    missing_chan = [ [] for _ in range(81)]
    all_channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz', 'F10', 'F9', 'P9', 'EOG AOG', 'P10', 'ECG EKG', 'Photic', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        fname = path+filename
        person_number = filename[:2]
        if person_number[0] == 0:
            person_number = person_number[1]
        pn = int(person_number)
        
        chan = mne.io.read_raw_edf(fname, verbose=False, preload=True).info.ch_names
        channels_rename = [i.replace('EEG ','') for i in chan]
        channels_rename = [i.replace('-REF','') for i in channels_rename]
        
        channels_len[pn].append(len(channels_rename))
        channels[pn].append(channels_rename)
        mchan = []
        for ch in all_channels:
            if ch not in channels_rename:
                mchan.append(ch)
        for ch in channels_rename:
            if ch not in all_channels:
                print('---------------')
                print('new chan: ', ch)
                print(filename)
                print('----------')
        missing_chan[pn].append(mchan)
         
    return channels, channels_len, missing_chan
path ="EDF filer/"
folder = os.fsencode(path)

'''
print(folder)
channels, channels_len, missing_chan = CountChannels(folder=folder)
print(channels)
print('-------------')
print(missing_chan)
'''




