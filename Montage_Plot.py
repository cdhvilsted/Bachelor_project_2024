import mne
import matplotlib.pyplot as plt

import os

raw = mne.read_epochs('JSXXX-epo.fif')
raw2 = mne.io.read_raw_edf('FuldEDF_test/02IT-EDF+.edf')
channels = raw2.info.ch_names
channels_rename = [i.replace('EEG ','') for i in channels]
channels_rename = [i.replace('-REF','') for i in channels_rename]
chan_dict = dict(zip(channels,channels_rename))
mne.rename_channels(raw2.info, chan_dict)
channels = raw2.info.ch_names
montage = mne.channels.make_standard_montage("standard_1020")
picks = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'T10', 'T9', 'F9', 'P9', 'P10', 'F10']

raw2.pick_channels(picks)
raw2.set_montage(montage)
montage2 = mne.channels.make_standard_montage('GSN-HydroCel-256')
raw.set_montage(montage2)

print(raw.info)
print(raw.info.ch_names)
print(raw.get_montage())
#fig, ax = plt.subplots(2)
raw.get_montage().plot()
raw2.get_montage().plot()

plt.show()