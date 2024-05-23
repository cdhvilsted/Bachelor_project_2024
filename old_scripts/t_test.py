import mne 
import numpy as np
import matplotlib.pyplot as plt
import os
from read_files import LoadFile

fname = "EDF filer/02IT-EDF+.edf"

raw, annot, annot_dict = LoadFile(fname)
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

picks = raw.info.ch_names