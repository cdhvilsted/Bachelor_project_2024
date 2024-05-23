"""BCI.

==================================================
Apply DOC-Forest recipe to single subject BCI data
==================================================

Here we use the resting state DOC-Forest [1] recipe to analyze BCI data.
Compared to the original reference, 2 major modifications are done.

1) For simplicity, we only compute 1 feature per marker, not 4
2) For speed, we use 200 trees, not 2000.

Compared to the common spatial patterns example form the MNE website,
the result is not particularly impressive. This is because a
global statistic like the mean or the standard deviation are a good
abstraction for severly brain injured patients but not for different
conditions in a BCI experiment conducted with healthy participants.


References
----------
[1] Engemann D.A.`*, Raimondo F.`*, King JR., Rohaut B., Louppe G.,
    Faugeras F., Annen J., Cassol H., Gosseries O., Fernandez-Slezak D.,
    Laureys S., Naccache L., Dehaene S. and Sitt J.D. (2018).
    Robust EEG-based cross-site and cross-protocol classification of
    states of consciousness. Brain. doi:10.1093/brain/awy251
"""

# Authors: Denis A. Engemann <denis.engemann@gmail.com>
#          Federico Raimondo <federaimondo@gmail.com>


#to run file enter the 'nice' folder in terminal and run the command py -m examples.plot_markers


import numpy as np
import mne

from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

from nice import Markers
from nice.markers import (PowerSpectralDensity,
                          KolmogorovComplexity,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          PowerSpectralDensitySummary,
                          PowerSpectralDensityEstimator)

from .LoadFile import LoadFile
import os
import sklearn

# avoid classification of evoked responses by using epochs that start 1s after
# cue onset.
tmin, tmax = -0.2, 15
#event_id = dict(rest=3, med=10, fam=10)
subject = 1
#runs = [6, 10, 14]  # motor imagery: hands vs feet

path ="EDF filer/"
folder = os.fsencode(path)

'''filenames = []
raw_files_list = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( '.edf'):
        filenames.append(filename)
        
        fname = path + filename
        raw_files, annot, annot_dict = LoadFile(fname)
        raw_files_list.append(raw_files)
        print('-----------------------')
        print(filename)
        print(raw_files.info.ch_names)
        print('-----------------------')'''

#raw_fnames = eegbci.load_data(subject, runs)
#raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames]
#raw = concatenate_raws(raw_files_list, preload=True)
raw, annot, annot_dict = LoadFile('EDF filer/02IT-EDF+.edf')

raw.filter(1, 45)

mne.set_eeg_reference(raw, copy=False)

# strip channel names of "." characters
#raw.rename_channels(lambda x: x.strip('.'))

events_from_annot, event_dict = mne.events_from_annotations(raw=raw)

''' skal rettes


specific_events = ['Resting', 'Medical']
specific_event_dict = dict()

for i in event_dict:
    if i.values() in specific_events:
        specific_event_dict[i.keys()] = i.values()

print(specific_event_dict)
'''

picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = mne.Epochs(
    raw, events_from_annot, event_id=event_dict, tmin=tmin, tmax=tmax, proj=True, picks=picks,
    baseline=(-0.2,0), preload=True, event_repeated='merge')

resting = epochs['Resting']
medical = epochs['Medical staff']
familiar = epochs['Familiar voice']
psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)


##############################################################################
# Prepare markers

backend = 'python'  # This gives maximum compatibility across platforms.
# For improved speed, checkout the optimization options using C extensions.

# We define one base estimator to avoid recomputation when looking up markers.
base_psd = PowerSpectralDensityEstimator(
    psd_method='welch', tmin=None, tmax=None, fmin=1., fmax=45.,
    psd_params=psds_params, comment='default')


# Here are the resting-state compatible markers we considered in the paper.

markers = Markers([
    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                         normalize=False, comment='delta'),
    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=4.,
                         normalize=True, comment='deltan'),
    PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                         normalize=False, comment='theta'),
    PowerSpectralDensity(estimator=base_psd, fmin=4., fmax=8.,
                         normalize=True, comment='thetan'),
    PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                         normalize=False, comment='alpha'),
    PowerSpectralDensity(estimator=base_psd, fmin=8., fmax=12.,
                         normalize=True, comment='alphan'),
    PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                         normalize=False, comment='beta'),
    PowerSpectralDensity(estimator=base_psd, fmin=12., fmax=30.,
                         normalize=True, comment='betan'),
    PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                         normalize=False, comment='gamma'),
    PowerSpectralDensity(estimator=base_psd, fmin=30., fmax=45.,
                         normalize=True, comment='gamman'),
    PowerSpectralDensity(estimator=base_psd, fmin=1., fmax=45.,
                         normalize=False, comment='summary_se'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.5, comment='summary_msf'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.9, comment='summary_sef90'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.95, comment='summary_sef95'),
    PermutationEntropy(tmin=None, tmax=0.6, backend=backend),
    # csd needs to be skipped
    SymbolicMutualInformation(
        tmin=None, tmax=0.6, method='weighted', backend=backend,
        method_params={'nthreads': 'auto', 'bypass_csd': True},
        comment='weighted'),

    KolmogorovComplexity(tmin=None, tmax=0.6, backend=backend,
                         method_params={'nthreads': 'auto'}),
])

##############################################################################
# Prepare reductions.
# Keep in mind that this is BCI, we have some localized effects.
# Therefore we will consider the standard deviation acros channels.
# Contraty to the paper, this is a single subject analysis. We therefore do
# not pefrorm a full reduction but only compute one statistic
# per marker and per epoch. In the paper, instead, we computed summaries over
# epochs and sensosrs, yielding one value per marker per EEG recoding.

epochs_fun = np.mean
channels_fun = np.std
reduction_params = {
    'PowerSpectralDensity': {
        'reduction_func': [
            {'axis': 'frequency', 'function': np.sum},
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    },
    'PowerSpectralDensitySummary': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    },
    'SymbolicMutualInformation': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun},
            {'axis': 'channels_y', 'function': channels_fun}]
    },
    'PermutationEntropy': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    },
    'KolmogorovComplexity': {
        'reduction_func': [
            {'axis': 'epochs', 'function': epochs_fun},
            {'axis': 'channels', 'function': channels_fun}]
    }
}

rest = np.empty((len(resting), len(markers)))
med = np.empty((len(medical), len(markers)))
fam = np.empty((len(familiar), len(markers)))

for ii in range(len(resting)):
    markers.fit(resting[ii])
    rest[ii, :] = markers.reduce_to_scalar(marker_params=reduction_params)
    # XXX hide this inside code
    for marker in markers.values():
        delattr(marker, 'data_')
    delattr(base_psd, 'data_')

for ii in range(len(medical)):
    markers.fit(medical[ii])
    med[ii, :] = markers.reduce_to_scalar(marker_params=reduction_params)
    # XXX hide this inside code
    for marker in markers.values():
        delattr(marker, 'data_')
    delattr(base_psd, 'data_')

for ii in range(len(familiar)):
    markers.fit(familiar[ii])
    fam[ii, :] = markers.reduce_to_scalar(marker_params=reduction_params)
    # XXX hide this inside code
    for marker in markers.values():
        delattr(marker, 'data_')
    delattr(base_psd, 'data_')

print(rest)
print('med:', med)
print('fam:', fam)

y = epochs.events[:, 2] - 2
print('this is y!!!!!', y)
var_names = list(markers.keys())
var_names = [var_names[ii].lstrip('nice/marker/') for ii in range(len(var_names))]
print(var_names)
fig, ax = plt.subplots(3,sharex=True)
ax[0].boxplot(rest)
ax[0].set_title('rest')
ax[0].set_ylim(0,4)
ax[1].boxplot(fam)
ax[1].set_title('familiar')
ax[1].set_ylim(0,4)
ax[2].boxplot(med)
ax[2].set_title('medical')
ax[2].set_ylim(0,4)
plt.xticks(range(17),var_names,rotation=45)
plt.grid()

##############################################################################
# Original DOC-Forest recipe


plt.show()
