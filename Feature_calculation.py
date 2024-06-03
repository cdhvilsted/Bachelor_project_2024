import numpy as np
import matplotlib.pyplot as plt
from fooof import FOOOF, FOOOFGroup

import matplotlib.pyplot as plt

from nice import Markers
from nice.markers import (PowerSpectralDensity,
                          KolmogorovComplexity,
                          PermutationEntropy,
                          SymbolicMutualInformation,
                          PowerSpectralDensitySummary,
                          PowerSpectralDensityEstimator,
                          ContingentNegativeVariation,
                          TimeLockedTopography,
                          TimeLockedContrast)

import os
from examples.ECG_features import hrv_data

#to run file enter the 'nice' folder in terminal and run the command py -m examples.XX


path ="EDF_test/"
folder = os.fsencode(path)

#r1, m1, f1 = LoadEpochsP(folder=folder, path=path, patient_number=4)

psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=250)

base_psd = PowerSpectralDensityEstimator(
    psd_method='welch', tmin=None, tmax=0.6, fmin=1., fmax=45.,
    psd_params=psds_params, comment='default')

m_list = [
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
                         normalize=True, comment='summary_se'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.5, comment='summary_msf'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.9, comment='summary_sef90'),
    PowerSpectralDensitySummary(estimator=base_psd, fmin=1., fmax=45.,
                                percentile=.95, comment='summary_sef95'),

    PermutationEntropy(tmin=None, tmax=0.6, backend='python'),

    SymbolicMutualInformation(
        tmin=None, tmax=0.6, method='weighted', backend='python',
        method_params={'nthreads': 'auto'}, comment='weighted'),

    KolmogorovComplexity(tmin=None, tmax=0.6, backend='python',
                         method_params={'nthreads': 'auto'}),

    # Evokeds
    #ContingentNegativeVariation(tmin=-0.004, tmax=0.596), # ???? values for tmin and tmax
    #TimeLockedTopography(tmin=0.05, tmax=0.15, comment='N1'), # change to N1 peak
    #TimeLockedTopography(tmin=0.15, tmax=0.25, comment='P2'), # change to P2 peak
]

mc = Markers(m_list)
num_patients = 1 # Change according to number of files run
ch_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

rest = np.empty((num_patients, len(m_list)+6+2))
med = np.empty((num_patients, len(m_list)+6+2))
fam = np.empty((num_patients, len(m_list)+6+2))

print('Length of markers: ', len(m_list)+6+2)

def entropy(a, axis=0):  # noqa
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])

epochs_fun = np.mean
channels_fun = np.mean #trying mean now, was std before
reduction_params = {}
reduction_params['PowerSpectralDensity'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun},
         {'axis': 'frequency', 'function': np.mean}],
    'picks': {
        'epochs': None,
        'channels': ch_names}}

reduction_params['PowerSpectralDensity/summary_se'] = {
    'reduction_func':
        [{'axis': 'frequency', 'function': entropy},
         {'axis': 'epochs', 'function': np.mean},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': ch_names}}

reduction_params['PowerSpectralDensitySummary'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': ch_names}}

reduction_params['PermutationEntropy'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': ch_names}}

reduction_params['SymbolicMutualInformation'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels_y', 'function': np.median},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels_y': ch_names,
        'channels': ch_names}}

reduction_params['KolmogorovComplexity'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': ch_names}}
'''
reduction_params['ContingentNegativeVariation'] = {
    'reduction_func':
        [{'axis': 'epochs', 'function': epochs_fun},
         {'axis': 'channels', 'function': channels_fun}],
    'picks': {
        'epochs': None,
        'channels': ch_names}} #change channels
'''

def foof_params(epoch):
    psd = epoch.compute_psd()
    psds, freqs = psd.get_data(return_freqs=True)
    fm = FOOOFGroup()
    fm.fit(freqs,psds[0])
    slope = -fm.get_params('aperiodic_params','exponent')
    slope_mean = np.mean(slope)
    slope_std = np.std(slope)
    return [slope_mean, slope_std]


def FeatureMatrix2(data, num_events, mc=mc, reduction_params=reduction_params):
    matrix = np.empty((num_events, 25))
    mc.fit(data)
    markers = mc.reduce_to_epochs(marker_params=reduction_params).values()

    for count, marker in enumerate(markers):  
        matrix[:,count] = marker.T

    for epoch_num in range(num_events):
        matrix[epoch_num,17:23] = hrv_data(data[epoch_num])
        matrix[epoch_num,23:] = foof_params(data[epoch_num])
    
    for marker in mc.values():
        delattr(marker, 'data_')
    delattr(base_psd, 'data_')

    return matrix

