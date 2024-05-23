import mne 
import numpy as np
import matplotlib.pyplot as plt
import os
from read_files import LoadFile
import scipy as scipy

from get_annot_data import CountAnnotations

path ="EDF filer/"
filename = "02IT-EDF+.edf"
fname = path + filename

picks = 'Cz'
raw, annot, annot_dict = LoadFile(fname=fname, picks=picks)
all_resting, all_medical, all_familiar = CountAnnotations(path=path, filename=filename)

events_from_annot, event_dict = mne.events_from_annotations(raw=raw)


'''
print(annot_dict)
print(raw.info)
events_from_annot, event_dict = mne.events_from_annotations(raw=raw)
print(events_from_annot)
print(event_dict)
channels = raw.info.ch_names
channels_rename = [i.replace('EEG ','') for i in channels]
channels_rename = [i.replace('-REF','') for i in channels_rename]
chan_dict = dict(zip(channels,channels_rename))
mne.rename_channels(raw.info, chan_dict)
channels = raw.info.ch_names
raw.drop_channels(['EOG AOG', 'ECG EKG', 'Photic', 'Pulse Rate','IBI', 'Bursts', 'Suppr'])
montage = mne.channels.make_standard_montage("standard_1020")
raw.set_montage(montage)
'''


#only Cz channel for ERP, baseline -200 ms to 0
#picks = raw.info.ch_names
#picks = ['Cz']
epochs = mne.Epochs(raw, events_from_annot, event_id=event_dict, picks='Cz', event_repeated = 'drop')
average_resting = epochs['Resting'].average()
average_familiar = epochs['Familiar voice'].average()
average_medical = epochs['Medical staff'].average()



evokeds1 = dict(voice=list(epochs['Familiar voice','Medical staff'].iter_evoked()),
               resting=list(epochs['Resting'].iter_evoked()))
evokeds2 = dict(Familiar=list(epochs['Familiar voice'].iter_evoked()),
               Medical=list(epochs['Medical staff'].iter_evoked()))
#fig = mne.viz.plot_compare_evokeds(evokeds2, combine='mean', picks='Cz', invert_y=True, title='Averages across channels')
average_resting.plot(spatial_colors=True)
average_familiar.plot(spatial_colors=True)
average_medical.plot(spatial_colors=True)

#average_familiar.plot(average='mean')



# distribution and stats
average_resting_chan = np.mean(average_resting.get_data(),axis=0)
average_resting_chan_std = np.std(average_resting.get_data(),axis=0)

mean_rest = np.mean(average_resting.get_data(['Cz', 'Pz', 'O1']),axis=1)
std_rest = np.std(average_resting.get_data(['Cz', 'Pz', 'O1']),axis=1)
mean_fam = np.mean(average_familiar.get_data(['Cz', 'Pz', 'O1']),axis=1)
std_fam = np.std(average_familiar.get_data(['Cz', 'Pz', 'O1']),axis=1)
mean_med = np.mean(average_medical.get_data(['Cz', 'Pz', 'O1']),axis=1)
std_med = np.std(average_medical.get_data(['Cz', 'Pz', 'O1']),axis=1)
print('______________________')
print('resting', mean_rest, std_rest)
print('familiar', mean_fam, std_fam)
print('medical', mean_med, std_med)
print('_____________________')
    #print(average_resting.get_data(picks[0]))
for i in range(len(picks)):
    
    skewness_rest = scipy.stats.skew(average_resting.get_data(picks[i])[0])
    kurtosis_rest = scipy.stats.kurtosis(average_resting.get_data(picks[i])[0])
    kstat_rest, kpval_rest = scipy.stats.kurtosistest(average_resting.get_data(picks[i])[0])
    print('resting for chan ',picks[i], ' skewness: ', skewness_rest, ' kurtosis: ', kurtosis_rest, ' pval for kurtosistest: ', kpval_rest)
    
    skewness_fam = scipy.stats.skew(average_familiar.get_data(picks[i])[0])
    kurtosis_fam = scipy.stats.kurtosis(average_familiar.get_data(picks[i])[0])
    kstat_fam, kpval_fam = scipy.stats.kurtosistest(average_familiar.get_data(picks[i])[0])
    print('familiar for chan ',picks[i], ' skewness: ', skewness_fam, ' kurtosis: ', kurtosis_fam, ' pval for kurtosistest: ', kpval_fam)
    
    skewness_med = scipy.stats.skew(average_medical.get_data(picks[i])[0])
    kurtosis_med = scipy.stats.kurtosis(average_medical.get_data(picks[i])[0])
    kstat_med, kpval_med = scipy.stats.kurtosistest(average_medical.get_data(picks[i])[0])
    print('medical for chan ',picks[i], ' skewness: ', skewness_med, ' kurtosis: ', kurtosis_med, ' pval for kurtosistest: ', kpval_med)



"""
#normalized #made no difference
skewness_rest_norm = scipy.stats.skew((average_resting_chan-np.mean(average_resting.get_data()))/np.std(average_resting.get_data()))
kurtosis_rest_norm = scipy.stats.kurtosis((average_resting_chan-np.mean(average_resting.get_data()))/np.std(average_resting.get_data()))
print('skewness resting_norm: (normal dist = 0) ', skewness_rest_norm)
print('kurtosis resting_norm: (normal dist = 0)', kurtosis_rest_norm)
"""
"""
average_familiar_chan = np.mean(average_familiar.get_data(),axis=0)
average_familiar_chan_std = np.std(average_familiar.get_data(),axis=0)
skewness_familiar = scipy.stats.skew(average_familiar_chan)
kurtosis_familiar = scipy.stats.kurtosis(average_familiar_chan)
kstat_fam, kpval_fam = scipy.stats.kurtosistest(average_familiar_chan)

average_medical_chan = np.mean(average_medical.get_data(),axis=0)
average_medical_chan_std = np.std(average_medical.get_data(),axis=0)
skewness_medical = scipy.stats.skew(average_medical_chan)
kurtosis_medical = scipy.stats.kurtosis(average_medical_chan)
kstat_med, kpval_med = scipy.stats.kurtosistest(average_medical_chan)

print('skewness resting: (normal dist = 0) ', skewness_rest)
print('kurtosis resting: (normal dist = 0)', kurtosis_rest, ',pval for kurtosis-test', kpval_rest)
print('skewness familiar: (normal dist = 0) ', skewness_familiar)
print('kurtosis familiar: (normal dist = 0)', kurtosis_familiar, ',pval for kurtosis-test', kpval_fam)
print('skewness medical: (normal dist = 0) ', skewness_medical)
print('kurtosis medical: (normal dist = 0)', kurtosis_medical, ',pval for kurtosis-test', kpval_med)
"""

#plotting
plt.hist(average_medical.get_data('Cz')[0])
plt.suptitle('Medical for Cz channel')
plt.xlabel('Voltage')
plt.ylabel('Count')
"""
x = np.linspace(0,0.5,176)
fig, ax = plt.subplots()
ax.plot(x, average_medical_chan, '-')
ax.fill_between(x, average_medical_chan-average_medical_chan_std,average_medical_chan+average_medical_chan_std, alpha=0.2)
ax.set_title('Medical person 2')
"""
plt.show()
