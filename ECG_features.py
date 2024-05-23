import mne 
import os
import numpy as np
import neurokit2 as nk

def hrv_data(r):
    
    """
    Input r: 1 epoch
    """
    picks = 'ECG EKG'
    ecg_data = r[0][0].get_data(picks=picks)[0]
    ecg_clean = nk.ecg_clean(ecg_data[0], sampling_rate=250)
    r_peaks = nk.ecg_findpeaks(ecg_clean,sampling_rate=250, show=False)
    #hrv_indices = nk.hrv(r_peaks, sampling_rate=250, show=False)
    hrv = nk.hrv_time(r_peaks, sampling_rate=250, show=False)
    signals, info = nk.ecg_process(ecg_data[0], sampling_rate=250)
    #hrv_freq = nk.hrv_frequency(signals["ECG_R_Peaks"], sampling_rate=250, show=False, normalize=False)
    results = [hrv['HRV_MeanNN'][0], hrv['HRV_SDNN'][0], np.mean(signals['ECG_Rate']),np.std(signals['ECG_Rate']), np.min(signals['ECG_Rate']), np.max(signals['ECG_Rate'])]# ,  hrv_freq['HRV_VLF'][0],hrv_freq['HRV_LF'][0], hrv_freq['HRV_HF'][0],hrv_freq['HRV_VHF'][0],hrv_freq['HRV_TP'][0],hrv_freq['HRV_LFHF'][0]
    return results
