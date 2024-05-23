import mne 
import os
import matplotlib.pyplot as plt
from read_files import LoadFile, LoadAllRaw
import numpy as np
import scipy

'''
#picks = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'T9', 'T10', 'Fz', 'Cz', 'Pz', 'F10', 'F9', 'P9', 'P10', 'ECG EKG', 'Photic', 'Pulse Rate', 'IBI', 'Bursts', 'Suppr']
path = 'EDF filer/'
fname = '02IT-EDF+.edf'
picks = 'ECG EKG-REF'

filename = path + fname
raw = mne.io.read_raw_edf(filename, verbose=False, preload=True, include='Pulse Rate')
print(raw.get_data())

raw = mne.io.read_raw_edf(filename, verbose=False, preload=True, include=picks)
channels = raw.info.ch_names
ecg_data = raw.get_data()

# Normalizing
ecg_n = (ecg_data[0] - ecg_data[0].mean()) / ecg_data[0].std()

# Creating sine filter
v = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
peak_filter = np.sin(v)

# Cross correlation between ECG and filter (convolution)
ecg_transformed = np.correlate(ecg_n, peak_filter, mode='same')
similarity = ecg_transformed / np.max(ecg_transformed)

# Finding RR peaks
rr_peaks, _ = scipy.signal.find_peaks(similarity, height=0.4)
rr_ecg = np.diff(rr_peaks)


y = np.linspace(0, len(ecg_data[0])*4/1000, len(ecg_data[0]))
fig, = plt.plot(y, ecg_data[0], alpha=1)
#plt.xlim(0,10)
plt.plot(y, ecg_transformed, alpha=0.8, c='orange')
plt.gca().legend(('Raw', 'Filtered'))
plt.title('ECG')
plt.xlabel('Time (s)')

y = np.linspace(0, len(similarity)*4, len(similarity))
fig_rr, = plt.plot(y,similarity, alpha=0.8)
plt.scatter(rr_peaks*4, similarity[rr_peaks], color='red')
plt.xlim(0,8000)
plt.title('ECG with RR peaks')
plt.xlabel('Time (s)')
#plt.show()

# Dealing with outliers + interpolation function for smoothing

# compute the diff along the time axis to end up with the R-R intervals
rr_ecg = np.diff(rr_peaks*4)

# fit function to the dataset
x_ecg = np.cumsum(rr_ecg)/1000 
f_ecg = scipy.interpolate.interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')

# sample rate for interpolation
fs = 4
steps = 1 / fs

# sample using the interpolation function
xx_ecg = np.arange(0, np.max(x_ecg), steps)
rr_interpolated_ecg = f_ecg(xx_ecg)

# Removing peaks greater than 2 from the below (above or under)
rr_ecg[np.abs(scipy.stats.zscore(rr_ecg)) > 2] = np.median(rr_ecg)

x_ecg = np.cumsum(rr_ecg)/1000
f_ecg = scipy.interpolate.interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')

xx_ecg = np.arange(0, np.max(x_ecg), steps)
clean_rr_interpolated_ecg = f_ecg(xx_ecg)

plt.figure(figsize=(25,5))
plt.title('Error using z-score')
plt.plot(rr_interpolated_ecg)
plt.plot(clean_rr_interpolated_ecg)
plt.xlabel('Time (s)')
plt.gca().legend(('Interpolated', 'Cleaned'))
plt.ylabel('RR-interval (ms)')
'''
#plt.show()



# TIME DOMAIN 
# ref: https://bartek-kulas.medium.com/working-with-ecg-heart-rate-data-on-python-7a45fa880d48

# STD RR/SDNN: Often calculated over a 24-hour period. SDNN reflects all the cyclic components responsible for variability, therefore it represents total variability
# RMSSD: square root of the mean of the squares of the successive differences between adjacent NNs
# STD: The standard deviation of the successive differences between adjacent NNs
# NNxx: the number of pairs of successive NNs that differ by more than xx ms (we used 50 ms in our example)
# pNNxx: the proportion of NNxx divided by total number of NNs

def ECG_features(folder, path):
    picks = 'ECG EKG-REF'
    all_time_domain = {}
    fxx_freq_domain = [ [] for _ in range(81)]
    pxx_freq_domain = [ [] for _ in range(81)]

    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        print(filename)
        fname = path+filename
        print(fname)
        person_number = filename[:2]
        if person_number[0] == 0:
            person_number = person_number[1]
        pn = int(person_number)

        raw = mne.io.read_raw_edf(fname, verbose=False, preload=True, include=picks)
        channels = raw.info.ch_names
        ecg_data = raw.get_data()

        # Normalizing
        ecg_n = (ecg_data[0] - ecg_data[0].mean()) / ecg_data[0].std()

        # Creating sine filter
        v = np.linspace(1.5 * np.pi, 3.5 * np.pi, 15)
        peak_filter = np.sin(v)

        # Cross correlation between ECG and filter (convolution)
        ecg_transformed = np.correlate(ecg_n, peak_filter, mode='same')
        similarity = ecg_transformed / np.max(ecg_transformed)

        # Finding RR peaks
        rr_peaks, _ = scipy.signal.find_peaks(similarity, height=0.4)
        rr_ecg = np.diff(rr_peaks)

        # Dealing with outliers + interpolation function for smoothing

        # compute the diff along the time axis to end up with the R-R intervals
        rr_ecg = np.diff(rr_peaks*4)

        all_time_domain[pn] = rr_ecg

        # fit function to the dataset
        x_ecg = np.cumsum(rr_ecg)/1000 

        print(len(x_ecg))
        print(len(rr_ecg))

        '''
        if len(x_ecg) != 0:

            f_ecg = scipy.interpolate.interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')

            # sample rate for interpolation
            fs = 4
            steps = 1 / fs

            # sample using the interpolation function
            xx_ecg = np.arange(0, np.max(x_ecg), steps)
            rr_interpolated_ecg = f_ecg(xx_ecg)

            # Removing peaks greater than 2 from the below (above or under)
            rr_ecg[np.abs(scipy.stats.zscore(rr_ecg)) > 2] = np.median(rr_ecg)

            x_ecg = np.cumsum(rr_ecg)/1000
            f_ecg = scipy.interpolate.interp1d(x_ecg, rr_ecg, kind='cubic', fill_value= 'extrapolate')

            xx_ecg = np.arange(0, np.max(x_ecg), steps)
            clean_rr_interpolated_ecg = f_ecg(xx_ecg)

            fxx, pxx = scipy.signal.welch(x=clean_rr_interpolated_ecg, fs=4, nperseg=256)

            fxx_freq_domain[pn].append(fxx)
            pxx_freq_domain[pn].append(pxx)
        '''
    
    return all_time_domain, fxx_freq_domain, pxx_freq_domain


path ="EDF_test/"
folder = os.fsencode(path)

all_time_domain, fxx_freq_domain, pxx_freq_domain = ECG_features(folder=folder, path=path)

rr_p2 = all_time_domain[2]
fxx_p2 = fxx_freq_domain[2]
pxx_p2 = pxx_freq_domain[2]


def timedomain(rr):
    results = {}
    rr = rr
    hr = 60000/rr
    
    # HRV metrics
    results['Mean RR (ms)'] = np.mean(rr)
    results['STD RR/SDNN (ms)'] = np.std(rr)
    #results['Mean HR (Kubios\' style) (beats/min)'] = 60000/np.mean(rr)
    results['Mean HR (beats/min)'] = np.mean(hr)
    results['STD HR (beats/min)'] = np.std(hr)
    results['Min HR (beats/min)'] = np.min(hr)
    results['Max HR (beats/min)'] = np.max(hr)
    #results['RMSSD (ms)'] = np.sqrt(np.mean(np.square(np.diff(rr))))
    #results['NN50'] = np.sum(np.abs(np.diff(rr)) > 50)*1
    #results['pNN50 (%)'] = 100 * np.sum((np.abs(np.diff(rr)) > 50)*1) / len(rr)
    return results

print("Time domain metrics:")
for k, v in timedomain(rr_p2).items():
    print("- %s: %.2f" % (k, v))



"""
fxx, pxx = scipy.signal.welch(x=clean_rr_interpolated_ecg, fs=4, nperseg=256)

# fit a function for plotting bands
powerspectrum_f = scipy.interpolate.interp1d(fxx, pxx, kind='cubic', fill_value= 'extrapolate')

plt.figure(figsize=(15,6))
plt.title("FFT Spectrum (Welch's periodogram)")

# setup frequency bands for plotting
x_VLF = np.linspace(0, 0.04, 100)
x_LF = np.linspace(0.04, 0.15, 100)
x_HF = np.linspace(0.15, 0.4, 100)

plt.gca().fill_between(x_VLF, powerspectrum_f(x_VLF), alpha=0.5, color="#F5866F", label="VLF")
plt.gca().fill_between(x_LF, powerspectrum_f(x_LF), alpha=0.5, color="#51A6D8", label="LF")
plt.gca().fill_between(x_HF, powerspectrum_f(x_HF), alpha=0.5, color="#ABF31F", label="HF")

plt.gca().set_xlim(0, 0.75)
plt.gca().set_ylim(0)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Density")

plt.legend()
plt.show()
"""

def freq_domain(fxx, pxx):

    #frequency bands: very low frequency (VLF), low frequency (LF), high frequency (HF) 
    cond_VLF = (fxx >= 0) & (fxx < 0.04)
    cond_LF = (fxx >= 0.04) & (fxx < 0.15)
    cond_HF = (fxx >= 0.15) & (fxx < 0.4)

    #calculate power in each band by integrating the spectral density using trapezoidal rule 
    VLF = scipy.integrate.trapezoid(pxx[cond_VLF], fxx[cond_VLF])
    LF = scipy.integrate.trapezoid(pxx[cond_LF], fxx[cond_LF])
    HF = scipy.integrate.trapezoid(pxx[cond_HF], fxx[cond_HF])

    #total power sum
    total_power = VLF + LF + HF

    # calculate power in each band by integrating the spectral density 
    vlf = scipy.integrate.trapezoid(pxx[cond_VLF], fxx[cond_VLF])
    lf = scipy.integrate.trapezoid(pxx[cond_LF], fxx[cond_LF])
    hf = scipy.integrate.trapezoid(pxx[cond_HF], fxx[cond_HF])


    #peaks (Hz) in each band
    peak_VLF = fxx[cond_VLF][np.argmax(pxx[cond_VLF])]
    peak_LF = fxx[cond_LF][np.argmax(pxx[cond_LF])]
    peak_HF = fxx[cond_HF][np.argmax(pxx[cond_HF])]

    #fractions
    LF_nu = 100 * lf / (lf + hf)
    HF_nu = 100 * hf / (lf + hf)

    results = {}
    results['Power VLF (ms2)'] = VLF
    results['Power LF (ms2)'] = LF
    results['Power HF (ms2)'] = HF   
    results['Power Total (ms2)'] = total_power

    results['LF/HF'] = (LF/HF)
    results['Peak VLF (Hz)'] = peak_VLF
    results['Peak LF (Hz)'] = peak_LF
    results['Peak HF (Hz)'] = peak_HF

    #results['Fraction LF (nu)'] = LF_nu
    #results['Fraction HF (nu)'] = HF_nu

    return results

'''
print("Freq domain metrics:")
for k, v in freq_domain(fxx_p2, pxx_p2).items():
    print("- %s: %.2f" % (k, v))
'''
# Another ref: https://www.kaggle.com/code/stetelepta/exploring-heart-rate-variability-using-python
