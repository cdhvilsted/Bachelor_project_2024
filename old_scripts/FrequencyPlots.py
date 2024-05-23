import mne 
import numpy as np
import matplotlib.pyplot as plt
import os
from nice.examples.LoadFile import LoadFile
import scipy
import seaborn as sns

'''
fname = "EDF filer/01CX-EDF+1.edf"

raw = mne.io.read_raw_edf(fname)

annot = raw.annotations
annot_dict = raw.annotations.count()

count = 0
for i in annot.description:
    if i == 'Resting':
        begin_rest = annot.onset[count]
        duration_rest = annot.duration[count]
        end_rest = begin_rest + duration_rest
    count += 1

print("--------------------")

#print(np.shape(raw.compute_psd(tmax=end_rest, tmin=begin_rest, fmin=4, fmax=7, average=False).get_data())) # (31 channels, 25 freqs, 73 segments)

#Finding channels
channels = raw.info.ch_names
picks = channels[0:25]

# Blot baseline frequency bands
fig, axes = plt.subplots(nrows=5, ncols=1, sharey=True)

fig_delta = raw.compute_psd(tmax=end_rest, tmin=begin_rest, fmin=0, fmax=4, average=False, picks=picks)
fig_theta = raw.compute_psd(tmax=end_rest, tmin=begin_rest, fmin=4, fmax=6, average=False, picks=picks)
fig_alpha = raw.compute_psd(tmax=end_rest, tmin=begin_rest, fmin=8, fmax=12, average=False, picks=picks)
fig_beta = raw.compute_psd(tmax=end_rest, tmin=begin_rest, fmin=12, fmax=30, average=False, picks=picks)
fig_gamma = raw.compute_psd(tmax=end_rest, tmin=begin_rest, fmin=30, fmax=np.inf, average=False, picks=picks)

fig1 = fig_delta.plot(axes=axes[0], show=False)
fig2 = fig_theta.plot(axes=axes[1], show=False)
fig3 = fig_alpha.plot(axes=axes[2], show=False)
fig4 = fig_beta.plot(axes=axes[3], show=False)
fig5 = fig_gamma.plot(axes=axes[4], show=False)

fig1.axes[0].set_title("Delta") # access the axes object and set the title
fig2.axes[1].set_title("Theta") 
fig3.axes[2].set_title("Alpha")
fig4.axes[3].set_title("Beta")
fig5.axes[4].set_title("Gamma")

#fig.tight_layout(pad=0.15)
plt.subplots_adjust(hspace=0.5)

plt.show()
'''

def AnnotationTimes(raw, annot, annot_type):
    count = 0
    liste_annot = []
    #if annot_type not in annot_dict:
    #    print("Annotation type not in file.")
    #    return None
    for i in annot.description:
        if i == annot_type:
            begin = annot.onset[count]
            duration = annot.duration[count]
            end = begin + duration
            liste_annot.append((begin, duration, end))
        count += 1
    
    return liste_annot

def FrequencyPlot(raw, liste_annot, annot_type):
    #Finding channels
    channels = raw.info.ch_names
    picks = channels[0:25]

    # Blot baseline frequency bands
    fig, axes = plt.subplots(nrows=5, ncols=1, sharey=True)

    fig_delta = raw.compute_psd(tmax=liste_annot[-1], tmin=liste_annot[0], fmin=0, fmax=4, average=False, picks=picks)
    fig_theta = raw.compute_psd(tmax=liste_annot[-1], tmin=liste_annot[0], fmin=4, fmax=6, average=False, picks=picks)
    fig_alpha = raw.compute_psd(tmax=liste_annot[-1], tmin=liste_annot[0], fmin=8, fmax=12, average=False, picks=picks)
    fig_beta = raw.compute_psd(tmax=liste_annot[-1], tmin=liste_annot[0], fmin=12, fmax=30, average=False, picks=picks)
    fig_gamma = raw.compute_psd(tmax=liste_annot[-1], tmin=liste_annot[0], fmin=30, fmax=np.inf, average=False, picks=picks)

    fig1 = fig_delta.plot(axes=axes[0], show=False)
    axes[0].set_ylabel('Power in dB')
    fig2 = fig_theta.plot(axes=axes[1], show=False)
    
    axes[1].set_ylabel('Power in dB')
    fig3 = fig_alpha.plot(axes=axes[2], show=False)
   
    axes[2].set_ylabel('Power in dB')
    fig4 = fig_beta.plot(axes=axes[3], show=False)
    axes[3].set_ylabel('Power in dB')
    fig5 = fig_gamma.plot(axes=axes[4], show=False)
    axes[4].set_xlabel('Frequency in Hz')
    axes[4].set_ylabel('Power in dB')
    fig1.axes[0].set_title("Delta") # access the axes object and set the title
    fig2.axes[1].set_title("Theta") 
    fig3.axes[2].set_title("Alpha")
    fig4.axes[3].set_title("Beta")
    fig5.axes[4].set_title("Gamma")
    
    
    #fig.tight_layout(pad=0.15)
    plt.subplots_adjust(hspace=0.5)
    fig.suptitle(annot_type, fontsize='xx-large')

    plt.show()

def FrequencyPlotSpecific(raw, liste_annot, annot_type, fmin, fmax):
    #Finding channels
    channels = raw.info.ch_names
    #picks = ['EEG F3-REF','EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF','EEG Cz-REF', 'EEG Pz-REF']
    #picks = ['EEG O1-REF', 'EEG O2-REF']
    picks = channels

    # Blot baseline frequency bands

    fig = raw.compute_psd(tmax=liste_annot[-1], tmin=liste_annot[0], fmin=fmin, fmax=fmax, picks=picks)
    fig.plot(show=False,average=False)
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Power in dB')

    #fig.tight_layout(pad=0.15)
    #fig.suptitle(annot_type, fontsize='xx-large')

    plt.show()


def FrequencyPlotSpecificNOFREQ(raw, liste_annot, annot_type,picks):
    #Finding channels
    channels = raw.info.ch_names
    #picks = ['EEG F3-REF','EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF','EEG Cz-REF', 'EEG Pz-REF']
    #picks = ['EEG O1-REF', 'EEG O2-REF']

    #print(picks)
    # Blot baseline frequency bands

    fig = raw.compute_psd(tmax=liste_annot[-1], tmin=liste_annot[0], picks=picks)
    fig.plot(show=False,average=False)
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Power in dB')

    #fig.tight_layout(pad=0.15)
    #fig.suptitle(annot_type, fontsize='xx-large')

    plt.axvline(x=50)
    plt.axvline(x=1)

    plt.show()


#fname = "EDF filer/01CX-EDF+1.edf"

#raw, annot, annot_dict = LoadFile(fname)
#annot_type='Resting'
#fname = "EDF filer/01CX-EDF+1.edf"
path = 'EDF filer/'
picks = ['Cz','Pz','C3','C4','P3','P4']
#epochs, raw = LoadAllRaw(path=path, picks = picks)
#annot = raw.annotations
annot_type='Resting'

file = 'FuldEDF_test/02IT-EDF+.edf'
raw, annot, annot_dict = LoadFile(fname=file)

#raw.plot_psd(area_mode='range', tmax=10.0, show=False, average=True)

#liste_annot_1 = AnnotationTimes(raw = raw, annot = annot, annot_type=annot_type)
#liste_annot_2 = [liste_annot_1[0][2], 8, liste_annot_1[0][2]+8] #if looking after event, 8 sec
#FrequencyPlotSpecific(raw, liste_annot_1[0], annot_type=annot_type, fmin=0, fmax=45)
#FrequencyPlot(raw, liste_annot_1[0], annot_type=annot_type)
#FrequencyPlotSpecificNOFREQ(raw, liste_annot_1[0], annot_type=annot_type, picks=picks)
#FrequencyPlotSpecific(raw, liste_annot_1[0], annot_type=annot_type, fmin=0, fmax=50)