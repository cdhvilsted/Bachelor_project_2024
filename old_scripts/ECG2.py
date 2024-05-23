import mne 
import os
import matplotlib.pyplot as plt
#from read_files import LoadFile, LoadAllRaw
import numpy as np
import scipy
from split_based_on_resting import LoadEpochsP

path ="EDF_test/"
folder = os.fsencode(path)


r, m, f = LoadEpochsP(folder=folder, path=path, patient_number = 3)

r1 = r[1].get_data(picks='ECG EKG')[0][0]
#r[0].plot(picks='ECG EKG')
r1 *=10**3
x = np.linspace(0,len(r1)*4,len(r1))
# plot
print(r1)
plt.title("ECG signal - 250 Hz")
plt.plot(x, r1)
plt.xlabel('Time (milliseconds)')
plt.ylabel('Amplitude in uV')
plt.show()