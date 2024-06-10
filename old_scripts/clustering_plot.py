import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import sklearn
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import os
import pandas as pd
import random
import matplotlib.pyplot as plt
import mne
import neurokit2 as nk
from sklearn.cluster import HDBSCAN, OPTICS, DBSCAN
from itertools import cycle, islice
import matplotlib.patches as mpatches
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import warnings
from sklego.meta import OrdinalClassifier
import seaborn as sns

random.seed(8)

path ='CSV_individual_cross_val/'
folder = os.fsencode(path)

def DataCreate(path, folder):
    renamed_markers_list = ['No.', 'Event', 'Delta', 'Delta_N', 'Theta', 'Theta_N', 'Alpha', 'Alpha_N', 'Beta', 'Beta_N', 'Gamma', 'Gamma_N', 'SE', 'MSF', 'Sef90', 'Sef95', 'PE', 'wSMI', 'Kolmogorov', 'MeanRR', 'StdRR', 'MeanHR', 'StdHR', 'MinHR', 'MaxHR', 'freq_slope_mean','freq_slope_std'] 
    m_list = ['Delta', 'Delta_N', 'Theta', 'Theta_N', 'Alpha', 'Alpha_N', 'Beta', 'Beta_N', 'Gamma', 'Gamma_N', 'SE', 'MSF', 'Sef90', 'Sef95', 'PE', 'wSMI', 'Kolmogorov', 'MeanRR', 'StdRR', 'MeanHR', 'StdHR', 'MinHR', 'MaxHR', 'freq_slope_mean','freq_slope_std'] 
    mean_list = [i + '_mean' for i in m_list]
    std_list = [i + '_std' for i in m_list]

    r_data = []
    m_data = []
    f_data = []
    all_data = []

    patient_numbers = []
    for file in os.listdir(folder):
        filename = os.fsdecode(file)
        patient_numbers.append(filename[0:3])

        fname = path + filename
        data = pd.read_csv(fname)
        data.columns = renamed_markers_list
        all_data.append(data)
        R_indicesToKeep = data['Event'] == 'R'
        df_r = data.loc[R_indicesToKeep]

        M_indicesToKeep = data['Event'] == 'M'
        df_m = data.loc[M_indicesToKeep]

        F_indicesToKeep = data['Event'] == 'F'
        df_f = data.loc[F_indicesToKeep]
    

        r_mean = pd.DataFrame(df_r[m_list].mean(axis=0)).transpose()
        m_mean = pd.DataFrame(df_m[m_list].mean(axis=0)).transpose()
        f_mean = pd.DataFrame(df_f[m_list].mean(axis=0)).transpose()

        r_mean.columns = mean_list
        m_mean.columns = mean_list
        f_mean.columns = mean_list

        r_std = pd.DataFrame(df_r[m_list].std(axis=0)).transpose()
        m_std = pd.DataFrame(df_m[m_list].std(axis=0)).transpose()
        f_std = pd.DataFrame(df_f[m_list].std(axis=0)).transpose()
            
        r_std.columns = std_list
        m_std.columns = std_list
        f_std.columns = std_list
        
        rest = pd.concat([r_mean,r_std],axis=1).to_numpy()
        med = pd.concat([m_mean,m_std],axis=1).to_numpy()
        fam = pd.concat([f_mean,f_std],axis=1).to_numpy()

        r_data.append(rest[0])
        m_data.append(med[0])
        f_data.append(fam[0])
    X = [f_data[i] - r_data[i] for i in range(len(r_data))] + [m_data[i] - r_data[i] for i in range(len(r_data))]
    X = sklearn.preprocessing.StandardScaler().fit(X).transform(X)
    y = [i for i in range(0,len(r_data))] + [i for i in range(0,len(r_data))]
    return X, y, patient_numbers, all_data


def clusteringPlots(path, folder, chosen_features, plot=True):
    X, y, patient_numbers, all_data = DataCreate(path, folder) 
    m_list = ['Delta', 'Delta_N', 'Theta', 'Theta_N', 'Alpha', 'Alpha_N', 'Beta', 'Beta_N', 'Gamma', 'Gamma_N', 'SE', 'MSF', 'Sef90', 'Sef95', 'PE', 'wSMI', 'Kolmogorov', 'MeanRR', 'StdRR', 'MeanHR', 'StdHR', 'MinHR', 'MaxHR', 'freq_slope_mean','freq_slope_std'] 
    mean_list = [i + '_mean' for i in m_list]
    std_list = [i + '_std' for i in m_list]
    b= mean_list+std_list
    ind = [True if b[i] in chosen_features else False for i in range(len(b))]
    X = X[:,ind]   
    
    sclustering = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0).fit(X)
    sy_pred = sclustering.labels_
    """
    sclustering = AgglomerativeClustering(linkage='single').fit(X)
    sy_pred = sclustering.labels_
    """
    patient_numbers2 = patient_numbers+patient_numbers
    
    
    hdb = HDBSCAN(min_cluster_size=5).fit(X)
    hy_pred = hdb.labels_
    patient_numbers2 = patient_numbers+patient_numbers

    dclustering = DBSCAN(min_samples=5, eps=1.5).fit(X)
    dy_pred = dclustering.labels_    
    
    oclustering = OPTICS(min_samples=5).fit(X)
    oy_pred = oclustering.labels_
        
    fig, ax = plt.subplots(2,2, figsize=(10,10))
    
    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(sy_pred) + 1),
            )
        )
    )
    # add black color for outliers (if any)
    colors = np.append(colors, ["#000000"])



    ax[0,0].scatter(X[:, 0], X[:, -2], c=colors[sy_pred])
    for i, txt in enumerate(y):
        ax[0,0].annotate(patient_numbers[txt], (X[i, 0], X[i, -2]))
    ax[0,0].set_aspect('equal', adjustable=None, anchor=None, share=False)
    ax[0,0].set_title('Spectral clustering')
    #ax[0,0].legend()

    ax[0,1].scatter(X[:, 0], X[:, -2], c=colors[oy_pred])
    for i, txt in enumerate(y):
        ax[0,1].annotate(patient_numbers[txt], (X[i, 0], X[i, -2]))
    ax[0,1].set_aspect('equal', adjustable=None, anchor=None, share=False)
    ax[0,1].set_title('OPTICS')
    #ax[0,1].legend()


    ax[1,0].scatter(X[:, 0], X[:, -2], c=colors[hy_pred])
    for i, txt in enumerate(y):
        ax[1,0].annotate(patient_numbers[txt], (X[i, 0], X[i, -2]))
    ax[1,0].set_aspect('equal', adjustable=None, anchor=None, share=False)
    ax[1,0].set_title('HDBSCAN')
    #ax[1,0].legend()

    ax[1,1].scatter(X[:, 0], X[:, -2], c=colors[dy_pred])
    for i, txt in enumerate(y):
        ax[1,1].annotate(patient_numbers[txt], (X[i, 0], X[i, -2]))
    ax[1,1].set_aspect('equal', adjustable=None, anchor=None, share=False)
    ax[1,1].set_title('DBSCAN')
    #ax[1,1].legend()
    pop_a = mpatches.Patch(color="#377eb8", label='Cluster 1') 
    pop_b = mpatches.Patch(color="#ff7f00", label='Cluster 2') 
    pop_c = mpatches.Patch(color="#000000", label='Outliers')
    fig.legend(handles=[pop_a,pop_b,pop_c])    

    #ax.set_xticks(np.arange(-6,25,2)) 
    plt.show()
    return sy_pred, oy_pred, hy_pred, dy_pred

#print(b)
#print(ind)

def featureSelection(path, folder):
    m_list = ['Delta', 'Delta_N', 'Theta', 'Theta_N', 'Alpha', 'Alpha_N', 'Beta', 'Beta_N', 'Gamma', 'Gamma_N', 'SE', 'MSF', 'Sef90', 'Sef95', 'PE', 'wSMI', 'Kolmogorov', 'MeanRR', 'StdRR', 'MeanHR', 'StdHR', 'MinHR', 'MaxHR', 'freq_slope_mean','freq_slope_std'] 

    X, y, patient_numbers, all_data = DataCreate(path, folder) 
    t = pd.CategoricalDtype(categories = ['R', 'M', 'F'], ordered = True)

    scores = []
    chosen_features_list = []
    subsets_list = []
    for i in range(len(all_data)):
        train_y = pd.Series(all_data[i]['Event'],dtype=t).cat.codes
        train_X = all_data[i][m_list]
        train_X = pd.DataFrame(sklearn.preprocessing.StandardScaler().fit(train_X).transform(train_X), columns=m_list)
        ord_clf = OrdinalClassifier(estimator=LogisticRegression())
        sfs = SFS(ord_clf, k_features=2, forward=True, floating=False,verbose=0, cv=3, scoring='accuracy') #cv = number of splits in crossval, automaticly it has stratifies kfold
        sfs.fit(train_X, train_y)
        score = sfs.k_score_
        features = sfs.k_feature_names_
        scores.append(score)
        chosen_features_list.append(features)
        #print(features)
        #print(score)
        subsets = sfs.subsets_
        subsets_list.append(subsets)
        #print(subsets)
    good = np.where(np.asarray(scores)>=0.70)[0]

    good_patients = [(patient_numbers[i], scores[i]) for i in good]

    majority = []
    for i in good:
        majority.append(chosen_features_list[i])
    major = list(zip(*np.unique(majority, return_counts=True)))
    major_sorted = sorted(major, key=lambda tup: tup[1], reverse=True)

    chosen_features = []
    for x,y in major_sorted:
        if y>1:
            chosen_features.append(x+'_mean')
            chosen_features.append(x+'_std')
    return chosen_features, good_patients

#chosen_features, good_patients = featureSelection(path,folder)
#print(chosen_features, good_patients)

chosen_features = ['Delta_mean', 'Delta_std','Theta_mean', 'Theta_std', 'Alpha_mean', 'Alpha_std', 'Beta_mean', 'Beta_std',  'PE_mean', 'PE_std'] 
sy_pred, oy_pred, hy_pred, dy_pred = clusteringPlots(path, folder, chosen_features = chosen_features, plot=True)
X, y, patient_numbers, all_data = DataCreate(path, folder)
m_list = ['Delta', 'Delta_N', 'Theta', 'Theta_N', 'Alpha', 'Alpha_N', 'Beta', 'Beta_N', 'Gamma', 'Gamma_N', 'SE', 'MSF', 'Sef90', 'Sef95', 'PE', 'wSMI', 'Kolmogorov', 'MeanRR', 'StdRR', 'MeanHR', 'StdHR', 'MinHR', 'MaxHR', 'freq_slope_mean','freq_slope_std'] 
mean_list = [i + '_mean' for i in m_list]
std_list = [i + '_std' for i in m_list]
b= mean_list+std_list
ind = [True if b[i] in chosen_features else False for i in range(len(b))]
X = X[:,ind] 
patient_numbers = patient_numbers+patient_numbers
X = pd.DataFrame(X, columns=chosen_features)
X.insert(0,'Group', patient_numbers)
for i in range(len(X['Group'])):
    if X['Group'][i] in ['p14', 'p31', 'p32','p34', 'p35', 'p41', 'p44', 'p58', 'p60', 'p65', 'p75']:
        X.loc[i,'Group']  = 'Responsive'
    elif X['Group'][i] == 'p76':
        X.loc[i,'Group'] = 'p76'
    else:
        X.loc[i,'Group'] = 'Unresponsive'

sns.pairplot(X, hue='Group')
plt.show()

"""print(sy_pred, len(sy_pred))
print(oy_pred, len(oy_pred))
print(hy_pred, len(hy_pred))
print(dy_pred, len(dy_pred))"""
