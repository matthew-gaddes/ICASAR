# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:17:22 2018

@author: eemeg

An example of the ICASAR software with synthetic data

"""


#%% Imports

import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
import random

from ICASAR_functions import ICASAR
from auxiliary_functions import col_to_ma, r2_to_r3, plot_temporal_signals


#%% Things to set


n_samples = 2000
n_mixtures = 10
noise_strength = 0.1                                             # try 0.2

ICASAR_settings = {"n_comp" : 5,                                    # number of components to recover with ICA (ie the number of PCA sources to keep)
                    "bootstrapping_param" : (200, 0),               # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                    "tsne_param" : (30, 12),                        # (perplexity, early_exaggeration)
                    "ica_param" : (1e-2, 150),                      # (tolerance, max iterations)
                    "hdbscan_param" : (35,10),                      # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
                    "out_folder" : 'example_temporal_01_outputs',     # outputs will be saved here
                    "figures" : "png+window",                       # if png, saved in a folder as .png.  If window, open as interactive matplotlib figures,
                    "inset_axes_side" : {'x':0.3, 'y':0.1}}         # the size of thei inset axes created in the clustering and manifold figure.  
 


#%% Make synthetic time series and view it
np.random.seed(1)                                                                       # 1 is standard for examples

time = np.linspace(0, 8, n_samples)
signal_names = ['Sin', 'Square', 'Saw tooth', ' Triangular']
s1 = np.sin(2 * time)                                                                   # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))                                                          # Signal 2 : square signal
s3 = signal.sawtooth(5 * time)                                                          # Signal 3 : saw tooth signal
s4 = signal.sawtooth(10*time, 0.5)                                                      # Signal 4 : triangular wave signal (ie a symetrical tringular wave)
S = np.c_[s1, s2, s3, s4].T                                                             # combine, rows are variables, columns are number of times/samples.  
n_signals = np.size(S, axis = 0)                                                        # get the number of signals.  

n_signals_used = np.random.randint(2, n_signals + 1)                                    # choose a random number (but at least 2) of signals to use
signals_used_args = np.random.randint(0,n_signals, n_signals_used)                      # generate the first guess of which sources to use (and there be as many sources as previously chosen)
while len(np.unique(signals_used_args)) != len(signals_used_args):                      # check that there are no duplicates in the list.  
    signals_used_args = np.random.randint(0,n_signals, n_signals_used)                  # regenerate if there are duplicates.  


S_selected = S[signals_used_args,]                                                      # select the signals chosen
A = -0.5 + np.random.rand(n_mixtures, S_selected.shape[0])                              # make a mixing matrix
N = noise_strength * np.random.normal(size=(A@S_selected).shape)                        # Also make noise
X = A@S_selected + N                                                                    # do the mixing

plot_temporal_signals(S, title = 'Synthetic Signals', signal_names = signal_names)
plot_temporal_signals(S_selected, title = 'Synthetic signals used')
plot_temporal_signals(X, title = 'Mixtures')


#%% do ICA with ICSAR function
 
temporal_data = {'mixtures_r2' : X,
                 'xvals' : np.arange(0, X.shape[1])}

S_best, time_courses, x_train_residual_ts, Iq, n_clusters, S_all_info, phUnw_mean  = ICASAR(temporal_data = temporal_data, **ICASAR_settings) 
      
#%% Or we can do vanilla ICA

from sklearn.decomposition import FastICA

X_mean = np.repeat(np.mean(X, axis = 1)[:,np.newaxis], X.shape[1], axis = 1)                        # calculate the mean of each signal (row), and repeat to be teh same shape as S
X_mc = X - X_mean                                                                                   # do the mean centering.  

# Compute ICA
ica = FastICA(n_components= ICASAR_settings['n_comp'])
FastICA_S = ica.fit_transform(X.T)                                      # Reconstruct signals, note that FastICA algorithm wants the data as obs x variables (e.g. 2000x10)

plot_temporal_signals(FastICA_S.T, title = 'FastICA sources')

#%%


