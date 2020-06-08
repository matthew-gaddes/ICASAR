# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:17:22 2018

@author: eemeg

An example of the ICASAR software with synthetic data

"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt 
import pickle                                    # used for opening synthetic data

from ICASAR_functions import ICASAR
from auxiliary_functions import col_to_ma


#%% Things to set

ICASAR_settings = {"n_comp" : 5,                                    # number of components to recover with ICA (ie the number of PCA sources to keep)
                    "bootstrapping_param" : (200, 000),               # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                    "tsne_param" : (30, 12),                        # (perplexity, early_exaggeration)
                    "ica_param" : (1e-2, 150),                      # (tolerance, max iterations)
                    "figures" : "png+window"}                       # if png, saved in a folder as .png.  If window, open as interactive matplotlib figures,
                                                                    # if 'png+window', both.  
                                                                    # default is "window" as 03_clustering_and_manifold is interactive.  
#%% Import the data 

with open('synthetic_data.pkl', 'rb') as f:
    A_dc = pickle.load(f)
    S_synth = pickle.load(f)
    N_dc = pickle.load(f)
    pixel_mask = pickle.load(f)
    

#%% Make synthetic time series and view it
    
X_dc = A_dc @ S_synth + N_dc                                                  # do the mixing 
phUnw = X_dc                                                                    # mixtures are the unwrapped phase


fig1, axes = plt.subplots(2,3)                                  # plot he synthetic sources
for i in range(3):
    axes[0,i].imshow(col_to_ma(S_synth[i,:], pixel_mask))
    axes[1,i].plot(range(A_dc.shape[0]), A_dc[:,i])
    axes[1,i].axhline(0)
fig1.suptitle('Synthetic sources and time courses')
fig1.canvas.set_window_title("Synthetic sources and time courses")


fig2, axes = plt.subplots(2,5)                                    # plot the synthetic interferograms
for i, ax in enumerate(np.ravel(axes[:])):
    ax.imshow(col_to_ma(phUnw[i,:], pixel_mask))
fig2.suptitle('Mixtures (intererograms)')
fig2.canvas.set_window_title("Mixtures (intererograms)")

fig3, axes = plt.subplots(1,3, figsize = (11,4))
axes[0].imshow(X_dc, aspect = 500)
axes[0].set_title('Data matix')
axes[1].imshow(pixel_mask)
axes[1].set_title('Mask')
axes[2].imshow(col_to_ma(X_dc[0,:], pixel_mask))
axes[2].set_title('Interferogram 1')
fig3.canvas.set_window_title("Interferograms as row vectors and a mask")

#%% do ICA with ICSAR function
 
S_best,  time_courses, x_train_residual_ts, Iq, n_clusters, S_all_info  = ICASAR(phUnw, pixel_mask, **ICASAR_settings) 
      

    




