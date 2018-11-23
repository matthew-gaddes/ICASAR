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

from ICASAR_functions import *

def col_to_ma(col, pixel_mask):
    """ A function to take a column vector and a 2d pixel mask and reshape the column into a masked array.  
    Useful when converting between vectors used by BSS methods results that are to be plotted
    Inputs:
        col | rank 1 array | 
        pixel_mask | array mask (rank 2)
    Outputs:
        source | rank 2 masked array | colun as a masked 2d array
    """
    import numpy.ma as ma 
    import numpy as np
    
    source = ma.array(np.zeros(pixel_mask.shape), mask = pixel_mask )
    source.unshare_mask()
    source[~source.mask] = col.ravel()   
    return source

#%% Things to set

n_comp = 5
bootstrapping_param = (200, 0)
hdbscan_param = (35, 10)                        # (min_cluster_size, min_samples)
tsne_param = (30, 12)                       # (perplexity, early_exaggeration)
ica_param = (1e-2, 150)                     # (tolerance, max iterations)


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


fig2, axes = plt.subplots(2,5)                                    # plot the synthetic interferograms
for i, ax in enumerate(np.ravel(axes[:])):
    ax.imshow(col_to_ma(phUnw[i,:], pixel_mask))
fig2.suptitle('Mixtures (intererograms)')

#%% do ICA with ICSAR function
 
S_best,  time_courses, x_train_residual_ts, Iq, n_clusters, S_all_info  = ICASAR(phUnw, bootstrapping_param, pixel_mask, n_comp, 
                                             ica_param = ica_param, tsne_param = tsne_param, hdbscan_param = hdbscan_param)
      

    
#%% Figure from the manual

fig1, axes = plt.subplots(1,3, figsize = (11,4))
axes[0].imshow(X_dc, aspect = 500)
axes[0].set_title('Data matix')
axes[1].imshow(pixel_mask)
axes[1].set_title('Mask')
axes[2].imshow(col_to_ma(X_dc[0,:], pixel_mask))
axes[2].set_title('Interferogram 1')
fig1.savefig('manual/figure_1.png', bbox_inches='tight')


#%%



fig1, axes = plt.subplots(2,3)                                  # plot he synthetic sources
for i in range(3):
    axes[0,i].imshow(col_to_ma(S_synth[i,:], pixel_mask))
    axes[1,i].plot(range(A_dc.shape[0]), A_dc[:,i])
    axes[1,i].axhline(0)
fig1.suptitle('Synthetic sources and time courses')


fig2, axes = plt.subplots(2,5)                                    # plot the synthetic interferograms
for i, ax in enumerate(np.ravel(axes[:])):
    ax.imshow(col_to_ma(X_dc[i,:], pixel_mask))
fig2.suptitle('Mixtures (intererograms)')

