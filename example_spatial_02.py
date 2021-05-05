# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:17:22 2018

@author: eemeg

An example of the ICASAR software with real data from LiCSBAS that contains a deformation signals that is not visible in a single 6/12 days Sentinel-1 interferogram.  

"""
#%% Imports

import numpy as np
import matplotlib.pyplot as plt   
from pathlib import Path
import sys

sys.path.append(str(Path('./lib/')))

from ICASAR_functions import ICASAR, LiCSBAS_to_ICASAR
from auxiliary_functions import visualise_ICASAR_inversion



#%% Things to set

ICASAR_settings = {"n_comp" : 5,                                         # number of components to recover with ICA (ie the number of PCA sources to keep)
                   "bootstrapping_param" : (200, 0),                    # (number of runs with bootstrapping, number of runs without bootstrapping)                    "hdbscan_param" : (35, 10),                        # (min_cluster_size, min_samples)
                   "tsne_param" : (30, 12),                             # (perplexity, early_exaggeration)
                   "ica_param" : (1e-2, 150),                           # (tolerance, max iterations)
                   "hdbscan_param" : (100,10),                           # (min_cluster_size, min_samples) Discussed in more detail in Mcinnes et al. (2017). min_cluster_size sets the smallest collection of points that can be considered a cluster. min_samples sets how conservative the clustering is. With larger values, more points will be considered noise. 
                   "out_folder" : Path('example_spatial_02_outputs'),   # outputs will be saved here
                   "create_all_ifgs_flag" : True,                       # small signals are hard for ICA to extact from time series, so make it easier by creating all possible long temporal baseline ifgs from the incremental data.  
                   "load_fastICA_results" : True,                      # If all the FastICA runs already exisit, setting this to True speeds up ICASAR as they don't need to be recomputed.  
                   "figures" : "png+window"}                            # if png, saved in a folder as .png.  If window, open as interactive matplotlib figures,
                                                                         # if 'png+window', both.  
                                                                         # default is "window" as 03_clustering_and_manifold is interactive.  
                                                                    
LiCSBAS_out_folder_campi_flegrei = Path('./campi_flegrei_LiCSBAS_example_data/')                               # the main results file from LICSBAS, which includes the incremental (between epoch or acquisition) deformation.                          


                                              
#%% Import the results of LiCSBAS

print(f"Opening the LiCSBAS .h5 file...", end = '')
displacement_r2, tbaseline_info = LiCSBAS_to_ICASAR(LiCSBAS_out_folder_campi_flegrei, figures=True)                             # open the h5 file produced by LiCSBAS, lons and lats are in geocode info and same resolution as the ifgs
print(f"Done.  ")


 

#%% do ICA with ICSAR function

spatial_data = {'mixtures_r2'    : displacement_r2['incremental'],
                'mask'           : displacement_r2['mask'],
                'ifg_dates'      : tbaseline_info['ifg_dates'],                           # in form YYYYMMDD_YYYYMMDD as a list of strings.  
                'dem'            : displacement_r2['dem'],
                'lons'           : displacement_r2['lons'],
                'lats'           : displacement_r2['lats']}
                

S_best, time_courses, x_train_residual_ts, Iq, n_clusters, S_all_info, phUnw_mean  = ICASAR(spatial_data = spatial_data, **ICASAR_settings) 


#%% We an also visualise how interferograms are fit using the learned components (ICs, contained in S_best)

visualise_ICASAR_inversion(displacement_r2['incremental'], S_best, time_courses, displacement_r2['mask'], n_data = 10)


#%%

